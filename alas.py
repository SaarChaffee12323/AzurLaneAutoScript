import os
import re
import threading
import time
from datetime import datetime, timedelta

import inflection
from cached_property import cached_property

from module.base.decorator import del_cached_property
from module.config.config import AzurLaneConfig, TaskEnd
from module.config.deep import deep_get, deep_set
from module.exception import *
from module.logger import logger
from module.notify import handle_notify


class AzurLaneAutoScript:
    stop_event: threading.Event = None

    def __init__(self, config_name='alas'):
        logger.hr('Start', level=0)
        self.config_name = config_name
        # Skip first restart
        self.is_first_task = True
        # Failure count of tasks
        # Key: str, task name, value: int, failure count
        self.failure_record = {}
        # Circuit breaker state: {task_name: datetime when to re-enable}
        self._circuit_breakers = {}
        self._load_circuit_breakers()

    def _circuit_breaker_path(self):
        return f'./log/{self.config_name}_circuit_breaker.json'

    def _load_circuit_breakers(self):
        """Load circuit breaker state from disk."""
        import json
        path = self._circuit_breaker_path()
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
                self._circuit_breakers = {k: datetime.fromisoformat(v) for k, v in raw.items()}
            logger.info(f'Loaded {len(self._circuit_breakers)} circuit breaker(s)')

    def _save_circuit_breaker(self, task, until):
        """Add or update a circuit breaker entry and persist to disk."""
        self._circuit_breakers[task] = until
        self._write_circuit_breakers()

    def _write_circuit_breakers(self):
        """Persist current circuit breaker state to disk."""
        import json
        path = self._circuit_breaker_path()
        raw = {k: v.isoformat() for k, v in self._circuit_breakers.items()}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)

    def _trigger_dependents(self, completed_task):
        """After a task succeeds, trigger any tasks that depend on it.
        Dependency is specified via Scheduler.RunAfter in the task config.
        """
        for section, data in self.config.data.items():
            sched = data.get("Scheduler", {}) if isinstance(data, dict) else None
            if not sched:
                continue
            run_after = sched.get("RunAfter", "")
            if run_after and run_after.strip() == completed_task:
                logger.info(f"Dependency: `{section}` runs after `{completed_task}` — "
                            f"setting next_run to now")
                self.config.modified[f"{section}.Scheduler.NextRun"] = datetime.now().replace(microsecond=0)
                if self.config.auto_update:
                    self.config.update()

    def _check_circuit_breakers(self):
        """Re-enable any tasks whose circuit breaker cooldown has expired."""
        now = datetime.now()
        re_enabled = []
        for task, until in list(self._circuit_breakers.items()):
            if now >= until:
                try:
                    self.config.task_enable(task)
                    re_enabled.append(task)
                    del self._circuit_breakers[task]
                    logger.info(f'Circuit breaker: re-enabled task `{task}`')
                except Exception:
                    # Config may have changed, just remove the breaker
                    del self._circuit_breakers[task]

        if re_enabled:
            self._write_circuit_breakers()

    @cached_property
    def config(self):
        try:
            config = AzurLaneConfig(config_name=self.config_name)
            return config
        except RequestHumanTakeover:
            logger.critical('Request human takeover')
            exit(1)
        except Exception as e:
            logger.exception(e)
            exit(1)

    @cached_property
    def device(self):
        try:
            from module.device.device import Device
            device = Device(config=self.config)
            return device
        except RequestHumanTakeover:
            logger.critical('Request human takeover')
            exit(1)
        except Exception as e:
            logger.exception(e)
            exit(1)

    @cached_property
    def checker(self):
        try:
            from module.server_checker import ServerChecker
            checker = ServerChecker(server=self.config.Emulator_ServerName)
            return checker
        except Exception as e:
            logger.exception(e)
            exit(1)

    def run(self, command, skip_first_screenshot=False):
        try:
            if not skip_first_screenshot:
                self.device.screenshot()
            self.__getattribute__(command)()
            return True
        except TaskEnd:
            return True
        except GameNotRunningError as e:
            logger.warning(e)
            self.config.task_call('Restart')
            return False
        except (GameStuckError, GameTooManyClickError) as e:
            logger.error(e)
            self.save_error_log()
            logger.warning(f'Game stuck, {self.device.package} will be restarted in 10 seconds')
            logger.warning('If you are playing by hand, please stop Alas')
            self.config.task_call('Restart')
            self.device.sleep(10)
            return False
        except GameBugError as e:
            logger.warning(e)
            self.save_error_log()
            logger.warning('An error has occurred in Azur Lane game client, Alas is unable to handle')
            logger.warning(f'Restarting {self.device.package} to fix it')
            self.config.task_call('Restart')
            self.device.sleep(10)
            return False
        except GamePageUnknownError:
            logger.info('Game server may be under maintenance or network may be broken, check server status now')
            self.checker.check_now()
            if self.checker.is_available():
                logger.critical('Game page unknown, restarting game to recover')
                self.save_error_log()
                handle_notify(
                    self.config.Error_OnePushConfig,
                    title=f"Alas <{self.config_name}> — Page Unknown",
                    content=f"Game page unknown, restarting game to recover.",
                )
                self.device.app_stop()
                self.device.app_start()
                return False
            else:
                self.checker.wait_until_available()
                return False
        except ScriptError as e:
            logger.exception(e)
            logger.critical('This is likely to be a mistake of developers, but sometimes just random issues')
            handle_notify(
                self.config.Error_OnePushConfig,
                title=f"Alas <{self.config_name}> crashed",
                content=f"<{self.config_name}> ScriptError",
            )
            exit(1)
        except RequestHumanTakeover:
            task = getattr(self, '_current_task', 'unknown')
            # Recovery tasks must never be paused — they ARE the recovery.
            if task in ('Restart', 'goto_main', 'Alas'):
                logger.critical(f'Request human takeover on critical task `{task}` — '
                                f'will retry instead of pausing')
                del_cached_property(self, 'config')
                return False
            # Circuit breaker: don't crash, just skip this task
            logger.critical('Request human takeover — pausing task via circuit breaker')
            self.save_error_log()
            cooldown_hours = getattr(self.config, 'Error_CircuitBreakerCooldown', 2)
            until = datetime.now() + timedelta(hours=cooldown_hours)
            self.config.task_disable(task)
            self._save_circuit_breaker(task, until)
            handle_notify(
                self.config.Error_OnePushConfig,
                title=f"Alas <{self.config_name}> — Circuit Breaker",
                content=f"Task `{task}` paused for {cooldown_hours}h (RequestHumanTakeover)\nResumes at {until.strftime('%Y-%m-%d %H:%M')}",
            )
            # Return failure so the loop's circuit breaker handles it
            del_cached_property(self, 'config')
            return False
        except Exception as e:
            logger.exception(e)
            self.save_error_log()
            handle_notify(
                self.config.Error_OnePushConfig,
                title=f"Alas <{self.config_name}> crashed",
                content=f"<{self.config_name}> Exception occured",
            )
            exit(1)

    def save_error_log(self):
        """
        Save last 60 screenshots in ./log/error/<timestamp>
        Save logs to ./log/error/<timestamp>/log.txt
        """
        from module.base.utils import save_image
        from module.handler.sensitive_info import (handle_sensitive_image,
                                                   handle_sensitive_logs)
        if self.config.Error_SaveError:
            if not os.path.exists('./log/error'):
                os.mkdir('./log/error')
            folder = f'./log/error/{int(time.time() * 1000)}'
            logger.warning(f'Saving error: {folder}')
            os.mkdir(folder)
            for data in self.device.screenshot_deque:
                image_time = datetime.strftime(data['time'], '%Y-%m-%d_%H-%M-%S-%f')
                image = handle_sensitive_image(data['image'])
                save_image(image, f'{folder}/{image_time}.png')
            with open(logger.log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                start = 0
                for index, line in enumerate(lines):
                    line = line.strip(' \r\t\n')
                    if re.match('^═{15,}$', line):
                        start = index
                lines = lines[start - 2:]
                lines = handle_sensitive_logs(lines)
            with open(f'{folder}/log.txt', 'w', encoding='utf-8') as f:
                f.writelines(lines)
            self._clean_old_error_logs()

    def _clean_old_error_logs(self, keep_days=7):
        """Remove error log directories older than *keep_days*."""
        import shutil
        error_dir = './log/error'
        if not os.path.exists(error_dir):
            return
        cutoff = time.time() - keep_days * 86400
        for name in os.listdir(error_dir):
            try:
                ts = int(name)
                age = cutoff - ts / 1000.0
            except (ValueError, OSError):
                continue
            if age > 0:
                path = os.path.join(error_dir, name)
                try:
                    shutil.rmtree(path, ignore_errors=True)
                    logger.info(f'Cleaned old error log: {name}')
                except Exception:
                    pass

    def restart(self):
        from module.handler.login import LoginHandler
        LoginHandler(self.config, device=self.device).app_restart()

    def start(self):
        from module.handler.login import LoginHandler
        LoginHandler(self.config, device=self.device).app_start()

    def goto_main(self):
        from module.handler.login import LoginHandler
        from module.ui.ui import UI
        if self.device.app_is_running():
            logger.info('App is already running, goto main page')
            UI(self.config, device=self.device).ui_goto_main()
        else:
            logger.info('App is not running, start app and goto main page')
            LoginHandler(self.config, device=self.device).app_start()
            UI(self.config, device=self.device).ui_goto_main()

    # ---- Task dispatch factory ----
    # NOTE: _task is NOT @staticmethod — it is called at class-definition time
    # to generate task methods, so it must be a plain callable, not a descriptor.
    def _task(module_path, class_name, method='run'):
        """Return a task dispatcher that imports and calls *class_name*.*method*."""
        def dispatch(self):
            import importlib
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            return getattr(cls(config=self.config, device=self.device), method)()
        dispatch.__name__ = method
        return dispatch

    # Simple tasks: Class(config, device).run()
    research = _task('module.research.research', 'RewardResearch')
    commission = _task('module.commission.commission', 'RewardCommission')
    tactical = _task('module.tactical.tactical_class', 'RewardTacticalClass')
    dorm = _task('module.dorm.dorm', 'RewardDorm')
    meowfficer = _task('module.meowfficer.meowfficer', 'RewardMeowfficer')
    guild = _task('module.guild.guild_reward', 'RewardGuild')
    reward = _task('module.reward.reward', 'Reward')
    awaken = _task('module.awaken.awaken', 'Awaken')
    shop_frequent = _task('module.shop.shop_reward', 'RewardShop', 'run_frequent')
    shop_once = _task('module.shop.shop_reward', 'RewardShop', 'run_once')
    shipyard = _task('module.shipyard.shipyard_reward', 'RewardShipyard')
    gacha = _task('module.gacha.gacha_reward', 'RewardGacha')
    freebies = _task('module.freebies.freebies', 'Freebies')
    minigame = _task('module.minigame.minigame', 'Minigame')
    private_quarters = _task('module.private_quarters.private_quarters', 'PrivateQuarters')
    daily = _task('module.daily.daily', 'Daily')
    hard = _task('module.hard.hard', 'CampaignHard')
    exercise = _task('module.exercise.exercise', 'Exercise')
    sos = _task('module.sos.sos', 'CampaignSos')
    war_archives = _task('module.war_archives.war_archives', 'WarArchives')
    event_ab = _task('module.event.campaign_abcd', 'CampaignAB')
    event_cd = _task('module.event.campaign_abcd', 'CampaignCD')
    raid_daily = _task('module.raid.daily', 'RaidDaily')
    event_sp = _task('module.event.campaign_sp', 'CampaignSP')
    maritime_escort = _task('module.event.maritime_escort', 'MaritimeEscort')
    event_a = _task('module.event.campaign_abcd', 'CampaignABCD')
    event_b = event_a
    event_c = event_a
    event_d = event_a
    raid = _task('module.raid.run', 'RaidRun')
    hospital = _task('module.event_hospital.hospital', 'Hospital')
    coalition = _task('module.coalition.coalition', 'Coalition')
    coalition_sp = _task('module.coalition.coalition_sp', 'CoalitionSP')
    opsi_ash_beacon = _task('module.os_ash.meta', 'OpsiAshBeacon')

    # Opsi tasks: OSCampaignRun(config, device).opsi_XXX()
    opsi_explore = _task('module.campaign.os_run', 'OSCampaignRun', 'opsi_explore')
    opsi_shop = _task('module.campaign.os_run', 'OSCampaignRun', 'opsi_shop')
    opsi_voucher = _task('module.campaign.os_run', 'OSCampaignRun', 'opsi_voucher')
    opsi_daily = _task('module.campaign.os_run', 'OSCampaignRun', 'opsi_daily')
    opsi_obscure = _task('module.campaign.os_run', 'OSCampaignRun', 'opsi_obscure')
    opsi_month_boss = _task('module.campaign.os_run', 'OSCampaignRun', 'opsi_month_boss')
    opsi_abyssal = _task('module.campaign.os_run', 'OSCampaignRun', 'opsi_abyssal')
    opsi_archive = _task('module.campaign.os_run', 'OSCampaignRun', 'opsi_archive')
    opsi_stronghold = _task('module.campaign.os_run', 'OSCampaignRun', 'opsi_stronghold')
    opsi_meowfficer_farming = _task('module.campaign.os_run', 'OSCampaignRun', 'opsi_meowfficer_farming')
    opsi_hazard1_leveling = _task('module.campaign.os_run', 'OSCampaignRun', 'opsi_hazard1_leveling')
    opsi_cross_month = _task('module.campaign.os_run', 'OSCampaignRun', 'opsi_cross_month')

    # Campaign tasks: CampaignRun(config, device).run(name=..., folder=..., mode=...)
    @staticmethod
    def _campaign(self):
        from module.campaign.run import CampaignRun
        CampaignRun(config=self.config, device=self.device).run(
            name=self.config.Campaign_Name, folder=self.config.Campaign_Event, mode=self.config.Campaign_Mode)

    main = _campaign
    main2 = _campaign
    main3 = _campaign
    event = _campaign
    event2 = _campaign
    c72_mystery_farming = _campaign
    c122_medium_leveling = _campaign
    c124_large_leveling = _campaign

    @staticmethod
    def gems_farming(self):
        from module.campaign.gems_farming import GemsFarming
        GemsFarming(config=self.config, device=self.device).run(
            name=self.config.Campaign_Name, folder=self.config.Campaign_Event, mode=self.config.Campaign_Mode)

    def daemon(self):
        from module.daemon.daemon import AzurLaneDaemon
        AzurLaneDaemon(config=self.config, device=self.device, task="Daemon").run()

    def opsi_daemon(self):
        from module.daemon.os_daemon import AzurLaneDaemon
        AzurLaneDaemon(config=self.config, device=self.device, task="OpsiDaemon").run()

    def event_story(self):
        from module.eventstory.eventstory import EventStory
        EventStory(config=self.config, device=self.device, task="EventStory").run()

    def azur_lane_uncensored(self):
        from module.daemon.uncensored import AzurLaneUncensored
        AzurLaneUncensored(config=self.config, device=self.device, task="AzurLaneUncensored").run()

    def benchmark(self):
        from module.daemon.benchmark import run_benchmark
        run_benchmark(config=self.config)

    def game_manager(self):
        from module.daemon.game_manager import GameManager
        GameManager(config=self.config, device=self.device, task="GameManager").run()

    def wait_until(self, future):
        """
        Wait until a specific time.

        Args:
            future (datetime):

        Returns:
            bool: True if wait finished, False if config changed.
        """
        future = future + timedelta(seconds=1)
        self.config.start_watching()
        while 1:
            if datetime.now() > future:
                return True
            if self.stop_event is not None:
                if self.stop_event.is_set():
                    logger.info("Update event detected")
                    logger.info(f"[{self.config_name}] exited. Reason: Update")
                    exit(0)

            time.sleep(5)

            if self.config.should_reload():
                return False

    def get_next_task(self):
        """
        Returns:
            str: Name of the next task.
        """
        while 1:
            task = self.config.get_next()
            self.config.task = task
            self.config.bind(task)

            from module.base.resource import release_resources
            if self.config.task.command != 'Alas':
                release_resources(next_task=task.command)

            if task.next_run > datetime.now():
                logger.info(f'Wait until {task.next_run} for task `{task.command}`')
                self.is_first_task = False
                method = self.config.Optimization_WhenTaskQueueEmpty
                if method == 'close_game':
                    logger.info('Close game during wait')
                    self.device.app_stop()
                    release_resources()
                    self.device.release_during_wait()
                    if not self.wait_until(task.next_run):
                        del_cached_property(self, 'config')
                        continue
                    if task.command != 'Restart':
                        self.config.task_call('Restart')
                        del_cached_property(self, 'config')
                        continue
                elif method == 'goto_main':
                    logger.info('Goto main page during wait')
                    self.run('goto_main')
                    release_resources()
                    self.device.release_during_wait()
                    if not self.wait_until(task.next_run):
                        del_cached_property(self, 'config')
                        continue
                elif method == 'stay_there':
                    logger.info('Stay there during wait')
                    release_resources()
                    self.device.release_during_wait()
                    if not self.wait_until(task.next_run):
                        del_cached_property(self, 'config')
                        continue
                else:
                    logger.warning(f'Invalid Optimization_WhenTaskQueueEmpty: {method}, fallback to stay_there')
                    release_resources()
                    self.device.release_during_wait()
                    if not self.wait_until(task.next_run):
                        del_cached_property(self, 'config')
                        continue
            break

        AzurLaneConfig.is_hoarding_task = False
        return task.command

    def loop(self):
        logger.set_file_logger(self.config_name)
        logger.info(f'Start scheduler loop: {self.config_name}')

        while 1:
            # Check update event from GUI
            if self.stop_event is not None:
                if self.stop_event.is_set():
                    logger.info("Update event detected")
                    logger.info(f"Alas [{self.config_name}] exited.")
                    break
            # Check game server maintenance
            self.checker.wait_until_available()
            if self.checker.is_recovered():
                # There is an accidental bug hard to reproduce
                # Sometimes, config won't be updated due to blocking
                # even though it has been changed
                # So update it once recovered
                del_cached_property(self, 'config')
                logger.info('Server or network is recovered. Restart game client')
                self.config.task_call('Restart')
            # Re-enable any circuit-breaker-paused tasks whose cooldown expired
            self._check_circuit_breakers()
            # Get task
            task = self.get_next_task()
            # Init device and change server
            _ = self.device
            self.device.config = self.config
            # Skip first restart
            if self.is_first_task and task == 'Restart':
                logger.info('Skip task `Restart` at scheduler start')
                self.config.task_delay(server_update=True)
                del_cached_property(self, 'config')
                continue

            # Run
            self._current_task = task
            logger.info(f'Scheduler: Start task `{task}`')
            self.device.stuck_record_clear()
            self.device.click_record_clear()
            logger.hr(task, level=0)
            success = self.run(inflection.underscore(task))
            logger.info(f'Scheduler: End task `{task}`')
            self.is_first_task = False

            # Check failures
            failed = deep_get(self.failure_record, keys=task, default=0)
            failed = 0 if success else failed + 1
            deep_set(self.failure_record, keys=task, value=failed)
            # Recovery tasks that must never be paused — pausing them causes
            # cascading failures across all other tasks.
            _CRITICAL_TASKS = {'Restart', 'goto_main', 'Alas'}

            if failed >= 3 and task not in _CRITICAL_TASKS:
                # Circuit breaker: auto-pause this task for cooldown_hours
                # Other tasks continue running normally.
                cooldown_hours = getattr(self.config, 'Error_CircuitBreakerCooldown', 2)
                until = datetime.now() + timedelta(hours=cooldown_hours)

                logger.critical(f"Task `{task}` failed 3 or more times — "
                                f"circuit breaker activated: pausing for {cooldown_hours}h (until {until.strftime('%H:%M')})")
                logger.critical("All other tasks will continue running normally.")

                # Disable the task in config
                self.config.task_disable(task)

                # Record circuit breaker state for re-enable + dashboard visibility
                self._save_circuit_breaker(task, until)

                # Reset failure count (so it gets a fresh start after cooldown)
                deep_set(self.failure_record, keys=task, value=0)

                handle_notify(
                    self.config.Error_OnePushConfig,
                    title=f"Alas <{self.config_name}> — Circuit Breaker",
                    content=f"Task `{task}` paused for {cooldown_hours}h\n"
                            f"Resumes at {until.strftime('%Y-%m-%d %H:%M')}\n"
                            f"Other tasks unaffected.",
                )

                # Continue loop — don't exit!
                del_cached_property(self, 'config')
                continue

            if success:
                # Dependency chain: trigger any tasks that depend on this one
                self._trigger_dependents(task)
                del_cached_property(self, 'config')
                continue
            elif self.config.Error_HandleError:
                # self.config.task_delay(success=False)
                del_cached_property(self, 'config')
                self.checker.check_now()
                continue
            else:
                break


if __name__ == '__main__':
    alas = AzurLaneAutoScript()
    alas.loop()
