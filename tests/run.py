#!/usr/bin/env python
"""Unit tests for ALAS pure-logic functions. Run with: python tests/run.py"""
import re
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module.base.filter import Filter


class Obj:
    """Minimal object with a 'value' attribute for Filter testing."""
    def __init__(self, v):
        self.value = v

    def __str__(self):
        return str(self.value)


class TestFilter(unittest.TestCase):
    """Filter is the core priority-selector used by all task modules."""

    def setUp(self):
        self.f = Filter(regex=re.compile(r'(\d+)'), attr=['value'])
        self.objs = [Obj(10), Obj(20), Obj(30)]

    def test_basic_single_filter(self):
        self.f.load('20')
        result = self.f.apply(self.objs)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].value, 20)

    def test_multi_filter_chain(self):
        self.f.load('30 > 20 > 10')
        result = self.f.apply(self.objs)
        self.assertEqual([o.value for o in result], [30, 20, 10])

    def test_filter_skips_missing(self):
        self.f.load('99 > 30 > 999')
        result = self.f.apply(self.objs)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].value, 30)

    def test_preset_strings_pass_through(self):
        f = Filter(regex=re.compile(r'(\d+)'), attr=['value'], preset=('reset',))
        f.load('30 > reset > 10')
        result = f.apply(self.objs)
        self.assertIn('reset', result)

    def test_invalid_filter_is_ignored(self):
        self.f.load('xyz')
        result = self.f.apply(self.objs)
        self.assertEqual(len(result), 0)

    def test_filter_str_alias(self):
        """Filter with > encoded as unicode fullwidth ＞ still works."""
        self.f.load('30＞20')  # fullwidth ＞
        result = self.f.apply(self.objs)
        self.assertEqual([o.value for o in result], [30, 20])


class TestResearchCorrectionRules(unittest.TestCase):
    """OCR correction rules for research project name detection."""

    def setUp(self):
        import yaml
        rules_path = os.path.join(
            os.path.dirname(__file__), '..', 'module', 'research', 'correction_rules.yaml')
        with open(rules_path, 'r', encoding='utf-8') as f:
            self.rules = yaml.safe_load(f)

    def test_rules_load(self):
        self.assertIsNotNone(self.rules)
        for key in ['exact', 'number_substitutions', 'prefix_substitutions',
                     'suffix_replacements', 'suffix_exact']:
            self.assertIn(key, self.rules)

    def test_exact_replacements(self):
        exact = {r['from']: r['to'] for r in self.rules['exact']}
        self.assertEqual(exact['G-185'], 'C-185')
        self.assertEqual(exact['D-T85'], 'C-185')
        self.assertEqual(exact['316-MI'], 'E-315-MI')
        self.assertEqual(exact['339-M'], 'H-339-MI')
        self.assertEqual(exact['6-236-MI'], 'G-236-MI')

    def test_m_suffix_to_mi(self):
        suffix_exact = {r['from']: r['to'] for r in self.rules['suffix_exact']}
        self.assertEqual(suffix_exact['M'], 'MI')

    def test_prefix_1_to_d_for_d_numbers(self):
        rules = [r for r in self.rules['prefix_substitutions'] if r['from'] == '1']
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0]['to'], 'D')

    def test_empty_prefix_to_d(self):
        rules = [r for r in self.rules['prefix_substitutions'] if r['from'] == '']
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0]['to'], 'D')


class TestDormFood(unittest.TestCase):
    """Dorm food detection thresholds are defined and reasonable."""

    def test_thresholds_defined(self):
        from module.dorm.dorm import RewardDorm
        self.assertTrue(hasattr(RewardDorm, 'FOOD_ICON_MIN_THRESHOLD'))
        self.assertTrue(hasattr(RewardDorm, 'FOOD_ICON_MEAN_THRESHOLD'))
        self.assertGreater(RewardDorm.FOOD_ICON_MIN_THRESHOLD, 0)
        self.assertLess(RewardDorm.FOOD_ICON_MIN_THRESHOLD, 255)
        self.assertGreater(RewardDorm.FOOD_ICON_MEAN_THRESHOLD, 0)
        self.assertLess(RewardDorm.FOOD_ICON_MEAN_THRESHOLD, 255)


if __name__ == '__main__':
    unittest.main(verbosity=2)
