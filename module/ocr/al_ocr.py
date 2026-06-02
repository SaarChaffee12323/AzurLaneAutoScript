import os

from module.logger import logger

# ---- Local model paths (bundled with project, no network needed) ----
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MODEL_DIR = os.path.join(_PROJECT_ROOT, 'bin', 'cnocr_models_v2')

REC_MODEL_PATH = os.path.join(_MODEL_DIR, 'cnocr', 'cnocr-v2.3-densenet_lite_136-gru-epoch=004-ft-model.onnx')
DET_MODEL_PATH = os.path.join(_MODEL_DIR, 'cnstd', 'ch_PP-OCRv5_det_infer.onnx')

_HF_MIRROR = 'https://hf-mirror.com'

logger.info('Loading OCR dependencies')
from cnocr import CnOcr


class AlOcr(CnOcr):
    """
    ALAS wrapper for cnocr v2.x providing v1.x-compatible API.

    - Uses bundled local ONNX models (no HF download needed).
    - Internal methods (ocr_for_single_line, ocr_for_single_lines) return
      cnocr v2.x native Dict format so parent's pipeline works correctly.
    - Atomic methods (atomic_ocr, atomic_ocr_for_single_lines) convert to
      v1.x str format for ALAS compatibility.
    """
    CNOCR_CONTEXT = 'cpu'

    def __init__(
            self,
            model_name='densenet_lite_136-gru',
            model_epoch=None,
            cand_alphabet=None,
            root=None,
            context='cpu',
            name=None,
    ):
        self._args = (model_name, model_epoch, cand_alphabet, root, context, name)
        self._model_loaded = False
        self._model_name = model_name
        self._alphabet = None
        self._cand_alph_idx = None

    def init(self,
             model_name='densenet_lite_136-gru',
             model_epoch=None,
             cand_alphabet=None,
             root=None,
             context='cpu',
             name=None,
             ):
        self._model_name = model_name

        if os.path.exists(REC_MODEL_PATH) and os.path.exists(DET_MODEL_PATH):
            logger.info('Using local OCR models')
            super().__init__(
                rec_model_fp=REC_MODEL_PATH,
                det_model_fp=DET_MODEL_PATH,
                rec_model_backend='onnx',
                det_model_backend='onnx',
                cand_alphabet=cand_alphabet,
                context=AlOcr.CNOCR_CONTEXT,
            )
        else:
            logger.warning(f'Local models missing, downloading from {_HF_MIRROR}')
            logger.warning(f'  Rec: {REC_MODEL_PATH}')
            logger.warning(f'  Det: {DET_MODEL_PATH}')
            if 'HF_ENDPOINT' not in os.environ:
                os.environ['HF_ENDPOINT'] = _HF_MIRROR
            super().__init__(
                rec_model_name=model_name,
                cand_alphabet=cand_alphabet,
                context=AlOcr.CNOCR_CONTEXT,
            )

        alphabet = None
        if hasattr(self, 'rec_model'):
            rec = self.rec_model
            for attr in ('vocab', '_vocab', '_alphabet'):
                if hasattr(rec, attr):
                    alphabet = getattr(rec, attr)
                    break

        if alphabet is not None and len(alphabet) > 0:
            self._alphabet = alphabet
            logger.info(f'OCR alphabet loaded: {len(self._alphabet)} characters')
        else:
            logger.warning('Could not extract alphabet, using fallback')
            self._alphabet = [' '] + [chr(i) for i in range(0x4e00, 0x9fff)]

    # ---- Type conversion helpers ----

    @staticmethod
    def _to_str(result):
        """v2.x dict → v1.x string"""
        if isinstance(result, dict):
            return result.get('text', '')
        return result

    def _to_str_list(self, result_list):
        return [self._to_str(r) for r in result_list]

    # ---- Core OCR (v2.x native format internally) ----
    # These return v2.x Dict so parent's internal pipeline works.
    # ALAS code should use the atomic_* methods instead.

    def ocr(self, img_fp, **kwargs):
        if not self._model_loaded:
            self.init(*self._args)
            self._model_loaded = True
        return super().ocr(img_fp, **kwargs)

    def ocr_for_single_line(self, img_fp):
        if not self._model_loaded:
            self.init(*self._args)
            self._model_loaded = True
        return super().ocr_for_single_line(img_fp)

    def ocr_for_single_lines(self, img_list, batch_size=1):
        if not self._model_loaded:
            self.init(*self._args)
            self._model_loaded = True
        return super().ocr_for_single_lines(img_list, batch_size=batch_size)

    def set_cand_alphabet(self, cand_alphabet):
        if not self._model_loaded:
            self.init(*self._args)
            self._model_loaded = True
        if hasattr(self, 'rec_model'):
            self.rec_model.set_cand_alphabet(cand_alphabet)
        self._cand_alph_idx = cand_alphabet

    # ---- Sampling for fine-tuning data collection ----

    def _maybe_sample(self, image, text, cand_alphabet):
        """Save a sample for OCR fine-tuning if the sampler decides to."""
        try:
            from module.ocr.sampler import SAMPLER
            if SAMPLER.should_sample():
                SAMPLER.save_sample(
                    image=image,
                    text=text,
                    cand_alphabet=cand_alphabet or '',
                )
        except Exception:
            pass  # sampler should never break OCR

    # ---- Atomic methods (v1.x compatible string output) ----
    # ALAS code calls these through Ocr -> OCR_MODEL -> AlOcr

    def atomic_ocr(self, img_fp, cand_alphabet=None):
        if not self._model_loaded:
            self.init(*self._args)
            self._model_loaded = True
        self.set_cand_alphabet(cand_alphabet)
        raw = super().ocr(img_fp)
        result = self._to_str_list(raw)
        # Sample first result for training data
        if result and raw:
            img = img_fp[0] if isinstance(img_fp, list) else img_fp
            self._maybe_sample(img, result[0], cand_alphabet)
        return result

    def atomic_ocr_for_single_line(self, img_fp, cand_alphabet=None):
        if not self._model_loaded:
            self.init(*self._args)
            self._model_loaded = True
        self.set_cand_alphabet(cand_alphabet)
        raw = super().ocr_for_single_line(img_fp)
        result = self._to_str(raw)
        self._maybe_sample(img_fp, result, cand_alphabet)
        return result

    def atomic_ocr_for_single_lines(self, img_list, cand_alphabet=None, batch_size=1):
        if not self._model_loaded:
            self.init(*self._args)
            self._model_loaded = True
        self.set_cand_alphabet(cand_alphabet)
        raw = super().ocr_for_single_lines(img_list, batch_size=batch_size)
        result = self._to_str_list(raw)
        # Sample one per batch to avoid filling disk
        if result and img_list:
            self._maybe_sample(img_list[0], result[0], cand_alphabet)
        return result
