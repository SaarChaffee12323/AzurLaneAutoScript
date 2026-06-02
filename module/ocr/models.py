from module.base.decorator import cached_property


class OcrModel:
    @cached_property
    def azur_lane(self):
        # cnocr v2.x model: densenet_lite_136-gru
        # Charset: 0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ:/- and more
        # _num_classes: 6682
        # Note: Original v1.x model (mxnet) was trained specifically on AL fonts.
        # Using v2.x generic model as fallback since mxnet is EOL.
        from module.ocr.al_ocr import AlOcr
        return AlOcr(model_name='densenet_lite_136-gru', name='azur_lane')

    @cached_property
    def azur_lane_jp(self):
        from module.ocr.al_ocr import AlOcr
        return AlOcr(model_name='densenet_lite_136-gru', name='azur_lane_jp')

    @cached_property
    def cnocr(self):
        # cnocr v2.x model: densenet_lite_136-gru
        # Charset: Numbers, English characters, Chinese characters, symbols, <space>
        # _num_classes: 6682
        from module.ocr.al_ocr import AlOcr
        return AlOcr(model_name='densenet_lite_136-gru', name='cnocr')

    @cached_property
    def jp(self):
        # cnocr v2.x does not have a dedicated JP model, using general model as fallback
        from module.ocr.al_ocr import AlOcr
        return AlOcr(model_name='densenet_lite_136-gru', name='jp')

    @cached_property
    def tw(self):
        # cnocr v2.x does not have a dedicated TW model, using general model as fallback
        from module.ocr.al_ocr import AlOcr
        return AlOcr(model_name='densenet_lite_136-gru', name='tw')


OCR_MODEL = OcrModel()
