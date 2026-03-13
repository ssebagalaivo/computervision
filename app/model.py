from __future__ import annotations

from io import BytesIO
from threading import Lock


class VGG16Classifier:
    _model = None
    _lock = Lock()

    @classmethod
    def _get_model(cls):
        if cls._model is None:
            with cls._lock:
                if cls._model is None:
                    from tensorflow.keras.applications.vgg16 import VGG16

                    cls._model = VGG16(weights="imagenet")
        return cls._model

    def predict(self, raw_bytes: bytes, top: int = 5) -> list[dict[str, float | str]]:
        import numpy as np
        from PIL import Image
        from tensorflow.keras.applications.vgg16 import (
            decode_predictions,
            preprocess_input,
        )
        from tensorflow.keras.preprocessing import image

        pil_image = Image.open(BytesIO(raw_bytes)).convert("RGB").resize((224, 224))
        image_array = image.img_to_array(pil_image)
        batch = np.expand_dims(image_array, axis=0)
        batch = preprocess_input(batch)
        predictions = self._get_model().predict(batch, verbose=0)
        decoded = decode_predictions(predictions, top=top)[0]
        return [
            {
                "id": imagenet_id,
                "label": label.replace("_", " "),
                "confidence": float(score),
            }
            for imagenet_id, label, score in decoded
        ]


classifier = VGG16Classifier()
