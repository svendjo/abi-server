"""Local handwriting OCR using TrOCR (microsoft/trocr-base-handwritten).

Reads a whole cell crop in one shot -- no character segmentation -- which sidesteps
the segmentation and MNIST-domain-mismatch problems of the bundled CNN. The heavy
dependencies (torch, transformers) are optional and loaded lazily; if they are not
installed the server falls back to the TFLite digit model. Enable with USE_TROCR=1.

Install the extra deps with:  pip install -r requirements-trocr.txt
The model (~1.3 GB) is downloaded from the Hugging Face hub on first use.
"""

from functools import lru_cache

MODEL_NAME = "microsoft/trocr-base-handwritten"


def available():
    """True if transformers can actually use a PyTorch backend.

    Uses transformers' own torch detection: if torch is missing or unusable,
    transformers reports it here too, so we fall back to the CNN cleanly instead
    of failing later when the model is built.
    """
    try:
        from transformers.utils import is_torch_available
        return bool(is_torch_available())
    except Exception:
        return False


@lru_cache(maxsize=1)
def _load():
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    model.eval()
    return processor, model


def read_crops(crops):
    """Recognize a batch of grayscale cell crops -> list of raw text strings.

    `crops` is a list of HxW uint8 numpy arrays (one per cell). Returns one string
    per crop, in order. Runs a single batched generation pass for speed.
    """
    if not crops:
        return []
    import torch
    from PIL import Image

    processor, model = _load()
    images = [Image.fromarray(c).convert("RGB") for c in crops]
    pixel_values = processor(images=images, return_tensors="pt").pixel_values
    with torch.no_grad():
        ids = model.generate(pixel_values, max_new_tokens=8)
    return processor.batch_decode(ids, skip_special_tokens=True)
