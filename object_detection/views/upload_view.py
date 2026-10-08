from django.shortcuts import render, redirect
from PIL import Image

from object_detection.models import Prediction
from object_detection.ml.classifier import classify_image

MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5 MB


def _is_valid_image(file):
    if file.size > MAX_IMAGE_SIZE:
        return False, "Image too large (max 5 MB)."
    try:
        img = Image.open(file)
        img.verify()
        file.seek(0)  # رجّع مؤشر الملف للبداية بعد verify
    except Exception:
        return False, "Upload a valid image file."
    return True, ""


def upload_view(request):
    error = ""

    if request.method == "POST" and request.FILES.get("image"):
        image = request.FILES["image"]
        ok, error = _is_valid_image(image)

        if ok:
            prediction = Prediction(image=image)
            prediction.save()

            try:
                label, confidence = classify_image(prediction.image.path)
            except Exception as e:
                prediction.delete()
                error = f"Classification failed: {e}"
            else:
                prediction.label = label
                prediction.confidence = confidence
                prediction.save()
                return redirect("result", pk=prediction.pk)

    return render(request, "object_detection/upload.html", {"error": error})