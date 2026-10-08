from django.shortcuts import render, get_object_or_404

from object_detection.models import Prediction


def result_view(request, pk):

    prediction = get_object_or_404(Prediction, pk=pk)


    history = Prediction.objects.order_by("-created_at")[:10]

    return render(
        request,
        "object_detection/result.html",
        {"prediction": prediction, "history": history},
    )