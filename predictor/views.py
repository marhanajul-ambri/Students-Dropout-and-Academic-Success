import numpy as np

from django.shortcuts import render

from .forms import PredictionForm
from .models import PredictionHistory

from ml.model_loader import model, feature_names


LABELS = {
    0: "Dropout",
    1: "Enrolled",
    2: "Graduate",
}


def welcome(request):
    return render(request, "welcome.html")


def home(request):

    prediction = None

    form = PredictionForm(request.POST or None)

    if request.method == "POST" and form.is_valid():

        values = []
        input_data = {}

        for feature in feature_names:

            value = float(form.cleaned_data[feature])

            values.append(value)

            input_data[feature] = value

        prediction_result = model.predict(
            np.array([values])
        )[0]

        prediction = LABELS.get(
            prediction_result,
            str(prediction_result)
        )

        PredictionHistory.objects.create(
            prediction=prediction,
            input_data=input_data,
        )

    return render(
        request,
        "home.html",
        {
            "form": form,
            "prediction": prediction,
        },
    )


def history(request):

    predictions = PredictionHistory.objects.order_by(
        "-prediction_date"
    )

    return render(
        request,
        "history.html",
        {
            "predictions": predictions,
        },
    )