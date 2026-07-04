from django import forms
from ml.model_loader import feature_names


class PredictionForm(forms.Form):
    pass


for feature in feature_names:
    PredictionForm.base_fields[feature] = forms.FloatField(
        label=feature,
        widget=forms.NumberInput(
            attrs={
                "class": "form-control",
                "step": "any",
                "placeholder": feature
            }
        )
    )