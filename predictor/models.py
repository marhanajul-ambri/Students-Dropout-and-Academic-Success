from django.db import models


class PredictionHistory(models.Model):
    prediction_date = models.DateTimeField(auto_now_add=True)

    prediction = models.CharField(max_length=20)

    input_data = models.JSONField(default=dict, blank=True)

    def __str__(self):
        return f"{self.prediction} - {self.prediction_date.strftime('%Y-%m-%d %H:%M')}"