from django.db import models


class Prediction(models.Model):

    image = models.ImageField(upload_to="uploads/")


    label = models.CharField(max_length=20, blank=True)


    confidence = models.FloatField(default=0.0)


    created_at = models.DateTimeField(auto_now_add=True)

    @property
    def confidence_percent(self):
        """Turn 0.87 into 87 so the page can show a friendly percentage."""
        return round(self.confidence * 100)

    def __str__(self):
        # How this row looks in the admin panel / shell.
        return f"{self.label} ({self.confidence_percent}%)"