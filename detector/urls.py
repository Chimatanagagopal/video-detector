from django.urls import path
from .views import VideoDetectView

urlpatterns = [
    path('detect-video/', VideoDetectView.as_view(), name='detect-video'),
]
