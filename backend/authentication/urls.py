from django.urls import path

from .views import DeleteAPIView, LoginAPIView, RegistrationAPIView, UserAPIView

app_name = 'authentication'
urlpatterns = [
    path('register/', RegistrationAPIView.as_view()),
    path('user/', UserAPIView.as_view()),
    path('login/', LoginAPIView.as_view()),
    path('me/', DeleteAPIView.as_view()),
]
