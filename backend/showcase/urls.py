from django.urls import path

from .views import OrderAPIView, OrdersAPIView, DelOrderAPIView

app_name = 'showcase'
urlpatterns = [
    path('order/', OrderAPIView.as_view()),
    path('orders/', OrdersAPIView.as_view()),
    path('del-order/', DelOrderAPIView.as_view()),
]
