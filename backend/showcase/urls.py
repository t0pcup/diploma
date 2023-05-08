from django.urls import path
from .views import OrderAPIView, OrdersAPIView

app_name = 'showcase'
urlpatterns = [
    path('order/', OrderAPIView.as_view()),
    path('orders/', OrdersAPIView.as_view()),
]

# Todo:
# GET /orders/ read all my orders
# POST /order/ create new order done?
# DELETE /orders?order_id= delete order
