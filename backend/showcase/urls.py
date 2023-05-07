from django.urls import path
from .views import OrderAPIView

app_name = 'showcase'
urlpatterns = [
    path('order/', OrderAPIView.as_view()),
]

# Todo:
# GET /orders/ read all my orders
# POST /order/ create new order
# DELETE /orders?order_id= create new order
