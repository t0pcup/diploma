from django.urls import path
# from .views import LoginAPIView, RegistrationAPIView

app_name = 'showcase'
urlpatterns = [
    path('register/', RegistrationAPIView.as_view()),
    path('login/', LoginAPIView.as_view()),
]

# Todo:
# GET /orders/ read all my orders
# POST /order/ create new order
# DELETE /orders?order_id= create new order
