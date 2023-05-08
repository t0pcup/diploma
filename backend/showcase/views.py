from django.contrib.auth import authenticate
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
import jwt
from django.conf import settings
from .renderers import OrderJSONRenderer, OrdersJSONRenderer
from .serializers import CreateSerializer, GetAllSerializer
import uuid
from .models import Order


class OrderAPIView(APIView):
    permission_classes = (AllowAny,)  # TODO: IsAuthenticated
    renderer_classes = (OrderJSONRenderer,)
    serializer_class = CreateSerializer

    def post(self, request):
        order = request.data.get('order', {})
        token = request.headers['Authorization'].split(' ')[1]
        token = jwt.decode(token, key=settings.SECRET_KEY, algorithms=['HS256'])['id']
        order['owner'] = token

        # Паттерн создания сериализатора, валидации и сохранения - довольно
        # стандартный, и его можно часто увидеть в реальных проектах.
        serializer = self.serializer_class(data=order)
        serializer.is_valid(raise_exception=True)
        serializer.save()

        return Response(serializer.data, status=status.HTTP_201_CREATED)


class OrdersAPIView(APIView):
    permission_classes = (AllowAny,)  # TODO: IsAuthenticated
    renderer_classes = (OrdersJSONRenderer,)
    serializer_class = GetAllSerializer

    def get(self, request):
        token = request.headers['Authorization'].split(' ')[1]
        token = {"owner": jwt.decode(token, key=settings.SECRET_KEY, algorithms=['HS256'])['id']}

        serializer = self.serializer_class(data=token)
        serializer.is_valid(raise_exception=True)

        return Response(serializer.data, status=status.HTTP_200_OK)
