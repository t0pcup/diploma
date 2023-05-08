import uuid

import jwt
from django.conf import settings
from rest_framework import status
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Order
from .renderers import OrderJSONRenderer, OrdersJSONRenderer, DelOrderRenderer
from .serializers import CreateSerializer, GetAllSerializer, DelOrderSerializer


class OrderAPIView(APIView):
    permission_classes = (AllowAny,)  # TODO: IsAuthenticated
    renderer_classes = (OrderJSONRenderer,)
    serializer_class = CreateSerializer

    def post(self, request):
        order = request.data.get('order', {})
        token = request.headers['Authorization'].split(' ')[1]
        order['owner'] = jwt.decode(token, key=settings.SECRET_KEY, algorithms=['HS256'])['id']

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


class DelOrderAPIView(APIView):
    permission_classes = (AllowAny,)  # TODO: IsAuthenticated
    renderer_classes = (DelOrderRenderer,)
    serializer_class = DelOrderSerializer

    def delete(self, request):
        token, id_ = request.headers['Authorization'].split('\t')
        token = token.split(' ')[1]
        data = {"id": uuid.UUID(id_.split(' ')[1])}

        try:
            order = Order.objects.get(id=data['id'])
        except:
            return Response(data=None, status=status.HTTP_404_NOT_FOUND)

        if order.owner != jwt.decode(token, key=settings.SECRET_KEY, algorithms=['HS256'])['id']:
            return Response(data=None, status=status.HTTP_403_FORBIDDEN)

        serializer = self.serializer_class(data=data)
        serializer.is_valid(raise_exception=True)

        return Response(serializer.data, status=status.HTTP_200_OK)
