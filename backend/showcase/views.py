from django.contrib.auth import authenticate
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from .renderers import OrderJSONRenderer
from .serializers import CreateSerializer


class OrderAPIView(APIView):
    permission_classes = (AllowAny,)  # TODO: IsAuthenticated
    renderer_classes = (OrderJSONRenderer,)
    serializer_class = CreateSerializer

    def post(self, request):
        order = request.data.get('order', {})

        # Паттерн создания сериализатора, валидации и сохранения - довольно
        # стандартный, и его можно часто увидеть в реальных проектах.
        serializer = self.serializer_class(data=order)
        serializer.is_valid(raise_exception=True)
        serializer.save()

        return Response(serializer.data, status=status.HTTP_201_CREATED)
