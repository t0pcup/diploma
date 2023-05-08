import jwt
from django.conf import settings
from rest_framework import status
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import User
from .renderers import UserJSONRenderer
from .serializers import DeleteSerializer, LoginSerializer, RegistrationSerializer, UserSerializer


class RegistrationAPIView(APIView):
    permission_classes = (AllowAny,)
    renderer_classes = (UserJSONRenderer,)
    serializer_class = RegistrationSerializer

    def post(self, request):
        user = request.data.get('user', {})
        serializer = self.serializer_class(data=user)
        serializer.is_valid(raise_exception=True)
        serializer.save()

        return Response(serializer.data, status=status.HTTP_201_CREATED)


class UserAPIView(APIView):
    permission_classes = (AllowAny,)
    renderer_classes = (UserJSONRenderer,)
    serializer_class = UserSerializer

    def patch(self, request):
        token = request.headers['Authorization'].split(' ')[1]
        token = jwt.decode(token, key=settings.SECRET_KEY, algorithms=['HS256'])
        auth = User.objects.get(id=token['id'])

        if auth is None:
            return Response(data=None, status=status.HTTP_401_UNAUTHORIZED)

        user = request.data.get('user', {})
        if sum([int(user[i] is None) for i in user]) == 4:
            return Response(data=None, status=status.HTTP_400_BAD_REQUEST)

        serializer = self.serializer_class(data=user)
        try:
            serializer.is_valid(raise_exception=True)
        except:
            return Response(data=None, status=status.HTTP_403_FORBIDDEN)

        return Response(serializer.data, status=status.HTTP_200_OK)


class LoginAPIView(APIView):
    permission_classes = (AllowAny,)
    renderer_classes = (UserJSONRenderer,)
    serializer_class = LoginSerializer

    def post(self, request):
        user = request.data.get('user', {})
        serializer = self.serializer_class(data=user)
        serializer.is_valid(raise_exception=True)

        return Response(serializer.data, status=status.HTTP_200_OK)


class DeleteAPIView(APIView):
    permission_classes = (AllowAny,)  # todo IsAuthenticated
    renderer_classes = (UserJSONRenderer,)
    serializer_class = DeleteSerializer

    def delete(self, request):
        token = {"token": request.headers['Authorization'].split(' ')[1]}
        serializer = self.serializer_class(data=token)
        serializer.is_valid(raise_exception=True)

        return Response(serializer.data, status=status.HTTP_200_OK)
