import datetime
import jwt
from django.conf import settings

from django.contrib.auth import authenticate
from rest_framework import serializers
from django.contrib.auth.models import update_last_login
from .models import User


class RegistrationSerializer(serializers.ModelSerializer):
    """ Сериализация регистрации пользователя и создания нового. """

    # Убедитесь, что пароль содержит не менее 8 символов, не более 128,
    # и так же что он не может быть прочитан клиентской стороной
    password = serializers.CharField(
        max_length=128,
        min_length=8,
        write_only=True
    )

    # Клиентская сторона не должна иметь возможность отправлять токен вместе с
    # запросом на регистрацию. Сделаем его доступным только на чтение.
    token = serializers.CharField(max_length=255, read_only=True)

    class Meta:
        model = User
        # Перечислить все поля, которые могут быть включены в запрос
        # или ответ, включая поля, явно указанные выше.
        fields = ['username', 'password', 'token', 'name', 'surname', 'patronymic']

    def create(self, validated_data):
        # Использовать метод create_user, который мы
        # написали ранее, для создания нового пользователя.
        return User.objects.create_user(**validated_data)


class LoginSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=255)
    password = serializers.CharField(max_length=128, write_only=True)
    token = serializers.CharField(max_length=255, read_only=True)

    def validate(self, data):
        username = data.get('username', None)
        password = data.get('password', None)

        if username is None:
            raise serializers.ValidationError('A username is required to log in.')

        if password is None:
            raise serializers.ValidationError('A password is required to log in.')

        try:
            user_1 = User.objects.get(username=username)
        except:
            raise serializers.ValidationError('Логин не существует')

        user = authenticate(username=username, password=password)

        if user is None:
            raise serializers.ValidationError('Неверный пароль')
            # raise serializers.ValidationError('A user with this username and password was not found.')

        if not user.is_active:
            raise serializers.ValidationError('This user has been deactivated.')

        update_last_login(None, user)

        return {
            'username': user.username,
            'token': user.token
        }


class DeleteSerializer(serializers.Serializer):
    token = serializers.CharField(max_length=255, read_only=False)

    class Meta:
        model = User
        # Перечислить все поля, которые могут быть включены в запрос
        # или ответ, включая поля, явно указанные выше.
        fields = ['token', 'username']

    def validate(self, data):
        token = jwt.decode(data['token'], key=settings.SECRET_KEY, algorithms=['HS256'])

        user = User.objects.get(id=token['id'])

        if user is None:
            raise serializers.ValidationError('Invalid token provided.')

        username = user.username
        user.delete()

        return {
            'username': username,
            'token': data['token']
        }
