import jwt
from django.conf import settings
from django.contrib.auth import authenticate
from django.contrib.auth.models import update_last_login
from rest_framework import serializers

from .models import User


class RegistrationSerializer(serializers.ModelSerializer):
    """ Сериализация регистрации пользователя и создания нового. """

    password = serializers.CharField(
        max_length=128,
        min_length=8,
        write_only=True
    )
    token = serializers.CharField(max_length=255, read_only=True)

    class Meta:
        model = User
        fields = ['username', 'password', 'token', 'name', 'surname', 'patronymic']

    def create(self, validated_data):
        return User.objects.create_user(**validated_data)


class UserSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=255)
    password = serializers.CharField(min_length=8, max_length=128, write_only=True)
    new_password = serializers.CharField(min_length=8, max_length=128, write_only=True, allow_null=True)
    name = serializers.CharField(max_length=255, default='', allow_null=True)
    surname = serializers.CharField(max_length=255, default='', allow_null=True)
    patronymic = serializers.CharField(max_length=255, default='', allow_null=True)

    class Meta:
        model = User
        fields = ['username', 'password', 'new_password', 'token', 'name', 'surname', 'patronymic']

    def validate(self, validated_data):
        print("change validated_data", validated_data)
        print(validated_data.keys())

        user = authenticate(username=validated_data['username'], password=validated_data['password'])
        if user is None:
            return None

        return User.objects.update_user(**validated_data)


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
