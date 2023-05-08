import uuid
from datetime import datetime, timedelta

import jwt
from django.conf import settings
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin
from django.db import models


class UserManager(BaseUserManager):
    def create_user(self, username, password=None, name='', surname='', patronymic=''):
        """ Создает и возвращает пользователя. """
        if username is None:
            raise TypeError('Users must have a username.')

        if password is None:
            raise TypeError('Users must have a password.')

        user = self.model(username=username, name=name, surname=surname, patronymic=patronymic)
        user.set_password(password)
        user.save()
        return user

    def update_user(self, username, password, new_password=None, name=None, surname=None, patronymic=None):
        """ Изменяет и возвращает пользователя. """
        user = self.model.objects.get(username=username)
        if new_password is not None:
            user.set_password(new_password)

        if name is not None and user.name != name:
            user.name = name

        if surname is not None and user.surname != surname:
            user.surname = surname

        if patronymic is not None and user.patronymic != patronymic:
            user.patronymic = patronymic

        user.save()
        print(user.name, user.surname, user.patronymic)
        return user

    def create_superuser(self, username, password):
        """ Создает и возвращает пользователя с привилегиями супер-админа. """
        if password is None:
            raise TypeError('Superusers must have a password.')

        user = self.create_user(username, password)
        user.is_superuser = True
        user.is_staff = True
        user.save()

        return user


class User(AbstractBaseUser, PermissionsMixin):
    id = models.UUIDField(primary_key=True, default=uuid.uuid1, editable=False)

    username = models.CharField(max_length=255, unique=True)

    name = models.CharField(max_length=255, default='', null=True, blank=True)
    surname = models.CharField(max_length=255, default='', null=True, blank=True)
    patronymic = models.CharField(max_length=255, default='', null=True, blank=True)

    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    USERNAME_FIELD = 'username'

    objects = UserManager()

    def __str__(self):
        """ Строковое представление модели (отображается в консоли) """
        return self.username

    @property
    def token(self):
        return self._generate_jwt_token()

    def get_full_name(self):
        return ' '.join(map(str, [self.name, self.patronymic, self.surname]))

    def get_short_name(self):
        return self.username

    def _generate_jwt_token(self):
        dt = datetime.now() + timedelta(hours=1)

        token = jwt.encode({
            'id': str(self.id),
            'exp': dt.utcfromtimestamp(dt.timestamp())
        }, settings.SECRET_KEY, algorithm='HS256')

        return token
