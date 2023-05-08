import uuid

from django.contrib.auth import authenticate
from rest_framework import serializers
from datetime import datetime
from .models import Order


class CreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Order
        fields = ['owner', 'poly_wkt', 'crs', 'imagery_start', 'imagery_end']

    def create(self, validated_data):
        return Order.objects.create_order(**validated_data)


class GetAllSerializer(serializers.Serializer):
    owner = serializers.CharField(max_length=255)

    class Meta:
        model = Order
        fields = ['owner', 'poly_wkt', 'crs', 'imagery_start', 'imagery_end', 'id',
                  'created_at', 'finished_at', 'predict']
