from django.contrib.auth import authenticate
from rest_framework import serializers

from .models import Order


class CreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Order
        # fields = ['token', 'imagery_start', 'poly_wkt', 'imagery_end', 'crs']
        fields = ['poly_wkt', 'crs', 'imagery_start', 'imagery_end']

    def create(self, validated_data):
        return Order.objects.create_order(**validated_data)
