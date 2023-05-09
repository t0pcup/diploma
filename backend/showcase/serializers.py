from rest_framework import serializers
import redis

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


class DelOrderSerializer(serializers.Serializer):
    id = serializers.UUIDField()

    def validate(self, data):
        order = Order.objects.get(id=data['id'])
        id_cp = order.id

        try:
            r_server = redis.Redis(host="127.0.0.1", port=6379)
            r_server.lrem("orderList", count=0, value=str(order.id))
        except Exception as e:
            print("REDIS EXCEPTION", e)

        order.delete()
        return {'id': id_cp}
