from rest_framework import serializers

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
        order.delete()

        return {'id': id_cp}
