import uuid
from datetime import datetime
import redis

import geopandas as gpd
from django.db import models
from shapely.errors import GEOSException
from shapely.wkt import dumps

from authentication.models import User


class OrderManager(models.Manager):
    def create_order(self, poly_wkt, imagery_start, owner, crs='EPSG:3857',
                     imagery_end=datetime.today().strftime('%Y-%m-%d')):
        if poly_wkt is None:
            raise TypeError('Orders must have a geometry.')

        if imagery_start is None:
            raise TypeError('Provide the date of imagery start.')

        if owner is None:
            raise TypeError('Order needs owner.')
        try:
            user = User.objects.get(id=owner)
            if user is None:
                raise TypeError('No such user.')
        except:
            raise TypeError('Invalid owner.')

        if crs not in ['EPSG:3857', 'EPSG:4326']:
            raise TypeError('Unsupported projection.')

        try:
            dataset = gpd.GeoSeries.from_wkt(data=[poly_wkt], crs=crs).to_crs('epsg:4326')
            poly = dumps(dataset.iloc[0])  # , rounding_precision=3
        except GEOSException as geo:
            raise TypeError(f'Bad geometry. {geo}')

        order = self.model(poly_wkt=str(poly),
                           crs=crs,
                           owner=owner,
                           imagery_start=imagery_start,
                           imagery_end=imagery_end)
        order.save()

        try:
            r_server = redis.Redis(host="127.0.0.1", port=6379)
            r_server.rpush("orderList", str(order.id))
        except Exception as e:
            print("REDIS EXCEPTION", e)

        return order


class Order(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid1, editable=False)
    owner = models.CharField(max_length=255, null=False)
    crs = models.CharField(max_length=128, default='EPSG:3857')
    poly_wkt = models.CharField(max_length=10485750, null=False)
    predict = models.CharField(max_length=10485750, blank=True, null=True)

    imagery_start = models.DateField(auto_now=False)
    imagery_end = models.DateField(default=datetime.today().strftime('%Y-%m-%d'), auto_now=False)
    created_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(default=None, blank=True, null=True)

    objects = OrderManager()

    def __str__(self):
        """ Строковое представление модели (отображается в консоли) """
        f = [self.poly_wkt, self.imagery_start, self.imagery_end, self.crs, self.created_at, self.finished_at,
             self.predict]
        return '\nORDER:' + ' '.join(map(str, f)) + '\n'
