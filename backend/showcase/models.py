from datetime import datetime
from django.db import models
import uuid
import geopandas as gpd
from shapely.errors import GEOSException
from shapely.wkt import dumps


class OrderManager(models.Manager):
    def create_order(self, poly_wkt, imagery_start, crs='EPSG:3857', imagery_end=datetime.today().strftime('%Y-%m-%d')):
        if poly_wkt is None:
            raise TypeError('Orders must have a geometry.')

        if imagery_start is None:
            raise TypeError('Provide the date of imagery start.')

        if crs not in ['EPSG:3857', 'EPSG:4326']:
            raise TypeError('Unsupported projection.')

        try:
            dataset = gpd.GeoSeries.from_wkt(data=[poly_wkt], crs=crs).to_crs('epsg:4326')
            poly = dumps(dataset.iloc[0], rounding_precision=3)
        except GEOSException as geo:
            raise TypeError(f'Bad geometry. {geo}')

        order = self.model(poly_wkt=str(poly),
                           crs=crs,
                           imagery_start=imagery_start,
                           imagery_end=imagery_end)
        order.save()
        print(order)
        return order


class Order(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid1, editable=False)

    poly_wkt = models.CharField(max_length=10485750, null=False)
    imagery_start = models.DateField(auto_now=False)
    imagery_end = models.DateField(default=datetime.today().strftime('%Y-%m-%d'), auto_now=False)
    crs = models.CharField(max_length=128, default='EPSG:3857')

    created_at = models.DateTimeField(auto_now_add=True)

    finished_at = models.DateTimeField(default=None, blank=True, null=True)
    predict = models.CharField(max_length=10485750, blank=True, null=True)

    objects = OrderManager()

    def __str__(self):
        """ Строковое представление модели (отображается в консоли) """
        f = [self.poly_wkt, self.imagery_start, self.imagery_end, self.crs, self.created_at, self.finished_at,
             self.predict]
        return '\nORDER:' + ' '.join(map(str, f)) + '\n'

    # def delete(self, using=None, keep_parents=False):
    #     self.is_active = False
    #     super().delete()
