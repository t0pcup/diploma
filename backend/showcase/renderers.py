import json

from rest_framework.renderers import JSONRenderer
from .models import Order


class OrderJSONRenderer(JSONRenderer):
    charset = 'utf-8'

    def render(self, data, media_type=None, renderer_context=None):
        token = data.get('token', None)

        if token is not None and isinstance(token, bytes):
            data['token'] = token.decode('utf-8')

        return json.dumps({
            'order': data
        })


class OrdersJSONRenderer(JSONRenderer):
    charset = 'utf-8'

    def render(self, data, media_type=None, renderer_context=None):
        not_ready, ready = [], []
        for order in Order.objects.filter(owner=data["owner"]):
            d_o = order.__dict__
            del d_o['_state']
            d_o['id'] = str(d_o['id'])
            d_o['imagery_start'] = str(d_o['imagery_start'])
            d_o['imagery_end'] = str(d_o['imagery_end'])
            d_o['created_at'] = str(d_o['created_at'])

            if order.finished_at is None:
                not_ready.append(d_o)
            else:
                d_o['finished_at'] = str(d_o['finished_at'])
                ready.append(d_o)

        return json.dumps([not_ready, ready])
