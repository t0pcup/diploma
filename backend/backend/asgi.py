import os

from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter
from channels.routing import URLRouter
from django.core.asgi import get_asgi_application
from django.urls import path

from backend.consumers import RealTimeConsumer
#
# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
#
# django_asgi_app = get_asgi_application()
#
# application = ProtocolTypeRouter({
#     'http': django_asgi_app,
#     'websocket': AuthMiddlewareStack(
#         URLRouter([
#             path('ws', RealTimeConsumer.as_asgi())
#         ])
#     )
# })
