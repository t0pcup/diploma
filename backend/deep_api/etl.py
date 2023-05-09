import time
import socket
from eodag import setup_logging

from helpers import *

# server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server.bind(("127.0.0.2", 65432))
# server.listen(5)
# (client, address) = server.accept()
root = 'C:/diploma/backend'

setup_logging(0)
etl_cleanup()

while True:
    all_orders = redis_get_all()
    if len(all_orders) > 0:
        print(all_orders)
    for order_id in all_orders:
        time_start = time.time()

        order = get_order(order_id)
        eo_do_it(order)
        os.system(f'{root}/venv/Scripts/python {root}/deep_api/predict.py {order.id}')

        print(order.id, f"ITER DURATION {convert(int(time.time() - time_start))}")
        # try:
        #     client.send(order.id.encode('utf-8'))
        #     m = client.recv(1024)
        #     print('client', m.decode('utf-8'))
        # except:
        #     _ = 0
