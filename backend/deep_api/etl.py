from helpers import *
import time

setup_logging(3)

while True:
    time.sleep(0.1)
    all_orders = redis_get_all()
    print(all_orders)
    for order_id in all_orders:
        etl_cleanup()
        order = get_order(order_id)
        eo_do_it(order)
    time.sleep(500)
