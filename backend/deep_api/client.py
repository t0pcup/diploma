import socket

from deep_api.helpers import get_predict, get_order, save_predict

s = socket.socket()
s.connect(("127.0.0.2", 65432))
while True:
    data = s.recv(1024)
    print(data.decode('utf-8'))
    order = get_order(data.decode('utf-8'))
    get_predict(order)
    save_predict(order)
    s.send("got'em".encode('utf-8'))
