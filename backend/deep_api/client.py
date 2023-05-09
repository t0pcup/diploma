import socket

s = socket.socket()
s.connect(("127.0.0.2", 65432))
while True:
    data = s.recv(1024)
    print(data.decode('utf-8'))
    s.send("got'em".encode('utf-8'))
