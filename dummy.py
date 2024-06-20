import websocket
import time

ws = websocket.WebSocket()
ws.connect("ws://localhost:8000/ws")

while True:
    ws.send('{"key": "value"}')
    result = ws.recv()
    print(result)
    time.sleep(1)