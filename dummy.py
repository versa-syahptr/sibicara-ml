import websocket
import time

ws = websocket.WebSocket()
ws.connect("wss://sibicara-model-7ihdb2vtkq-uc.a.run.app/ws")

while True:
    ws.send('{"key": "value"}')
    result = ws.recv()
    print(result)
    time.sleep(1)