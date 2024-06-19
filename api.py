from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from predictor import Predictor
import asyncio
import string
import random


app = FastAPI()

predictor = Predictor("modelSIBI.h5")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_json()
            # TODO parse data
            asyncio.sleep(0.5)
            result, conf = random.choice(string.ascii_uppercase), random.random()
            await websocket.send_text(f"{result} ({conf*100:.2f}%)")
        except ValueError:
            await websocket.send_text('Error')
            continue
        except WebSocketDisconnect:
            break
