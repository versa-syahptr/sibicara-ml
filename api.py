from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from predictor import Predictor
from utils import Landmark


app = FastAPI()
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

predictor = Predictor("saved_model")

@app.get("/")
async def read_root():
    return FileResponse('frontend/index.html')

@app.websocket("/predict")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_json()
            # TODO parse data
            landmark = Landmark.from_dict(data)
            array = landmark.numpy()
            result, conf = predictor.predict(array)
            await websocket.send_text(f"{result} ({conf*100:.2f}%)")
        except ValueError:
            await websocket.send_text('Error')
            continue
        except WebSocketDisconnect:
            break
