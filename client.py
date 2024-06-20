#!/usr/bin/env python

import asyncio
import websockets
import json
import pandas as pd
import glob
import random

import mpipe
import utils


async def main():
    uri = "ws://localhost:8000/predict"
    async with websockets.connect(uri) as websocket:
        # file = random.choice(files)
        # print(file)
        # data = pd.read_csv(file).to_dict('records')
        for landmark in mpipe.generate_stream():
            data = utils.Landmark.from_legacy_mp_result(landmark)
            await websocket.send(data.to_json())
            result = await websocket.recv()
            print(f">>> {result}")

if __name__ == "__main__":
    asyncio.run(main())