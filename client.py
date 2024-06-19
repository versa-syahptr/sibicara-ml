#!/usr/bin/env python

import asyncio
import websockets
import json
import pandas as pd
import glob
import random

import mpipe
import utils

files = glob.glob("landmarks\*\*.csv")

async def main():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as websocket:
        # file = random.choice(files)
        # print(file)
        # data = pd.read_csv(file).to_dict('records')
        for landmark in mpipe.generate_stream():
            data = utils.landmark2records(landmark)
            await websocket.send(json.dumps(data))
            result = await websocket.recv()
            print(f">>> {result}")

if __name__ == "__main__":
    asyncio.run(main())