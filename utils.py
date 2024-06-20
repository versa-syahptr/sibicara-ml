from dataclasses import dataclass, field
from typing import List
import numpy as np
import json

@dataclass
class Coordinate:
    x: float
    y: float
    z: float

    def numpy(self):
        return np.array([self.x, self.y, self.z])

@dataclass
class Landmark:
    handedness: int
    hand: List[Coordinate] = field(default_factory=list)
    world: List[Coordinate] = field(default_factory=list)



    @classmethod
    def from_dict(cls, data):
        # data = json.loads(json_str)
        handedness = data["handedness"]
        hand = [Coordinate(coord["x"], coord["y"], coord["z"]) for coord in data["hand"]]
        world = [Coordinate(coord["x"], coord["y"], coord["z"]) for coord in data["world"]]
        return cls(handedness, hand, world)
    
    def to_json(self):
        data = {
            "handedness": self.handedness,
            "hand": [dict(x=coord.x, y=coord.y, z=coord.z) for coord in self.hand],
            "world": [dict(x=coord.x, y=coord.y, z=coord.z) for coord in self.world]
        }
        return json.dumps(data)
    
    @classmethod
    def from_legacy_mp_result(cls, mp_result):
        handedness = mp_result.multi_handedness[0].classification[0].index
        hand = [Coordinate(landmark.x, landmark.y, landmark.z) for landmark in mp_result.multi_hand_landmarks[0].landmark]
        world = [Coordinate(landmark.x, landmark.y, landmark.z) for landmark in mp_result.multi_hand_world_landmarks[0].landmark]
        return cls(handedness, hand, world)
    
    def numpy(self):
        hand_flatten = np.array([coord.numpy() for coord in self.hand]).flatten()
        world_flatten = np.array([coord.numpy() for coord in self.world]).flatten()
        return np.array([self.handedness] + list(hand_flatten) + list(world_flatten), dtype=np.float32)
