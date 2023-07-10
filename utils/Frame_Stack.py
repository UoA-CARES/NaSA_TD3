
import numpy as np
from collections import deque

class FrameStack:
    def __init__(self, env, k=3):
        self.env  = env
        self.k    = k  # number of frames to be stacked
        self.frames_stacked = deque([], maxlen=k)

    def reset(self):
        _ = self.env.reset()
        frame = self.env.physics.render(84, 84, camera_id=0) # --> shape= (84, 84, 3)
        frame = np.moveaxis(frame, -1, 0)                    # --> shape= (3, 84, 84)
        for _ in range(self.k):
            self.frames_stacked.append(frame)
        stacked_frames = np.concatenate(list(self.frames_stacked), axis=0) # --> shape = (9, 84, 84)
        return stacked_frames

    def step(self, action):
        time_step    = self.env.step(action)
        reward, done = time_step.reward, time_step.last()
        frame = self.env.physics.render(84, 84, camera_id=0)
        frame = np.moveaxis(frame, -1, 0)
        self.frames_stacked.append(frame)
        stacked_frames = np.concatenate(list(self.frames_stacked), axis=0)
        return stacked_frames, reward, done
