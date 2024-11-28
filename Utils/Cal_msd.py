import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

class Cal_msd:

    def __init__ ( self, trajectories : list[tuple[np.array, np.array]] = None):
        self.trajectories = trajectories

    def __call__(self, ) -> list:
        ## Get Traj
        trajectories = self.trajectories
        ## Record MSD & lag of each traj
        all_msd = self.cal_all_msd(trajectories)

        return all_msd


    def cal_all_msd(self, trajectories) -> list[np.array]:
        all_msd = []
        for frames, coord in trajectories:
            start_frame, end_frame = frames.min(), frames.max() 
            frame_length = end_frame - start_frame
            msd = np.zeros((2, int(frame_length)))
            msd = self.cal_msd(msd, frames, coord)

            # remain msd != np.nan
            mask = ~np.isnan(msd[1, :])
            msd = msd[:, mask]
            all_msd.append(msd)
        
        return all_msd
    
    def cal_msd(self, msd: np.array, frames: np.array, coord: np.array) -> np.array:
        frame_length = int(frames.max() - frames.min())
        
        msd_values = np.full(frame_length, np.nan)  # Initial MSD values
        lags = np.arange(1, frame_length + 1)       # Lags from 1 to max lag
        
        for lag in lags:
            valid_indices = np.isin(frames + lag, frames)  # Check valid lag 
            current_frames = frames[valid_indices]
            future_frames = current_frames + lag
            current_coords = coord[valid_indices]
            future_coords = coord[np.isin(frames, future_frames)]
            
            # Calculate displacements and squared displacements
            displacements = future_coords - current_coords
            squared_displacements = np.sum(displacements**2, axis=1)
            if len(squared_displacements) > 0:
                msd_values[lag - 1] = np.mean(squared_displacements)
        
        # Combine lags and MSD values into the output array
        msd[0, :] = lags
        msd[1, :] = msd_values
        
        return msd

