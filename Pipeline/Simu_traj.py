import numpy as np
import pandas as pd
import torch
from andi_datasets.models_phenom import models_phenom
import random


class simulate_traj:
    def __init__(
            self, num_particle_sim, len_frame_sim, 
            num_frame_sim, D_sim, max_gap, prob_noise
            ):
        self.num_particle_sim = num_particle_sim
        self.len_frame_sim = len_frame_sim
        self.num_frame_sim = num_frame_sim
        self.D_sim = D_sim
        self.max_gap = max_gap
        self.prob_noise = prob_noise
        
    def __call__(self, particle_feature_pth, prop_steady):
        np.random.seed(42)
        ## simulate traj
        particle_feature_df = pd.read_csv(particle_feature_pth).to_numpy() 
        traj_simu = self.simu_traj(prop_steady)  

        stacked_array = np.reshape(traj_simu, (-1, 2))
        frames = np.hstack([
            np.full(traj_simu.shape[1], idx) for idx in range(traj_simu.shape[0])
        ])
        node_index_labels = np.hstack([np.arange(traj_simu.shape[1]) for _ in range(traj_simu.shape[0])]).astype(float)
        
        # Add additional features (fixed term + rand term)
        num_move = int(self.num_particle_sim * (1 - prop_steady))
        other_feature_1 = np.zeros((self.num_particle_sim, particle_feature_df.shape[1] - 3))  
        for col_idx in range(particle_feature_df.shape[1] - 3):
            min_val = particle_feature_df[:, col_idx + 3].min()
            max_val = particle_feature_df[:, col_idx + 3].max()
            other_feature_1[:, col_idx] = np.random.uniform(low=min_val, high=max_val, size=self.num_particle_sim)
        blocks = [None] * self.num_frame_sim
        blocks[0] = other_feature_1.copy()
        # Pre-define probabilities
        possible_choices = np.array([0, 1])
        # Store min/max values for each column
        n_features = particle_feature_df.shape[1] - 3
        
        min_vals = np.array([particle_feature_df[:, col_idx + 3].min() for col_idx in range(n_features)])
        max_vals = np.array([particle_feature_df[:, col_idx + 3].max() for col_idx in range(n_features)])
    
        # Process frames
        current_feature = other_feature_1.copy()
        for idx in range(1, self.num_frame_sim):

            # Initialize noise array
            current_pos = stacked_array[0+idx*self.num_frame_sim: self.num_particle_sim+idx*self.num_frame_sim, :]
            previous_pos = stacked_array[0+(idx-1)*self.num_frame_sim: self.num_particle_sim+(idx-1)*self.num_frame_sim, :]
            distances = np.linalg.norm(current_pos - previous_pos, axis=1)

            noise = np.zeros((self.num_particle_sim, n_features))
            moving_choices = np.random.choice(
                possible_choices,
                size=(num_move, n_features),
                p=[0.8, 0.2]  # [no_change_prob, change_prob] [0.8, 0.2]
            )
            moving_mask = (moving_choices == 1)
            
            # Generate choices for static particles (remaining particles)
            static_choices = np.random.choice(
                possible_choices,
                size=(self.num_particle_sim - num_move, n_features),
                p=[0.9, 0.1]  # [no_change_prob, change_prob]
            )
            static_mask = (static_choices == 1)
            
            # Generate and apply noise for both groups
            for col_idx in range(n_features):
                if moving_mask[:, col_idx].any():
                    dist_std = distances[:num_move] 
                    noise[:num_move, col_idx][moving_mask[:, col_idx]] = np.random.normal(
                        loc=0,  # Mean remains zero
                        scale=dist_std[moving_mask[:, col_idx]],  # Use dynamic std_dev
                        size=moving_mask[:, col_idx].sum()
                    )
                
                #Handle static particles (lower variance)
                if static_mask[:, col_idx].any():
                    dist_std = distances[num_move:]
                    noise[num_move:, col_idx][static_mask[:, col_idx]] = np.random.normal(
                        loc=0,  # Mean remains zero
                        scale=dist_std[static_mask[:, col_idx]],  
                        size=static_mask[:, col_idx].sum()
                    )
            current_feature = current_feature + noise

            # Clip each column to its min/max values
            for col_idx in range(n_features):
                current_feature[:, col_idx] = np.clip(
                    current_feature[:, col_idx],
                    min_vals[col_idx],
                    max_vals[col_idx]
                )
            blocks[idx] = current_feature.copy()
    
        # Stack all blocks at once
        other_features = np.vstack(blocks)
        stacked_array = np.hstack([stacked_array, other_features]) 

        ### Delete consecutive frames
        all_centroids, all_frames, all_node_index_labels = self.rm_consec(stacked_array, frames, node_index_labels)

        # Stack the results across all frames
        centroids = np.vstack(all_centroids)
        frames = np.hstack(all_frames)
        node_index_labels = np.hstack(all_node_index_labels)

        ## Get relations 
        relations = np.array([[idx, 0, self.num_frame_sim-1, 0] for idx in range(self.num_particle_sim)])  ## For real, set last ele as "0"
        
        return centroids, node_index_labels, frames, relations


    def simu_traj(self,prop_steady):
        ## generate moving particles
        traj_move,_ = models_phenom().multi_state(
                    N = int(self.num_particle_sim * (1-prop_steady)),
                    L = self.len_frame_sim,
                    T = self.num_frame_sim,
                    alphas =  [0.7, 1.2],  
                    Ds = [[24*self.D_sim, 0.1], [0.06*self.D_sim, 0]],  ## 24(previous) TODO 0.05*self.D_sim, 0
                    M = [[0.9, 0.10], [0.3, 0.7]],  ## [[0.98, 0.02], [0.02, 0.98]]
                ) 
        ## generate steady particles
        num_steady = int((self.num_particle_sim * prop_steady) * 0.2) ## 0.8 TODO
        traj_steady, _ = models_phenom().single_state(
                    N = num_steady, 
                    L = self.len_frame_sim / 1, 
                    T = self.num_frame_sim,
                    Ds = [0.02*self.D_sim, 0.1], # Mean and variance
                    alphas = 1
                )
        ## set steady particle regions
        #traj_steady += self.len_frame_sim / 10 TODO

        ## generate trapped particles
        num_trap = int((self.num_particle_sim * prop_steady) * 0.8)
        traj_trap, _ = models_phenom().multi_state(
                    N = int(num_trap),
                    L = self.len_frame_sim / 5 ,
                    T = self.num_frame_sim, 
                    alphas =  [1.2, 0.7],  
                    Ds = [[0.06*self.D_sim, 0.1], [24*self.D_sim, 0]],  ## 24(previous) TODO 0.05*self.D_sim, 0  
                    M = [[0.9, 0.10], [0.3, 0.7]],  ## [[0.9, 0.1], [0.3, 0.7]]
                ) 
        ## set steady particle regions
        #traj_trap += Len 

        ## stack multistate & steady state & trapped
        traj_simu = np.concatenate((traj_move, traj_steady, traj_trap), axis=1)
        return traj_simu

    def gen_rand_consec(self, mode, min_consec, max_consec):
        while True:
            random_numbers = [random.randint(min_consec, max_consec) for _ in range(self.num_particle_sim)]
            if mode == "particle_gap":  ### TODO
                condition = (sum(random_numbers) - 50) * (self.prob_noise * self.num_particle_sim * self.num_frame_sim - sum(random_numbers))
                if condition > 0:
                    return np.array(random_numbers)
            else:
                if len(np.unique(random_numbers)) >= 0.5 * self.num_frame_sim:
                    return np.array(random_numbers)     

    def rm_consec(self, stacked_array, frames, node_index_labels):
        len_consec = self.gen_rand_consec(mode="particle_gap", min_consec=0, max_consec=self.max_gap)
        start_frames = self.gen_rand_consec(mode="start_frame", min_consec=1, max_consec=self.num_frame_sim - 5)
        end_frames = start_frames + len_consec
        mask_frame = []
        for idx in range(self.num_frame_sim):
            frame_idx = frames[frames == idx]
            mask_frame.append((frame_idx >= start_frames) & (frame_idx <= end_frames))
        mask_frame = np.hstack(mask_frame)
        remain_array = stacked_array[~ mask_frame]  ### delete rows whose corresponding value in mask_frame is True
        remain_frame = frames[~ mask_frame] ###
        remain_node_idx = node_index_labels[~ mask_frame]

        return remain_array, remain_frame, remain_node_idx