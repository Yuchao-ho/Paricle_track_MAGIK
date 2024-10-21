import numpy as np
import pandas as pd
from skimage import measure
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from torch_geometric.data import Data
from andi_datasets.models_phenom import models_phenom
import random
from tqdm import tqdm


class Graph_Generator:
    def __init__(
            self, connectivity_radius=None, frame_test=None,
            num_particle_sim=None, len_frame_sim=None, num_frame_sim=None, 
            D_sim=None, max_gap = None, prob_noise = None
        ):
        self.connectivity_radius = connectivity_radius
        self.frame_test = frame_test
        if self.frame_test is not None:
            self.max_frame_distance = 5
            #self.max_frame_distance = self.frame_test[1] -self.frame_test[0] +1   
        
        self.num_particle_sim = num_particle_sim
        self.len_frame_sim = len_frame_sim
        self.num_frame_sim = num_frame_sim
        self.D_sim = D_sim
        self.max_gap = max_gap
        self.prob_noise = prob_noise


    def __call__(self, particle_feature_pth):
        x, node_index_labels, frames, relations = self.get_node_feature(particle_feature_pth, self.frame_test)

        edge_index, edge_attr = self.compute_connectivity(x, frames)
        edge_ground_truth = self.compute_ground_truth(node_index_labels, edge_index, relations)
        edge_index = edge_index.T
        edge_attr = edge_attr[:, None]
        edge_ground_truth = edge_ground_truth[:, None]

        graph = Data(
            x=torch.tensor(x, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float),
            distance=torch.tensor(edge_attr, dtype=torch.float),
            frames=torch.tensor(frames, dtype=torch.float),
            y=torch.tensor(edge_ground_truth, dtype=torch.float),
        )
        return graph

    def get_node_feature(self, particle_feature_pth, frame_test):
        # For training Simulation: 
        if frame_test is None:
            traj_simulation = simulate_traj(
                num_particle_sim= self.num_particle_sim, len_frame_sim= self.len_frame_sim, 
                num_frame_sim= self.num_frame_sim, D_sim= self.D_sim, 
                max_gap= self.max_gap, prob_noise= self.prob_noise
            )
            x, node_index_labels, frames, relations = traj_simulation(particle_feature_pth = particle_feature_pth)
            self.max_frame_distance =  self.max_gap  ### TODO

        # For testing video:
        else:
            particle_feature_df = pd.read_csv(particle_feature_pth)
            particle_feature_df['label'] = particle_feature_df.groupby('frame').cumcount()
            test_feature_df = particle_feature_df[
                (particle_feature_df['frame'] >= frame_test[0]) & (particle_feature_df['frame'] <= frame_test[1])]
            
            x = test_feature_df.iloc[:, 1:-1].to_numpy()   
            #x *= np.array([1.0, 1.0, 0.4, 0.8, 0.5])    ## add weighted features

            node_index_labels = np.arange(x.shape[0]) / x.shape[0]
            frames = test_feature_df.iloc[:, 0].to_numpy()

            num_particle = particle_feature_df[(particle_feature_df['frame'] == frame_test[0])].shape[0]
            relations = np.array([[idx, frame_test[0], frame_test[1], 0] for idx in range(num_particle)])
        
        return x, node_index_labels, frames, relations

    def compute_connectivity(self, x, frames):
        positions = x[:, :2]   ## TODO
        distances = np.linalg.norm(positions[:, None] - positions, axis=-1)  # Broadcast
        frame_diff = (frames[:, None] - frames) * -1  # Broadcast
        mask = (distances < self.connectivity_radius) & (
            (frame_diff <= self.max_frame_distance) & (frame_diff > 0)
            )
        edge_index = np.argwhere(mask)
        edge_attr = distances[mask]
        return edge_index, edge_attr

    def compute_ground_truth(self, indices, edge_index, relation):
        sender = indices[edge_index[:, 0]]
        receiver = indices[edge_index[:, 1]]
        self_connections_mask = sender == receiver
        relation[:,-1] = 0
        relation_indices = relation[:, [-1, 0]]
        #relation_indices = relation_indices[relation_indices[:, 0] != 0]  ## Old Way to build
        relation_indices = relation_indices[relation_indices[:, 0] == 0]   ## New Way 
        relation_mask = np.zeros(len(edge_index), dtype=bool)
        zipped_node = zip(sender, receiver)

        for i, (s, r) in enumerate(zipped_node):
            if np.any((relation_indices == [s, r]).all(1)):
                relation_mask[i] = True
        ground_truth = self_connections_mask | relation_mask
        return ground_truth


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
        
    def __call__(self, particle_feature_pth, num_noise= 1):
        ## simulate traj
        particle_feature_df = pd.read_csv(particle_feature_pth).to_numpy() 

        traj_simu,_ = models_phenom().multi_state(
            N = self.num_particle_sim,
            L = self.len_frame_sim,
            T = self.num_frame_sim,
            alphas =  [1.2, 0.7],
            Ds = [[10*self.D_sim, 0.1], [1.2*self.D_sim, 0.0]],
            M = [[0.98, 0.02], [0.02, 0.98]],
        ) 

        stacked_array = np.reshape(traj_simu, (-1, 2))
        frames = np.hstack([
            np.full(traj_simu.shape[1], idx) for idx in range(traj_simu.shape[0])
        ])
        node_index_labels = np.hstack([np.arange(traj_simu.shape[1]) for _ in range(traj_simu.shape[0])]).astype(float)

        # Add additional features (fixed term + rand term)
        other_feature_1 = np.zeros((self.num_particle_sim, particle_feature_df.shape[1] - 3))  
        for col_idx in range(particle_feature_df.shape[1] - 3):
            min_val = particle_feature_df[:, col_idx + 3].min()
            max_val = particle_feature_df[:, col_idx + 3].max()
            # Generate random values for this column
            other_feature_1[:, col_idx] = np.random.uniform(low=min_val, high=max_val, size=self.num_particle_sim)

        blocks = [other_feature_1]  
        for idx in range(1, self.num_frame_sim):
            if idx % 5 == 0: ## change along time slots (5)
                noise = np.random.normal(loc=-0.05, scale=0.05, size=(other_feature_1.shape[0], other_feature_1.shape[1]))
                other_feature_1 = other_feature_1 + noise
            blocks.append(other_feature_1)
        other_features = np.vstack(blocks)
        ## Normalize features
        other_features = self.norm_mat(other_features, particle_feature_df)

        stacked_array = np.hstack([stacked_array/self.len_frame_sim, other_features]) 
        ### Delete consecutive frames
        remain_array, remain_frame, remain_node_idx = self.rm_consec(stacked_array, frames, node_index_labels)

        all_centroids, all_frames, all_node_index_labels, noise_frame_idx = self.gen_noise(
            remain_array, remain_node_idx, remain_frame, num_noise
            )
        # Stack the results across all frames
        centroids = np.vstack(all_centroids)
        frames = np.hstack(all_frames)
        node_index_labels = np.hstack(all_node_index_labels)
        #node_index_labels /= max(node_index_labels)

        ## Get relations 
        relations = np.array([[idx, 0, self.num_frame_sim-1, 0] for idx in range(self.num_particle_sim)])  ## For real, set last ele as "0"
        noise_relation = np.array([[idx+self.num_particle_sim, noise_idx, noise_idx, 1]   ## For noise, set last ele as "1"
                                   for idx, noise_idx in enumerate(noise_frame_idx)])
        relations = np.vstack([relations, noise_relation])

        ## weighted node features
        #centroids *= np.array([1.0, 1.0, 0.4, 0.8, 0.5])  ## changed TODO

        return centroids, node_index_labels, frames, relations

    def norm_mat(self, full_matrix, particle_feature_df):
        normalized_matrix = np.zeros_like(full_matrix)  
        for col_idx in range(full_matrix.shape[1]):
            col = full_matrix[:, col_idx]
            col_min, col_max = col.min(), col.max()
            df_min= particle_feature_df[:, col_idx + 3].min()

            if df_min < 0: ## check df not gen_matrix
                # Normalize to (-1, 1)
                normalized_matrix[:, col_idx] = 2 * (col - col_min) / (col_max - col_min) - 1
            else:
                # Normalize to (0, 1)
                normalized_matrix[:, col_idx] = (col - col_min) / (col_max - col_min)

        return normalized_matrix

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

    def gen_noise(self, remain_array, remain_node_idx, remain_frame, num_noise):
        all_centroids = []
        all_frames = []
        all_node_index_labels = []
        noise_frame_idx = np.random.choice(np.arange(self.num_frame_sim), 10, replace=True)
        for frame in range(self.num_frame_sim):
            # Extract points and labels for the current frame
            current_points = remain_array[remain_frame == frame]
            current_node_idx = remain_node_idx[remain_frame == frame]

            if frame in noise_frame_idx:
                # Add  noise particles for the current frame
                selected_points = current_points[np.random.choice(current_points.shape[0], num_noise, replace=True)]
                noise_points = selected_points + np.random.normal(scale=0.1, size=selected_points.shape)
                combined_points = np.vstack([current_points, noise_points])
                # Add node index labels
                if len(all_node_index_labels) == 0:
                    combined_node_idx = np.hstack([current_node_idx, self.num_particle_sim + np.arange(num_noise)])
                else:
                    max_num = max([np.max(node_index_label) for node_index_label in all_node_index_labels])
                    combined_node_idx = np.hstack([current_node_idx, max_num + np.arange(num_noise)])

            else:
                combined_points = current_points
                combined_node_idx = current_node_idx           
            all_centroids.append(combined_points)
            # Add frame labels
            combined_frames = np.full(combined_points.shape[0], frame)
            all_frames.append(combined_frames)          
            all_node_index_labels.append(combined_node_idx)

        return all_centroids, all_frames, all_node_index_labels, noise_frame_idx