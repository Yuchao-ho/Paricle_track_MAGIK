import numpy as np
import pandas as pd
from skimage import measure, exposure
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from torch_geometric.data import Data
from andi_datasets.models_phenom import models_phenom
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

class Graph_Generator:
    def __init__(
            self, connectivity_radius=None, frame_test=None,
            num_particle_sim=None, len_frame_sim=None, num_frame_sim=None, 
            D_sim=None, max_gap = None, prob_noise = None, prop_steady= None
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
        self.prop_steady = prop_steady
        if prop_steady is None:
            self.prop_steady = 0.5


    def __call__(self, particle_feature_pth):
        x, node_index_labels, frames, relations = self.get_node_feature(particle_feature_pth, self.frame_test)

        x, edge_index, edge_attr = self.compute_connectivity(x, frames) ## TODO
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
            x, node_index_labels, frames, relations = traj_simulation(
                particle_feature_pth = particle_feature_pth, prop_steady= self.prop_steady
                )
            self.max_frame_distance =  self.max_gap  ### TODO

        # For testing video:
        else:
            particle_feature_df = pd.read_csv(particle_feature_pth)
            test_feature_df = particle_feature_df[
                (particle_feature_df['frame'] >= frame_test[0]) & (particle_feature_df['frame'] <= frame_test[1])]
            
            x = test_feature_df.iloc[:, 1:].to_numpy()   
            #node_index_labels = np.arange(x.shape[0]) / x.shape[0]
            node_index_labels = np.arange(x.shape[0])
            frames = test_feature_df.iloc[:, 0].to_numpy()

            num_particle = particle_feature_df[(particle_feature_df['frame'] == frame_test[0])].shape[0]
            relations = np.array([[idx, frame_test[0], frame_test[1], 0] for idx in range(num_particle)])
        
        return x, node_index_labels, frames, relations

    def compute_connectivity(self, x, frames):
        positions = x[:, :2]   
        distances = np.linalg.norm(positions[:, None] - positions, axis=-1)  # Broadcast

        frame_diff = (frames[:, None] - frames) * -1  # Broadcast
        mask = ((distances <= self.connectivity_radius)) & ((frame_diff <= self.max_frame_distance) & (frame_diff > 0))
        edge_index = np.argwhere(mask) 
        
        edge_attr = distances[mask] 
        edge_attr = (edge_attr - edge_attr.min()) / edge_attr.max()
        #edge_attr = exposure.equalize_hist(edge_attr)
        ## normalize feature map
        if self.frame_test is None:
            x[:, :2] /= self.len_frame_sim
        else:
            x[:, :2] /= np.array([1314,1054])
        
        """ for idx in range(2, x.shape[1]):
            x[:, idx] = (x[:, idx] - x[:, idx].min()) /(x[:, idx].max() - x[:, idx].min()) """
        x[:, 2] = x[:, 2]/ 255
        #x[:, 3] = x[:, 3]/ 255    
        #x[:, 4] = x[:, 4]/ 255 # TODO
        

        return x, edge_index, edge_attr

    def compute_ground_truth(self, indices, edge_index, relation):
        sender = indices[edge_index[:, 0]]
        receiver = indices[edge_index[:, 1]]
        self_connections_mask = sender == receiver
        relation[:,-1] = 0
        relation_indices = relation[:, [-1, 0]]
        #relation_indices = relation_indices[relation_indices[:, 0] == 0]  ## Old Way to build
        relation_indices = relation_indices[relation_indices[:, -1] != 0]   ## choose merging path
        relation_mask = np.zeros(len(edge_index), dtype=bool)
        zipped_node = zip(sender, receiver)

        for i, (s, r) in enumerate(zipped_node):
            if np.any((relation_indices == [s, r]).all(1)):
                relation_mask[i] = True
        #ground_truth = self_connections_mask | relation_mask
        ground_truth = self_connections_mask
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
        
    def __call__(self, particle_feature_pth, prop_steady):
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
            mean_val = (min_val + max_val) / 2  
            std_dev = (max_val - min_val) / 4   
            other_feature_1[:, col_idx] = np.random.normal(loc=mean_val, scale=std_dev, size=self.num_particle_sim)
            other_feature_1[:, col_idx] = np.clip(other_feature_1[:, col_idx], min_val, max_val)
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
            if idx % 1 == 0:  # 5 TODO
                # Initialize noise array
                noise = np.zeros((self.num_particle_sim, n_features))
                moving_choices = np.random.choice(
                    possible_choices,
                    size=(num_move, n_features),
                    p=[0.8, 0.2]  # [no_noise_prob, noise_prob] [0.2, 0.8]
                )
                moving_mask = (moving_choices == 1)
                
                # Generate choices for static particles (remaining particles)
                # 10% probability for noise
                static_choices = np.random.choice(
                    possible_choices,
                    size=(self.num_particle_sim - num_move, n_features),
                    p=[0.8, 0.2]  # [no_noise_prob, noise_prob]
                )
                static_mask = (static_choices == 1)
                
                # Generate and apply noise for both groups
                for col_idx in range(n_features):
                    # Handle moving particles (higher variance)
                    if moving_mask[:, col_idx].any():
                        noise[:num_move, col_idx][moving_mask[:, col_idx]] = np.random.normal(
                            0, 50, moving_mask[:, col_idx].sum()
                        )
                    
                    #Handle static particles (lower variance)
                    if static_mask[:, col_idx].any():
                       noise[num_move:, col_idx][static_mask[:, col_idx]] = np.random.normal(
                           0, 5, static_mask[:, col_idx].sum()
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

        """ all_centroids, all_frames, all_node_index_labels, noise_frame_idx = self.gen_noise(
            remain_array, remain_node_idx, remain_frame, num_noise
            ) """
        # Stack the results across all frames
        centroids = np.vstack(all_centroids)
        frames = np.hstack(all_frames)
        node_index_labels = np.hstack(all_node_index_labels)

        ## Get relations 
        relations = np.array([[idx, 0, self.num_frame_sim-1, 0] for idx in range(self.num_particle_sim)])  ## For real, set last ele as "0"
        """ noise_relation = np.array([[idx+self.num_particle_sim, noise_idx, noise_idx, 0]   ## For noise, set last ele as "1"
                                   for idx, noise_idx in enumerate(noise_frame_idx)])
        relations = np.vstack([relations, noise_relation]) """

        return centroids, node_index_labels, frames, relations


    def simu_traj(self,prop_steady):
        ## generate moving particles
        traj_move,_ = models_phenom().multi_state(
                    N = int(self.num_particle_sim * (1-prop_steady)),
                    L = self.len_frame_sim,
                    T = self.num_frame_sim,
                    alphas =  [1.2, 0.7],  
                    Ds = [[24*self.D_sim, 0.1], [0.05*self.D_sim, 0]],  ## 24(previous) TODO 0.05*self.D_sim, 0
                    M = [[0.9, 0.10], [0.90, 0.10]],  ## [[0.98, 0.02], [0.02, 0.98]]
                ) 
        ## generate steady particles
        num_steady = int((self.num_particle_sim * prop_steady) * 0.7) ## 0.8 TODO
        traj_steady, _ = models_phenom().single_state(
                    N = num_steady, 
                    L = self.len_frame_sim / 1, ## / 10  TODO
                    T = self.num_frame_sim,
                    Ds = [0.02*self.D_sim, 0.1], # Mean and variance
                    alphas = 0.5
                )
        ## set steady particle regions
        #traj_steady += self.len_frame_sim / 10 TODO

        ## generate trapped particles
        num_trap = int((self.num_particle_sim * prop_steady) * 0.2)
        traj_trap, _ = models_phenom().single_state(
                    N = num_trap, 
                    L = self.len_frame_sim / 5, ## / 10  TODO
                    T = self.num_frame_sim,
                    Ds = [0.1*self.D_sim, 0.1], # Mean and variance
                    alphas = 1.0 ## 0.5 TODO 
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

    def gen_noise(self, remain_array, remain_node_idx, remain_frame, num_noise):
        all_centroids = []
        all_frames = []
        all_node_index_labels = []
        noise_frame_idx = np.random.choice(np.arange(self.num_frame_sim), 1, replace=True)
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
    

if __name__ == "__main__":
    graph_Generator = Graph_Generator(
        connectivity_radius=20, frame_test= (260, 360)
    )
    graph_Generator = Graph_Generator(
        connectivity_radius=20, num_particle_sim= 100, 
        len_frame_sim= 1314, num_frame_sim= 80, 
        D_sim= 1, max_gap= 5, prob_noise= 0.05, prop_steady = 0.5
    )
    train_graph = graph_Generator(particle_feature_pth= "/home/user/Project_thesis/Particle_Hana/Cell7__ground_truth/particle_fea(mean_intens)(original).csv")