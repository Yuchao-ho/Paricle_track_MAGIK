import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from Pipeline.Simu_traj import simulate_traj

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

        x, edge_index, edge_attr = self.compute_connectivity(x, frames) 
        edge_ground_truth = self.compute_ground_truth(node_index_labels, edge_index, relations)
        edge_index = edge_index.T
        #edge_attr = edge_attr[:, None]
        edge_ground_truth = edge_ground_truth[:, None]

        #print(f"true number: {np.sum(edge_ground_truth)}")
        #print(f"false number: {edge_ground_truth.size - np.sum(edge_ground_truth)}")
        #pos_weight = (edge_ground_truth.size - np.sum(edge_ground_truth)) / np.sum(edge_ground_truth)

        graph = Data(
            x=torch.tensor(x, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float),
            distance=torch.tensor(edge_attr[:,0][:,None], dtype=torch.float),
            angle=torch.tensor(edge_attr[:,1][:,None], dtype=torch.float),
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
        """ positions = x[:, :2]   
        distances = np.linalg.norm(positions[:, None] - positions, axis=-1)  # Broadcast

        frame_diff = (frames[:, None] - frames) * -1  # Broadcast
        mask = ((distances <= self.connectivity_radius)) & ((frame_diff <= self.max_frame_distance) & (frame_diff > 0))
        edge_index = np.argwhere(mask) 
        
        edge_attr = distances[mask] 
        edge_attr = (edge_attr - edge_attr.min()) / edge_attr.max() """
        
        positions = x[:, :2]
        deltas = positions[:, None] - positions  # Pairwise differences between positions
        distances = np.linalg.norm(deltas, axis=-1)  # Euclidean distance

        # Calculate frame differences
        frame_diff = (frames[:, None] - frames) * -1  # Ensure direction is future frames
        mask = ((distances <= self.connectivity_radius)) & ((frame_diff <= self.max_frame_distance) & (frame_diff > 0))

        # Find valid edges
        edge_index = np.argwhere(mask)

        # Normalize distances (optional but recommended)
        edge_distances = distances[mask]
        edge_distances_normalized = (edge_distances - edge_distances.min()) / (edge_distances.max() - edge_distances.min())
        # Compute θ_ij = atan2(Δy, Δx) and normalize to [0, 2π)
        angles = np.arctan2(deltas[..., 1], deltas[..., 0])  # Angle in radians (-π, π)
        angle_2pi = np.mod(angles, 2 * np.pi)  # Normalize to range [0, 2π)
        angle_normalized = angle_2pi[mask] / (2 * np.pi)  # Normalize to range [0, 1]

        # Define edge attributes as [distance, θ_ij]
        edge_attr = np.vstack((edge_distances_normalized, angle_normalized)).T

        if self.frame_test is None:
            x[:, :2] /= self.len_frame_sim
        else:
            x[:, :2] /= np.array([1314,1054])

        """ for idx in range(2, x.shape[1]):
        x[:, idx] = (x[:, idx] - x[:, idx].min()) /(x[:, idx].max() - x[:, idx].min()) """
        x[:, 2] = x[:, 2]/ 255# TODO

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
