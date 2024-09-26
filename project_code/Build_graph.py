import numpy as np
import pandas as pd
from skimage import measure
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from torch_geometric.data import Data
from andi_datasets.models_phenom import models_phenom
from tqdm import tqdm


class Graph_Generator:
    def __init__(
            self, connectivity_radius=None, frame_test=None,
            num_particle_sim=None, len_frame_sim=None, num_frame_sim=None, D_sim=None
        ):
        self.connectivity_radius = connectivity_radius
        self.frame_test = frame_test
        self.max_frame_distance = frame_test[1] -frame_test[0] +1
        self.num_particle_sim = num_particle_sim
        self.len_frame_sim = len_frame_sim
        self.num_frame_sim = num_frame_sim
        self.D_sim = D_sim


    def __call__(self, mode, particle_feature_pth):
        x, node_index_labels, frames, relations = self.get_node_feature(mode, particle_feature_pth, self.frame_test)

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

    def get_node_feature(self, mode, particle_feature_pth, frame_test):
        # For training Simulation: 
        if mode == "train":
            x, node_index_labels, frames, relations = self.build_simulation(particle_feature_pth)
        # For testing video:
        else:
            particle_feature_df = pd.read_csv(particle_feature_pth)
            particle_feature_df['label'] = particle_feature_df.groupby('frame').cumcount()
            test_feature_df = particle_feature_df[
                (particle_feature_df['frame'] >= frame_test[0]) & (particle_feature_df['frame'] <= frame_test[1])]
            
            x = test_feature_df.iloc[:, 1:-1].to_numpy()
            node_index_labels = test_feature_df.iloc[:, -1].to_numpy()
            frames = test_feature_df.iloc[:, 0].to_numpy()

            num_particle = particle_feature_df[(particle_feature_df['frame'] == frame_test[0])].shape[0]
            relations = np.array([[idx, frame_test[0], frame_test[1], 0] for idx in range(num_particle)])
       
        return x, node_index_labels, frames, relations

    def build_simulation(self, particle_feature_pth):
        particle_feature_df = pd.read_csv(particle_feature_pth).to_numpy() 

        traj_simu,_ = models_phenom().multi_state(
            N = self.num_particle_sim,
            L = self.len_frame_sim,
            T = self.num_frame_sim,
            alphas =  [1.2, 0.7],
            Ds = [[10*self.D_sim, 0.1], [1.2*self.D_sim, 0.0]],
            M = [[0.98, 0.02], [0.02, 0.98]],
        )
        stacked_array = np.vstack([np.hstack([np.transpose(traj_simu[i,:, j]) for i in range(traj_simu.shape[0])]) 
                                   for j in range(traj_simu.shape[-1])])
        centroids = np.transpose(stacked_array)
        # add new features:
        other_features = np.zeros((centroids.shape[0], particle_feature_df.shape[1]-3))
        for idx in range(particle_feature_df.shape[1]-3):
            max_feature = max(np.abs(other_features[:, idx]))
            if np.any(particle_feature_df[:, idx+3] < 0):
                other_features[:, idx] = np.random.uniform(-max_feature, max_feature, size=centroids.shape[0])
            else:
                other_features[:, idx] = np.random.uniform(0, max_feature, size=centroids.shape[0])
        x = np.hstack([centroids, other_features])
        node_index_labels = np.hstack([np.arange(traj_simu.shape[1]) for idx in range(traj_simu.shape[0])])
        frames = np.hstack([np.ones(traj_simu.shape[1])* idx  for idx in range(traj_simu.shape[0])])

        relations = np.array([[idx, 0, self.num_frame_sim-1, 0] for idx in range(self.num_particle_sim)])
        return x, node_index_labels, frames, relations


    def compute_connectivity(self, x, frames):
        positions = x[:, :2]
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
        relation_indices = relation_indices[relation_indices[:, 0] != 0]
        relation_mask = np.zeros(len(edge_index), dtype=bool)
        zipped_node = zip(sender, receiver)

        for i, (s, r) in enumerate(zipped_node):
            if np.any((relation_indices == [s, r]).all(1)):
                relation_mask[i] = True
        ground_truth = self_connections_mask | relation_mask
        return ground_truth