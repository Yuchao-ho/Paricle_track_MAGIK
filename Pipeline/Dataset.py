from torch.utils.data import Dataset
import torch
import numpy as np
from torch_geometric.data import Data
import math

class Tracing_Dataset(Dataset):
    def __init__(self, graph, window_size, dataset_size, transform=None):
        self.graph = graph

        self.window_size = window_size # (1)
        self.dataset_size = dataset_size

        frames, edge_index = graph.frames, graph.edge_index
        self.pair_frames = torch.stack(
            [frames[edge_index[0, :]], frames[edge_index[1, :]]], axis=1
        )
        self.frames = frames
        self.max_frame = frames.max()

        self.transform = transform

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        frame_idx = np.random.randint(self.window_size, self.max_frame + 1) # (2)

        start_frame = frame_idx - self.window_size
        node_mask = (self.frames >= start_frame) & (self.frames < frame_idx) # (3)
        x = self.graph.x[node_mask] # (4)

        edge_mask = (self.pair_frames >= start_frame) & (self.pair_frames < frame_idx) # (5)
        edge_mask = edge_mask.all(axis=1)

        edge_index = self.graph.edge_index[:, edge_mask] # (6)  ## TODO
        if edge_index.numel() > 0:
            edge_index -= edge_index.min()  # Avoid error when edge_index is empty
        """ else:
            print("Warning: edge_index is empty for idx:", idx)
            return None  # or handle this case as needed
        edge_index -= edge_index.min() """

        edge_attr = self.graph.edge_attr[edge_mask] # (7)

        # sample ground truth edges
        ground_truth_edges = self.graph.y[edge_mask] # (8)

        graph = Data( # (9)
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            distance=edge_attr[:,0][:,None],
            angle = edge_attr[:,1][:,None],
            y=ground_truth_edges,
        )

        if self.transform: # (10)
            graph = self.transform(graph)

        return graph


class RandomRotation: # (1)
    def __call__(self, graph):
        graph = graph.clone()
        centered_features = graph.x[:, :2] - 0.5

        angle = np.random.rand() * 2 * math.pi
        rotation_matrix = torch.tensor(
            [
                [math.cos(angle), -math.sin(angle)],
                [math.sin(angle), math.cos(angle)],
            ]
        )
        rotated_features = torch.matmul(centered_features, rotation_matrix)

        graph.x[:, :2] = rotated_features + 0.5
        return graph

class RandomFlip: # (2)
    def __call__(self, graph):
        graph = graph.clone()
        centered_features = graph.x[:, :2] - 0.5

        if np.random.randint(2):
            centered_features[:, 0] *= -1

        if np.random.randint(2):
            centered_features[:, 1] *= -1


        graph.x[:, :2] = centered_features + 0.5
        return graph