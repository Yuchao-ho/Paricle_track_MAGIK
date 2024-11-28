import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from Pipeline.Generate_filter_csv import gen_fil_csv
from Pipeline.Build_graph_re import Graph_Generator
from Pipeline.MAGIK_model import Classifier_model
from Pipeline.Find_traj import compute_trajectories
from deeplay import BinaryClassifier, Adam

class process_traj:

    def __init__(
            self, video_pth=None, particle_csv_pth=None, position_csv=None,
            len_sub=None, len_overlap=None, prob_thre= None):
        self.video_path = video_pth
        self.particle_csv_path = particle_csv_pth
        self.position_csv_path = position_csv
        self.len_sub = len_sub
        self.len_overlap = len_overlap
        self.prob_thre = prob_thre

        if self.particle_csv_path is None:
            gen_csv = gen_fil_csv(
                video_pth=self.video_path, position_pth=self.position_csv_path, side_len=30
                )
            gen_csv(
                feature_list=["area", "orientation"], output_pth=self.particle_csv_path
                )
        ## divide frame list
        particle_csv = pd.read_csv(self.particle_csv_path)
        frame_length = int(particle_csv["frame"].max())
        self.frame_range_list = [(start, min(start + self.len_sub, frame_length))
                                 for start in range(0, frame_length + 1, (self.len_sub-self.len_overlap))]
        
    def __call__(self, checkpt_pth, connect_radius, len_thre):
        ## generate graph list
        graph_list, prediction_list = self.test_video(checkpt_pth, connect_radius)

        ## combine graphs & predictions
        combined_graph, combined_pre = self.combine_graph_list(graph_list, prediction_list)

        ## build traj
        return self.build_traj(combined_graph, combined_pre, len_thre)

    def test_video(self, checkpt_pth, connect_radius):
        graph_list, prediction_list  = [], []
        new_model = Classifier_model()
        classifier = BinaryClassifier(model=new_model, optimizer=Adam(lr=1e-3))
        classifier = classifier.create()
        classifier.model.load_state_dict(torch.load(checkpt_pth, weights_only=True))
        classifier.eval()

        with tqdm(total=len(self.frame_range_list), desc="Test Video") as pbar:
            for frame_range in self.frame_range_list:
                graph, predictions = self.generate_pre(frame_range, connect_radius, classifier)
                graph_list.append(graph)
                prediction_list.append(predictions)
                pbar.update(1)

        return graph_list, prediction_list

    def generate_pre(self, frame_range, connect_radius, classifier):

        graph_Generator = Graph_Generator(
            connectivity_radius= connect_radius, frame_test=frame_range
        )
        graph = graph_Generator(self.particle_csv_path)
        pred = classifier(graph)
        predictions = pred.detach().numpy()
        predictions = np.where(predictions < self.prob_thre, 0.0, predictions)  # TODO

        return graph, predictions

    def combine_graph_list(self, graph_list, prediction_list):

        start_graph, start_pre = graph_list[0], prediction_list[0]
        with tqdm(total=len(graph_list)-1, desc="Combine Graph") as pbar:
            for idx in range(1, len(graph_list)):
                start_graph, start_pre = self.combine_graph(
                    start_graph, graph_list[idx], start_pre, prediction_list[idx]
                    )
                pbar.update(1)
        return start_graph, start_pre

    def combine_graph(self, graph1, graph2, pre1, pre2):

        x1, x2 = graph1.x, graph2.x  
        x1_pos, x2_pos = x1[:, :2], x2[:, :2]  ## TODO
        frames1, frames2 = graph1.frames.clone(), graph2.frames.clone()
        # Check shared nodes: same position and same frame
        position_mask = (x2_pos.unsqueeze(1) == x1_pos.unsqueeze(0)).all(-1)  
        frame_mask = (frames2.unsqueeze(1) == frames1.unsqueeze(0))  
        shared_mask = position_mask & frame_mask

        #shared_mask = (x2_pos.unsqueeze(1) == x1_pos.unsqueeze(0)).all(-1)
        shared_x2_idx = shared_mask.any(1).nonzero(as_tuple=False).squeeze()
        shared_x1_idx = shared_mask.any(0).nonzero(as_tuple=False).squeeze()
        unique_x2_mask = ~shared_mask.any(1)
        unique_x2_dict = {x2_idx: i for i, x2_idx in enumerate(unique_x2_mask.nonzero(as_tuple=False).squeeze().tolist())}
        shared_nodes_dict = dict(zip(shared_x2_idx.tolist(), shared_x1_idx.tolist()))

        combined_x = torch.cat([x1, x2[unique_x2_mask]], dim=0)

        # Shift indices of shared edges
        num_nodes_graph1 = graph1.num_nodes
        edge_idx2 = graph2.edge_index.clone()
        ## Create edge mask for shared edges
        #edge_mask = ((edge_idx2.unsqueeze(-1) == shared_x2_idx).any(-1)).all(0)  maybe wrong
        edge_mask0 = ((edge_idx2[0,:].unsqueeze(-1) == shared_x2_idx).any(-1))
        edge_mask1 = ((edge_idx2[1,:].unsqueeze(-1) == shared_x2_idx).any(-1))
        edge_mask = (edge_mask0) & (edge_mask1)
        edge_idx2 = edge_idx2[:, ~edge_mask]
        # Create a lookup tensor for shared nodes mapping
        lookup = torch.arange(max(edge_idx2.max().item(), max(shared_nodes_dict.keys())) + 1)
        for k, v in shared_nodes_dict.items():
            lookup[k] = v
        edge_idx2_mapped = lookup[edge_idx2]

        not_in_dict_mask = (edge_idx2 == lookup[edge_idx2])  # True for new nodes

        # Apply unique_x2_dict mapping and shift by num_nodes_graph1
        mapped_values = torch.tensor([unique_x2_dict[idx.item()] for idx in edge_idx2[not_in_dict_mask].flatten()])
        edge_idx2_mapped[not_in_dict_mask] = mapped_values.view(edge_idx2[not_in_dict_mask].shape) + num_nodes_graph1

        # Combine edge indices with mapped values
        combined_edge_index = torch.cat([graph1.edge_index, edge_idx2_mapped], dim=1)
       
        # combine edge_attr, distance
        combined_edge_attr = torch.cat([graph1.edge_attr, graph2.edge_attr[~edge_mask]], dim=0)
        combined_distance = torch.cat([graph1.distance, graph2.distance[~edge_mask]], dim=0)

        # check shared frames
        combined_frames = torch.cat([frames1, frames2[~frame_mask.any(1)]], dim=0)
        ## check ground truth
        combined_y = torch.cat([graph1.y, graph2.y], dim=0)

        combined_graph = Data(
                x=combined_x,
                edge_index=combined_edge_index,
                edge_attr=combined_edge_attr,
                distance=combined_distance,
                frames=combined_frames,
                y=combined_y
            )

        # Combine predictions
        combined_pre = np.concatenate((pre1, pre2[~edge_mask]), axis=0)
        return combined_graph, combined_pre

    def build_traj(self, combined_graph, combined_pre, len_thre):
        post_processor = compute_trajectories()
        trajectories = post_processor(combined_graph, combined_pre.squeeze())
        filter_trajectories = [traj for traj in trajectories if len(traj) > len_thre]
        traj_coord = []
        with tqdm(total=len(filter_trajectories), desc="Gen Coord") as pbar:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            for traj in filter_trajectories:
                frames = combined_graph.frames[list(traj)].to(device)
                traj_tensor = torch.tensor(list(traj), device=device, dtype=torch.int64)
                coordinates = combined_graph.x.to(device)[traj_tensor]
                coordinates[:, 0] = coordinates[:, 0]*1314
                coordinates[:, 1] = coordinates[:, 1]*1054
                traj_coord.append((frames.cpu().numpy(), coordinates.cpu().numpy()))
                pbar.update(1)
        
        torch.cuda.empty_cache()
        return traj_coord