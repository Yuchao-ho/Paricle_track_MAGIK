import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import os
from tqdm import tqdm
import networkx as nx
from Generate_filter_csv import gen_fil_csv
from Build_graph import Graph_Generator
from MAGIK_model import Classifier_model
from deeplay import BinaryClassifier, Adam

class process_traj:

    def __init__(self, video_pth, particle_csv_pth, position_csv):
        self.video_path = video_pth
        self.particle_csv_path = particle_csv_pth
        self.position_csv_path = position_csv

        if not os.path.exists(self.particle_csv_path):
            gen_csv = gen_fil_csv(
                video_pth=self.video_path, position_pth=self.position_csv_path, side_len=30
                )
            gen_csv(
                feature_list=["area", "orientation"], output_pth=self.particle_csv_path
                )
        ## divide frame list
        particle_csv = pd.read_csv(self.particle_csv_path)
        frame_length = int(particle_csv["frame"].max())
        self.frame_range_list = [(start, min(start + 50, frame_length))
                                 for start in range(0, frame_length + 1, 45)]
        
    def __call__(self, checkpt_pth, len_thre= 8):
        ## generate graph list
        graph_list = []
        prediction_list = []
        with tqdm(total=len(self.frame_range_list), desc="Test Video") as pbar:
            for frame_range in self.frame_range_list:
                graph, predictions = self.generate_pre(frame_range, checkpt_pth)
                graph_list.append(graph)
                prediction_list.append(predictions)
                pbar.update(1)

        ## combine graphs & predictions
        combined_graph = self.combine_graph_list(graph_list)
        combined_pre = np.concatenate(prediction_list, axis=0)

        ## build traj
        post_processor = compute_trajectories()
        trajectories = post_processor(combined_graph, combined_pre.squeeze())
        filter_trajectories = [traj for traj in trajectories if len(traj) > len_thre]
        traj_coord = []
        with tqdm(total=len(filter_trajectories), desc="Gen Coord") as pbar:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            for traj in filter_trajectories:
                frames = combined_graph.frames[list(traj)].to(device)
                traj_tensor = torch.tensor(list(traj), device=device)
                sorted_frames, sorted_idx = torch.sort(frames)
                sorted_traj = traj_tensor[sorted_idx]
                coordinates = combined_graph.x.to(device)[sorted_traj]
                coordinates[:, 0] = coordinates[:, 0]*1314
                coordinates[:, 1] = coordinates[:, 1]*1054
                traj_coord.append((sorted_frames.cpu().numpy(), coordinates.cpu().numpy()))
                pbar.update(1)

        return traj_coord
    
    def generate_pre(self, frame_range, checkpt_pth):
        new_model = Classifier_model()
        classifier = BinaryClassifier(model=new_model, optimizer=Adam(lr=1e-3))
        classifier = classifier.create()
        classifier.model.load_state_dict(torch.load(checkpt_pth, weights_only=True))
        classifier.eval()

        mode='test'
        graph_Generator = Graph_Generator(
            connectivity_radius=0.02, frame_test=frame_range,
            num_particle_sim= 100, len_frame_sim= 1034, num_frame_sim= 60, 
            D_sim= 0.1
        )
        graph = graph_Generator(mode, self.particle_csv_path)
        pred = classifier(graph)
        predictions = pred.detach().numpy() > 0.5
        return graph, predictions

    def combine_graph_list(self, graph_list):
        start = graph_list[0]
        with tqdm(total=len(graph_list)-1, desc="Combine Graph") as pbar:
            for idx in range(1, len(graph_list)):
                start = self.combine_graph(start, graph_list[idx])
                pbar.update(1)
        return start

    def combine_graph(self, graph1, graph2):
        ## check shared nodes
        x1, x2 = graph1.x, graph2.x
        shared_nodes_mask = (x2.unsqueeze(1) == x1.unsqueeze(0)).all(-1).any(1)
        shared_nodes_indices = torch.nonzero(shared_nodes_mask).squeeze()
        combined_x = torch.cat([x1, x2[~shared_nodes_mask]], dim=0)

        ## shift indices of shared edges
        num_nodes_graph1 = graph1.num_nodes
        edge_index2_updated = graph2.edge_index.clone()
        edge_index2_updated[0, :] += (num_nodes_graph1 - len(shared_nodes_indices))
        edge_index2_updated[1, :] += (num_nodes_graph1 - len(shared_nodes_indices))
        combined_edge_index = torch.cat([graph1.edge_index, edge_index2_updated], dim=1)
        
        ## combine edge_attr, distance
        combined_edge_attr = torch.cat([graph1.edge_attr, graph2.edge_attr], dim=0)
        combined_distance = torch.cat([graph1.distance, graph2.distance], dim=0)

        ## check shared frames
        frames1 = graph1.frames.clone()
        frames2 = graph2.frames.clone()
        frame_mask = (frames2.unsqueeze(1) == frames1.unsqueeze(0)).any(1)
        combined_frames = torch.cat([frames1, frames2[~frame_mask]], dim=0)
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
        return combined_graph
    
class compute_trajectories:
    def __call__(self, graph, predictions):
        pruned_edges = self.prune_edges_batch(graph, predictions)

        pruned_graph = nx.Graph()
        pruned_graph.add_edges_from(pruned_edges)

        trajectories = list(nx.connected_components(pruned_graph))

        return trajectories

    def prune_edges_batch(self, graph, predictions, batch_size=100):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        graph.edge_index = graph.edge_index.to(device)
        graph.frames = graph.frames.to(device)
        predictions = torch.tensor(predictions, device=device)
        
        frame_pairs = torch.stack([
            graph.frames[graph.edge_index[0]], 
            graph.frames[graph.edge_index[1]]
        ], dim=1)

        senders = torch.unique(graph.edge_index[0])
        all_pruned_edges = []

        try:
            for i in tqdm(range(0, len(senders), batch_size), desc="Prune Edges (Batched)"):
                batch_senders = senders[i:i+batch_size]
                batch_pruned_edges = self.prune_edges_single_batch(graph, predictions, frame_pairs, batch_senders)
                all_pruned_edges.extend(batch_pruned_edges)
                
                # Clear unnecessary tensors
                del batch_senders, batch_pruned_edges
                torch.cuda.empty_cache()

        except RuntimeError as e:
            print(f"RuntimeError encountered: {e}")
            print("Attempting to free memory and continue...")
            del frame_pairs, senders
            torch.cuda.empty_cache()
        return all_pruned_edges

    def prune_edges_single_batch(self, graph, predictions, frame_pairs, batch_senders):
        pruned_edges = []
        for sender in batch_senders:
            sender_mask = graph.edge_index[0] == sender
            if not torch.any(sender_mask):
                continue

            try:
                candidate = predictions[sender_mask]
                frame_diff = frame_pairs[sender_mask, 1] - frame_pairs[sender_mask, 0]
                
                if not torch.any(candidate):
                    continue
                
                candidates_frame_diff = frame_diff[candidate]
                if candidates_frame_diff.numel() == 0:
                    continue
                
                candidate_min_frame_diff = candidates_frame_diff.min()
                final_mask = candidate & (frame_diff == candidate_min_frame_diff)
                
                candidate_edge_index = graph.edge_index[:, sender_mask][:, final_mask]
                candidate_edge_index = candidate_edge_index.reshape(-1, 2)
                if candidate_edge_index.numel() > 0 and candidate_edge_index.shape[0] == 1:
                    pruned_edges.append(tuple(*candidate_edge_index.cpu().numpy()))

            except Exception as e:
                print(f"Error processing sender {sender}: {e}")

        return pruned_edges


if __name__ == "__main__":
    combinator_traj = process_traj(
        video_pth= "/home/user/Project_thesis/Particle_Hana/01_18_Cell7_PC3_cropped3_1_1000ms.avi",
        particle_csv_pth= "/home/user/Project_thesis/Particle_Hana/Cell7__ground_truth/particle_feature.csv",
        position_csv= "/home/user/Project_thesis/Particle_Hana/Cell7__ground_truth/lodestar_detection.csv"
    )
    trajectories = combinator_traj(
        checkpt_pth= "/home/user/Project_thesis/Particle_Hana/Cell7__ground_truth/model_pretrained.pt"
    )