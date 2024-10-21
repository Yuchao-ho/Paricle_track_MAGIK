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
        
    def __call__(self, checkpt_pth, frame_gap: int, dist_gap: float, feature_gap: float, len_thre= 1):
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
        
        torch.cuda.empty_cache()
        ## link segments:  TODO
        """ coord_head, coord_tail, frame_head, frame_tail = [], [], [], []
        coord_head = np.vstack([traj[1][0, :2] for traj in traj_coord])
        coord_tail = np.vstack([traj[1][-1, :2] for traj in traj_coord])
        feature_head = np.vstack([traj[1][0, 2:-1] for traj in traj_coord])
        feature_tail = np.vstack([traj[1][-1, 2:-1] for traj in traj_coord])
        frame_head = np.array([traj[0][0] for traj in traj_coord]).reshape(-1, 1).astype(int)
        frame_tail = np.array([traj[0][-1] for traj in traj_coord]).reshape(-1, 1).astype(int)

        dist_matrix = np.linalg.norm(coord_head[:, np.newaxis, :] - coord_tail[np.newaxis, :, :], axis=-1)
        frame_diff_matrix = frame_head - frame_tail.T   ### 
        feature_diff_matrix = -1 * np.log( np.abs((feature_head - feature_tail * 1e4) / (1e4*feature_head) ))
        
        mask_link = ( (dist_matrix < dist_gap) & (frame_diff_matrix < frame_gap) & 
                     (frame_diff_matrix > 0) & (feature_diff_matrix > feature_gap)
                    ) ###
        index_pair = np.where(mask_link)
        index_pair = set(tuple(indices) for indices in zip(*index_pair))  ###
        pruned_graph = nx.Graph()
        pruned_graph.add_edges_from(index_pair)
        trajectories = list(nx.connected_components(pruned_graph))

        unique_index = set(value for tup in trajectories for value in tup)
        lone_index = [value for value in np.arange(len(traj_coord)) if value not in unique_index]

        new_traj = []
        with tqdm(total=len(trajectories) + len(lone_index), desc="Link Coord") as pbar:
            for link_index in trajectories:
                new_traj_coord = np.vstack([traj_coord[i][1] for i in link_index])
                new_traj_frame = np.vstack([traj_coord[i][0].reshape(-1, 1) for i in link_index])
                sort_idx = np.argsort(new_traj_frame, axis=0).flatten()
                new_traj_frame = new_traj_frame[sort_idx]
                new_traj_coord = new_traj_coord[sort_idx]
                new_traj.append((new_traj_frame, new_traj_coord))
                pbar.update(1)

            if len(lone_index) > 0:
                for idx in lone_index:
                    new_traj_coord = traj_coord[idx][1] 
                    new_traj_frame = traj_coord[idx][0] 
                    new_traj.append((new_traj_frame, new_traj_coord))
                    pbar.update(1) """
        
        return traj_coord


    def generate_pre(self, frame_range, checkpt_pth):
        new_model = Classifier_model()
        classifier = BinaryClassifier(model=new_model, optimizer=Adam(lr=1e-3))
        classifier = classifier.create()
        classifier.model.load_state_dict(torch.load(checkpt_pth, weights_only=True))
        classifier.eval()

        #mode='test'
        graph_Generator = Graph_Generator(
            connectivity_radius=0.02, frame_test=frame_range
        )
        graph = graph_Generator(self.particle_csv_path)
        pred = classifier(graph)
        predictions = pred.detach().numpy() > self.prob_thre
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
    def __call__(self, graph, predictions,):
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
        all_pruned_edges = set()  # Use a set to automatically remove duplicates

        try:
            for i in tqdm(range(0, len(senders), batch_size), desc="Prune Edges (Batched)"):
                batch_senders = senders[i:i+batch_size]
                batch_pruned_edges = self.prune_edges_single_batch(graph, predictions, frame_pairs, batch_senders)
                all_pruned_edges.update(batch_pruned_edges)  # Use update instead of extend
                
                # Clear unnecessary tensors
                del batch_senders, batch_pruned_edges
                torch.cuda.empty_cache()

        except RuntimeError as e:
            print(f"RuntimeError encountered: {e}")
            print("Attempting to free memory and continue...")
            del frame_pairs, senders
            torch.cuda.empty_cache()
        
        return list(all_pruned_edges)  # Convert set back to list before returning

    def prune_edges_single_batch(self, graph, predictions, frame_pairs, batch_senders):
        pruned_edges = set()  # Use a set here as well
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
                
                candidate_min_frame_diff = max(candidates_frame_diff.min(), 1)
                final_mask = candidate & (frame_diff == candidate_min_frame_diff)
                
                candidate_edge_index = graph.edge_index[:, sender_mask][:, final_mask]
                candidate_edge_index = candidate_edge_index.reshape(-1, 2)
                if candidate_edge_index.numel() > 0 and candidate_edge_index.shape[0] == 1:
                    edge = tuple(map(int, candidate_edge_index[0].cpu().numpy()))
                    pruned_edges.add(edge)  # Add the edge as a tuple to the set

            except Exception as e:
                print(f"Error processing sender {sender}: {e}")

        return pruned_edges
    


if __name__ == "__main__":
    gen_video = process_traj(
        video_pth = "/home/user/Project_thesis/Particle_Hana/Video/01_18_Cell7_PC3_cropped3_1_1000ms.avi",
        len_sub = 60,
        len_overlap = 5,
        prob_thre = 0.35,
        particle_csv_pth= "/home/user/Project_thesis/Particle_Hana/Cell7__ground_truth/particle_fea(new).csv",
    )
    gen_video(
        len_thre = 1,
        checkpt_pth = "/home/user/Project_thesis/Particle_Hana/Cell7__ground_truth/model_(blink=2, num=50).pt",
        frame_gap= 6, 
        dist_gap= 10.0, 
        feature_gap= 0.001
    )