import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from collections import defaultdict
from Pipeline.Generate_filter_csv import gen_fil_csv
from Pipeline.Build_graph import Graph_Generator
from Pipeline.MAGIK_model import Classifier_model
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

        ## combine graphs & predictions
        combined_graph, combined_pre = self.combine_graph_list(graph_list, prediction_list)
        #combined_pre = np.concatenate(prediction_list, axis=0)

        ## build traj
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

                """ unique_frames = []
                unique_coordinates = []
                last_frame = None
                for i, frame in enumerate(sorted_frames):
                    if last_frame is None:
                        unique_frames.append(frame)
                        unique_coordinates.append(coordinates[i])
                        last_frame = frame
                    
                    elif frame != last_frame:
                        last_coord = unique_coordinates[-1]
                        frame_mask = sorted_frames == frame
                        coord_frame = coordinates[frame_mask, :]
                        distances = np.linalg.norm(coord_frame.cpu().numpy() - last_coord.cpu().numpy(), axis=1)
                        min_distance_index = np.argmin(distances)
                        unique_frames.append(frame)
                        unique_coordinates.append(coord_frame[min_distance_index] )
                        last_frame = frame

                # Convert lists to tensors
                unique_frames = torch.stack(unique_frames)
                unique_coordinates = torch.stack(unique_coordinates)
                traj_coord.append((unique_frames.cpu().numpy(), unique_coordinates.cpu().numpy())) """

                traj_coord.append((frames.cpu().numpy(), coordinates.cpu().numpy()))
                pbar.update(1)
        
        torch.cuda.empty_cache()
        
        return traj_coord


    def generate_pre(self, frame_range, connect_radius, classifier):
        #mode='test'
        graph_Generator = Graph_Generator(
            connectivity_radius= connect_radius, frame_test=frame_range
        )
        graph = graph_Generator(self.particle_csv_path)
        pred = classifier(graph)
        #predictions = pred.detach().numpy() > self.prob_thre
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
        # check shared nodes
        x1, x2 = graph1.x, graph2.x  
        x1_pos, x2_pos = x1[:, :2], x2[:, :2]  ## TODO
        shared_mask = (x2_pos.unsqueeze(1) == x1_pos.unsqueeze(0)).all(-1)
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

        # Combine predictions
        combined_pre = np.concatenate((pre1, pre2[~edge_mask]), axis=0)
        return combined_graph, combined_pre
    
class compute_trajectories:
    def __call__(self, graph, predictions,):
        pruned_edges = self.prune_edges_batch(graph, predictions)
        adj_dict = self.build_adjacency_dict(pruned_edges)
        max_value = max(max(tup) for tup in pruned_edges)
        # Use a more memory-efficient boolean array
        id_mask = np.zeros(max_value + 1, dtype=np.bool_)
        path_ls = []
        with tqdm(total=max_value + 1, desc="Gen Traj") as pbar:
            for idx in range(max_value + 1):
                if not id_mask[idx]:
                    path = self.recurr(adj_dict, idx, id_mask)
                    if path:
                        path_ls.append(path)
                pbar. update(1)       

        return path_ls

    def prune_edges_batch(self, graph, predictions, batch_size=100):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        graph.edge_index = graph.edge_index.to(device)
        graph.frames = graph.frames.to(device)
        predictions = torch.tensor(predictions, device=device)
        
        frame_pairs = torch.stack([
            graph.frames[graph.edge_index[0]], 
            graph.frames[graph.edge_index[1]]
        ], dim=1)

        #senders = torch.unique(graph.edge_index[0])  TODO
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
                candidate_mask = candidate > 0.0
                frame_diff = frame_pairs[sender_mask, 1] - frame_pairs[sender_mask, 0]
                
                if not torch.any(candidate):
                    continue
                
                candidates_frame_diff = frame_diff[candidate_mask] # TODO
                if candidates_frame_diff.numel() == 0:
                    continue
                
                candidate_min_frame_diff = max(candidates_frame_diff.min(), 1)
                
                final_mask = candidate_mask & (frame_diff == candidate_min_frame_diff) 
                candidate = candidate[final_mask]
                
                max_index = torch.argmax(candidate)
                candidate_edge_index = graph.edge_index[:, sender_mask][:, final_mask]
                #sorted_traj = torch.cat([candidate_edge_index[1, :], sender.unsqueeze(0)], dim=0)
                #coordinates = graph.x.to("cuda")[sorted_traj]

                #candidate_edge_index = candidate_edge_index.reshape(-1, 2)
                if candidate_edge_index.numel() > 0 and candidate_edge_index.shape[1] >= 1:
                    edge = tuple(candidate_edge_index[:,max_index].cpu().numpy())
                    pruned_edges.add(edge)  # Add the edge as a tuple to the set

            except Exception as e:
                print(f"Error processing sender {sender}: {e}")

        return pruned_edges
    
    def build_adjacency_dict(self, prune_edges):
        adj_dict = defaultdict(list)
        for i, j in prune_edges:
            adj_dict[i].append(j)
        return adj_dict
 
    def recurr(self, adj_dict, start_node, id_mask):
        stack = [start_node]
        path = []

        while stack:
            node = stack.pop()
            if not id_mask[node]:
                path.append(node)
                id_mask[node] = True
                # Add neighbors in reverse order to maintain path sequence
                stack.extend(adj_dict[node][::-1])

        return path


if __name__ == "__main__":
    gen_video = process_traj(
        video_pth = "/home/user/Project_thesis/Particle_Hana/Video/01_18_Cell7_PC3_cropped3_1_1000ms.avi",
        len_sub = 30,
        len_overlap = 3,
        prob_thre = 0.5,
        particle_csv_pth= "/home/user/Project_thesis/Particle_Hana/Cell7__ground_truth/particle_fea(mean_intens)(orient).csv",
    )
    gen_video(
        len_thre = 2,
        checkpt_pth = "/home/user/Project_thesis/Particle_Hana/Cell7__ground_truth/model_(Consec(mean), num=50, w_size=20)(500).pt",
        connect_radius = 30
    )