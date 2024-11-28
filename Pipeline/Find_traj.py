import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

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