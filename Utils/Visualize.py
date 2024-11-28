import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


class visualization:

    def __init__(
            self, video_path=None, particle_csv_path=None, 
            frame_test: tuple=None, frame_region: list[tuple]=None,
            ) :
        self.video_path = video_path
        self.particle_csv_path = particle_csv_path
        self.frame_test = frame_test
        self.frame_region = frame_region

    def gen_video(self, trajectories, fps, output_path):
        image_list = self.load_frame()
        ## generate .avi video           
        track_img_list = self.gen_track_avi(trajectories, self.frame_test, self.frame_region, image_list)
        height, width, _ = track_img_list[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for img in track_img_list:
            video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        video.release()

    @staticmethod
    def traj_distrib(trajectories):
        frame_diff = []
        for frames, coord in trajectories:
            frame_diff.append(len(frames))
            
        _, bins = np.histogram(frame_diff, bins=30)  
        plt.hist(frame_diff, bins=bins, alpha=0.5)
        plt.xlabel("num of points")
        plt.ylabel("Frequency")
        plt.title("Len of Trajectories")
        plt.show()

    @staticmethod
    def plot_msd(all_msd):
        fig, ax = plt.subplots()
        for arr in all_msd:
            time_lags = arr[0, :]  
            msds = arr[1, :]  
            mask = time_lags <= 600      
            time_lags = time_lags[mask]
            msds = msds[mask]
            msds = np.log10(msds)
            time_lags = np.log10(time_lags)
            #ax.plot(time_lags, msds, linestyle='-')
            ax.scatter(time_lags, msds, s=1)

        ax.set_xlabel("Time Lag")
        ax.set_ylabel("MSD")
        ax.set_title("MSD vs Time Lag")
        ax.legend()
        plt.show()

    @staticmethod
    def plot_clusters(arrays, labels, kmeans):
        n_clusters = len(np.unique(labels))
        
        # Create subplot for each cluster
        fig, axs = plt.subplots(n_clusters, 1, figsize=(12, 4*n_clusters))
        if n_clusters == 1:
            axs = [axs]
        
        for i in range(n_clusters):
            # Plot original curves for this cluster
            cluster_indices = np.where(labels == i)[0]
            
            for idx in cluster_indices:
                axs[i].plot(arrays[idx][0], arrays[idx][1], 'gray', alpha=0.3)
                
            # Plot cluster center (denormalized)
            center_curve = kmeans.cluster_centers_[i]
            axs[i].plot(arrays[0][0], center_curve, 'r-', linewidth=2, label='Cluster Center')
            
            axs[i].set_title(f'Cluster {i} (n={len(cluster_indices)})')
            axs[i].set_xlabel('Time Lag')
            axs[i].set_ylabel('MSD')
            axs[i].legend()
        
        plt.tight_layout()
        plt.show()


    def load_frame(self, ):
        image_list = []
        video = cv2.VideoCapture(self.video_path)
        while True:
            ret, frame = video.read()
            if not ret:
                break
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image_list.append(np.array(gray_frame))
        video.release()
        cv2.destroyAllWindows()
        return image_list
    
    def gen_track_avi(self, trajectories, frame_test, frame_region, image_list):
        track_img_list = []
        
        # Define crop bounds
        x_start, x_end = frame_region[0]
        y_start, y_end = frame_region[1]
        
        with tqdm(total=frame_test[1] - frame_test[0], desc="Gen video") as pbar:
            for frame_idx in range(frame_test[0], frame_test[1]):
                fig, ax = plt.subplots(figsize=(10, 10))

                # Load and crop the image
                img = np.copy(image_list[frame_idx])[y_start:y_end, x_start:x_end]

                # Load particle positions from CSV
                pos_df = pd.read_csv(self.particle_csv_path)
                filtered_df = pos_df[pos_df['frame'] == frame_idx]
                selected_columns = filtered_df[['centroid-0', 'centroid-1']].to_numpy()

                # Plot particle centroids, adjusted for crop region
                for idx in range(selected_columns.shape[0]):
                    x_coord = selected_columns[idx, 0] #* 1314
                    y_coord = selected_columns[idx, 1] #* 1054
                    
                    # Only plot if the point is within the cropped bounds
                    if x_start <= x_coord <= x_end and y_start <= y_coord <= y_end:
                        # Adjust the coordinates relative to the cropped region
                        adjusted_x = x_coord - x_start
                        adjusted_y = y_coord - y_start
                        ax.scatter(adjusted_x, adjusted_y, marker='o', s=100, edgecolor='red', facecolor='none', linewidth=0.8, alpha=0.8)
                
                # Plot trajectories, adjusting for crop region
                for frames, coordinates in trajectories:
                    if np.where(frames == frame_idx)[0].size > 0:  ### on time
                        mask = (frame_test[0] <= frames.flatten()) & (frames.flatten() <= frame_idx)   ### 
                        coordinates_idx = coordinates[mask]
                        
                        # Adjust trajectory coordinates for the cropped region
                        adjusted_coords = []
                        for coord in coordinates_idx:
                            x, y = coord[0], coord[1] 
                            if x_start <= x <= x_end and y_start <= y <= y_end:
                                adjusted_coords.append([x - x_start, y - y_start])
                        
                        if len(adjusted_coords) > 1:
                            adjusted_coords = np.array(adjusted_coords)
                            ax.plot(adjusted_coords[:, 0], adjusted_coords[:, 1], linewidth=1, color="blue")
                            #ax.scatter(adjusted_coords[:, 0], adjusted_coords[:, 1], marker='*', s=5, edgecolor='blue', facecolor='none', linewidth=0.4, alpha=1)

                ax.imshow(img, cmap="gray")
                # Finalize the plot and save the image to the list
                fig.canvas.draw()
                img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                img = img[:, :, :3]  # Drop the alpha channel
                track_img_list.append(img)
                plt.close(fig)
                
                pbar.update(1)
        
        return track_img_list