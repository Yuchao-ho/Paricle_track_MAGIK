import cv2
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from project_code.Combine_graph import process_traj

class visualize_track:

    def __init__ (
              self, video_pth=None, particle_csv_pth=None, position_csv=None, 
              trajectories=None, checkpt_pth=None
              ):
        self.video_path = video_pth
        self.particle_csv_pth = particle_csv_pth
        self.position_csv = position_csv
        self.trajectories = trajectories
        self.checkpt_pth = checkpt_pth

    def __call__(self, mode, output_path=None):  
        if self.trajectories is None:
            ## generate traj
            combinator_traj = process_traj(
                video_pth= self.video_path,
                particle_csv_pth= self.particle_csv_pth,
                position_csv= self.position_csv,
            )
            self.trajectories = combinator_traj(
                checkpt_pth= self.checkpt_pth
            )

        if mode == "gen_video":
        ## import frame list
            image_list = self.load_frame(self.video_path)
        ## generate .avi video (Example)
            frame_test = (0, len(image_list)-1)
            
            track_img_list = self.gen_track_avi(frame_test, self.trajectories, self.combined_graph)
            height, width, _ = track_img_list[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            video = cv2.VideoWriter(output_path, fourcc, 5, (width, height))
            for img in track_img_list:
                video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            video.release()

        elif mode == "check_traj":
            
        



    def load_frame(self, video_path):
        image_list = []
        video = cv2.VideoCapture(video_path)
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
    
    def gen_track_avi(frame_test, image_list, trajectories, combined_graph):
        track_img_list = []
        with tqdm(total=frame_test[1]-frame_test[0], desc="Generate Track Video") as pbar:
            for frame_idx in range(frame_test[0], frame_test[1]):
                fig, ax = plt.subplots(figsize=(10, 10))
                img = np.copy(image_list[frame_idx])
                pos_df = pd.read_csv("/content/drive/MyDrive/Particle Tracking/Cell7__ground_truth/particle_feature.csv")
                filtered_df = pos_df[(pos_df['frame'] == frame_idx)]
                selected_columns = filtered_df[['centroid-0', 'centroid-1']].to_numpy()
                for idx in range(selected_columns.shape[0]):
                    ax.scatter(selected_columns[idx, 0]*1314, selected_columns[idx, 1]*1054, marker='o', s=100, edgecolor='red', facecolor='none', linewidth=0.8, alpha=0.8)
                ax.imshow(img, cmap="gray")
                for traj in trajectories:
                    frames = combined_graph.frames[list(traj)]
                    traj_tensor = torch.tensor(list(traj))
                    sorted_frames, sorted_idx = torch.sort(frames)
                    sorted_traj = traj_tensor[sorted_idx]
                    indices = torch.where(sorted_frames <= frame_idx)[0]
                    traj_frame = sorted_traj[indices]

                    if len(traj_frame) == 0:
                        continue
                    else:
                        coordinates = combined_graph.x[traj_frame]
                        corrdinate_x = coordinates[:, 0]*1314
                        corrdinate_y = coordinates[:, 1]*1054
                        ax.plot(corrdinate_x, corrdinate_y, linewidth=1, color="blue")
                    fig.canvas.draw()
                img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                track_img_list.append(img)

                ax.clear()
                plt.close(fig)
                pbar.update(1)
        return track_img_list
    

    