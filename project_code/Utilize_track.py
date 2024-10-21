import cv2
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from Test_com_2 import process_traj

class utilize_track:

    def __init__ (
                self, video_path=None, particle_csv_path=None, position_csv=None,
                len_sub=None, len_overlap=None, prob_thre=None, len_thre=None,
                trajectories=None, checkpt_path=None
                ):
        self.video_path = video_path
        self.particle_csv_path = particle_csv_path
        self.position_csv = position_csv
        self.len_sub = len_sub
        self.len_overlap = len_overlap
        self.prob_thre = prob_thre
        self.len_thre = len_thre
        self.trajectories = trajectories
        self.checkpt_path = checkpt_path

    def __call__(
            self, mode, frame_gap, dist_gap, feature_gap, 
            fps=None, frame_region=None, 
            frame_test=None, output_path=None
            ):  
        if self.trajectories is None:
            ## generate traj
            combinator_traj = process_traj(
                particle_csv_pth= self.particle_csv_path,
                len_sub= self.len_sub,
                len_overlap= self.len_overlap,
                prob_thre= self.prob_thre
            )
            self.trajectories = combinator_traj(
                checkpt_pth= self.checkpt_path,
                len_thre= 1,
                frame_gap= frame_gap,
                dist_gap= dist_gap,
                feature_gap= feature_gap
            )

        if mode == "gen_video":
        ## import frame list
            image_list = self.load_frame(self.video_path)
        ## generate .avi video           
            track_img_list = self.gen_track_avi(frame_test, frame_region, image_list)
            height, width, _ = track_img_list[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            for img in track_img_list:
                video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            video.release()

        elif mode == "traj_msd":
            pass  ## TODO 
        
        elif mode == "gen_distrib":
            frame_diff = []
            for frames, coordinates in self.trajectories:
                if (frames[-1]-frames[0]) >= 3:
                    frame_diff.append(frames[-1]-frames[0])
                
            hist, bins = np.histogram(frame_diff, bins=10)  # Adjust number of bins as needed
            plt.hist(frame_diff, bins=bins, alpha=0.5)
            plt.xlabel("Len of Tracking")
            plt.ylabel("Frequency")
            plt.title("Len of Tracking (Overlap=1)")
            plt.show()

                        

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
    
    def gen_track_avi(self, frame_test, frame_region, image_list):
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
                    x_coord = selected_columns[idx, 0] * 1314
                    y_coord = selected_columns[idx, 1] * 1054
                    
                    # Only plot if the point is within the cropped bounds
                    if x_start <= x_coord <= x_end and y_start <= y_coord <= y_end:
                        # Adjust the coordinates relative to the cropped region
                        adjusted_x = x_coord - x_start
                        adjusted_y = y_coord - y_start
                        ax.scatter(adjusted_x, adjusted_y, marker='o', s=100, edgecolor='red', facecolor='none', linewidth=0.8, alpha=0.8)
                
                # Plot trajectories, adjusting for crop region
                for frames, coordinates in self.trajectories:
                    if np.where(frames == frame_idx)[0].size > 0:  ### on time
                        mask = frames.flatten() <= frame_idx   ### 
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
    

if __name__ == "__main__":
    gen_video = utilize_track(
        video_path = "/home/user/Project_thesis/Particle_Hana/Video/01_18_Cell7_PC3_cropped3_1_1000ms.avi",
        len_sub = 60,
        len_overlap = 5,  ## 5(most often)
        prob_thre = 0.4,
        len_thre = 1,
        checkpt_path = "/home/user/Project_thesis/Particle_Hana/Cell7__ground_truth/model_(Consec(dist), num=100, gap=4).pt",
        particle_csv_path= "/home/user/Project_thesis/Particle_Hana/Cell7__ground_truth/particle_fea(std_intens).csv",
    )
    
    """ gen_video(
        mode= "gen_video", 
        frame_gap= 10,
        dist_gap= 50.0,
        feature_gap= 0.7,
        fps= 1, 
        frame_region= [(200, 500), (700, 900)],
        frame_test=(200, 400), 
        output_path="/home/user/Project_thesis/Particle_Hana/Video/traj_(Consec(steady), num=100, gap=4).avi"
    )  """

    gen_video(
        mode= "gen_video", 
        frame_gap= 10,
        dist_gap= 50.0,
        feature_gap= 0.7,
        fps= 1, 
        frame_region= [(600, 800), (200, 500)],
        frame_test=(200, 400), 
        output_path="/home/user/Project_thesis/Particle_Hana/Video/traj_(Consec(dist), num=100, gap=4)(real_20).avi"
    ) 