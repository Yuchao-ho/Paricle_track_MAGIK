import cv2
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from scipy.spatial import cKDTree
import random
import tqdm

class gen_fil_csv:

    def __init__(self, video_pth, position_pth, side_len):
        self.video_path = video_pth
        self.position_path = position_pth
        self.side_length = side_len

    def __call__(self, tolerence, feature_list, output_pth):
        video = cv2.VideoCapture(self.video_path)
        image_list = []
        frame_idx = 0 
        column_name = ["frame", "centroid-0", "centroid-1"] + feature_list

        with tqdm.tqdm(total=int(video.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Process Img") as pbar:
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                particle_data = self.Process_img(gray_frame, feature_list, frame_idx)
                ## remove too near particles
                particle_data = self.rm_duplicate(particle_data, tolerence)

                frame_idx += 1
                pbar.update(1)
                image_list.append(particle_data)   ###
            video.release()
            cv2.destroyAllWindows()

        particle_data = np.vstack(image_list)
        particle_df = pd.DataFrame(particle_data, columns=column_name)
        ## remove duplicated particles
        #particle_df = particle_df.drop_duplicates(subset=particle_df.columns[-1], keep='first')
        
        particle_df.to_csv(output_pth, index=False)

    def Process_img(self, frame, feature_list, frame_idx):
        position_df = pd.read_csv(self.position_path)
        pos_x = np.array(
            position_df[position_df["frame"] == frame_idx]["centroid-0"]
        ) 
        pos_y = np.array(
            position_df[position_df["frame"] == frame_idx]["centroid-1"]
        )
        pos_zip = list(zip(pos_x, pos_y))

        # create matrix to store data
        particle_data_frame = np.zeros((len(pos_zip), len(feature_list)+3))

        for idx, (x, y) in enumerate(pos_zip):
            bottom_left_x = x - self.side_length / 2
            bottom_left_y = y - self.side_length / 2
            top_right_x = bottom_left_x + self.side_length
            top_right_y = bottom_left_y + self.side_length

            region = frame[
                max(0,int(bottom_left_y)) : min(frame.shape[0],int(top_right_y)), max(0,int(bottom_left_x)) : min(frame.shape[1],int(top_right_x))
            ]
            # choose latter 15% value
            threshold_value = min(
                np.percentile(region.flatten(), 85), 254
            )
            _, thresholded_region = cv2.threshold(region, threshold_value, 255, cv2.THRESH_BINARY)

            labeled_image = label(thresholded_region)
            props = regionprops(labeled_image, intensity_image=region)
            max_area = max([prop.area for prop in props])
            max_area_regions = [prop for prop in props if prop.area == max_area][0]
            max_area_prop = self.Get_region_prop(max_area_regions, feature_list)
            #particle_data_frame[idx,:] = np.array([frame_idx, *(x/frame.shape[1],y/frame.shape[0])] + max_area_prop)
            particle_data_frame[idx,:] = np.array([frame_idx, *(x,y)] + max_area_prop)

        """ for col_idx in range(3, len(feature_list)+3):
            particle_data_frame[:, col_idx] /= np.abs(particle_data_frame[:, col_idx]).max() """
        
        return particle_data_frame 

    def Get_region_prop(self, region, feature_list):
        region_prop = []
        prop_dispatch = {
            "area": lambda prop: prop.area,
            "perimeter": lambda prop: prop.perimeter,
            "eccentricity": lambda prop: prop.eccentricity,
            "orientation": lambda prop: prop.orientation,
            "mean_intens": lambda prop: np.mean(prop.intensity_image),   ## mean_intensity  (bad)
            "std_intens": lambda prop: np.std(prop.intensity_image),     ## std_intensity
            "median_intens": lambda prop: np.median(prop.intensity_image)
        }
        for feature in feature_list:
            if feature in prop_dispatch:
                region_prop.append(prop_dispatch[feature](region))

        return region_prop

    def rm_duplicate(self, particle_data, tolerence):
        particle_pos = particle_data[:, 1:3] #(x,y)
        ## use cdk tree to find neighbors within tolerence
        tree = cKDTree(particle_pos)
        pos_pairs = tree.query_pairs(tolerence)
        ## choose row index of deleted particles
        del_row = self.pick_unique(pos_pairs)
        filtered_particle_pos = np.delete(particle_data, del_row, axis=0)

        return filtered_particle_pos

    def pick_unique(self, pairs):
        selected_elements = set()  
        result = []
        # randomize tuples
        pairs_list = list(pairs)
        random.shuffle(pairs_list)
        for pair in pairs_list:
            # pick unique ele
            element_1, element_2 = pair
            if element_1 not in selected_elements:
                selected_elements.add(element_1)
                result.append(element_1)
            elif element_2 not in selected_elements:
                selected_elements.add(element_2)
                result.append(element_2)

        return result

if __name__ == "__main__":
    gen_detection = gen_fil_csv(
        video_pth= "/home/user/Project_thesis/Particle_Hana/Video/01_18_Cell7_PC3_cropped3_1_1000ms.avi", 
        position_pth= "/home/user/Project_thesis/Particle_Hana/Cell7__ground_truth/lodestar_detection(new).csv", 
        side_len= 20  #(20 best)
        )
    gen_detection(
        tolerence= 10,  #(5 pixel best)
        feature_list= ["mean_intens"], 
        output_pth= "/home/user/Project_thesis/Particle_Hana/Cell7__ground_truth/particle_fea(mean_intens)(orient).csv"
    )
                



