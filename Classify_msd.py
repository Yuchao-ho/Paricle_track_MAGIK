from Pipeline.Combine_graph import process_traj
from Utils.Cal_msd import Cal_msd
from Utils.Visualize import visualization
from Utils.Classify import Classify
import matplotlib.pyplot as plt
import numpy as np

def main(video_pth, particle_csv_pth, checkpt_pth):
    ## get trajectories
    gen_traj = process_traj(
        video_pth = video_pth,
        len_sub = 30,
        len_overlap = 5,
        prob_thre = 0.5,
        particle_csv_pth= particle_csv_pth,
    )
    trajectories = gen_traj(
        len_thre = 10,
        checkpt_pth = checkpt_pth,
        connect_radius = 35
    )

    #visualization.traj_distrib(trajectories)
    """ filter_traj = []
    for frame, coord in trajectories:
        if len(frame) < 20  and len(frame) > 10:
            filter_traj.append((frame, coord)) """
    #print(f"num of traj > 10 points: {len(trajectories) - len(filter_traj)}")

    ## get msd
    """ cal_msd = Cal_msd(trajectories= trajectories)
    all_msd = cal_msd() """
    #visualization.plot_msd(all_msd=all_msd)
    
    ## get video
    visualization.traj_distrib(trajectories)
    """ recorder = visualization(
        video_path= video_pth, 
        particle_csv_path= particle_csv_pth, 
        frame_region= [(0, 1314), (0, 1054)], 
        frame_test= (300, 600), 
        )
    recorder.gen_video(
        trajectories= trajectories, 
        fps= 1, 
        output_path= "Video/traj_(len_thre > 10)(direct)(time).avi"
        ) """

    ## classify
    """ classifier = Classify(all_msd= all_msd)
    classifier.slope_classify() """
    """ labels, kmeans, normalized_features = classifier.kmean_arrays(all_msd, n_clusters= None)
    visualization.plot_clusters(all_msd, labels, kmeans) """
    #return labels, kmeans, normalized_features


if __name__ == "__main__":

    main(
        video_pth = "Video/01_18_Cell7_PC3_cropped3_1_1000ms.avi", 
        particle_csv_pth = "Cell7__ground_truth/particle_fea(mean_intens)(orient).csv", 
        checkpt_pth = "Cell7__ground_truth/model_(w_size=20, motion)(direct)(1).pt"
        )