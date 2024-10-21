import deeplay as dl
import deeptrack as dt
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
import os
from torch.utils.data import Dataset

class gen_detect:
    
    def __call__(
            self, len_dataset= None, batch_size= None, max_epochs= None,
            video_pth= None, checkpth= None, detect_pth= None
            ):
               
        ## get gray image list:
        image_list = self.catch_frame(video_pth)
        if not os.path.exists(checkpth):
            self.gen_chkpt(image_list, len_dataset, batch_size, max_epochs, checkpth)
        
        ## Generate detections
        if not os.path.exists(detect_pth):
            self.gen_detection_df(image_list, checkpth, detect_pth)
   
    def gen_chkpt(self, image_list, len_dataset, batch_size, max_epochs, checkpth):
        regions = self.crop_particle(image_list)
        transform = transforms.Compose([
            dt.Multiply(lambda: np.random.uniform(0.9, 1.1)),
            dt.Add(lambda: np.random.uniform(-0.1, 0.1)),
            dt.MoveAxis(-1, 0),
            transforms.ToTensor()
        ])
        train_dataset = ImageDataset(regions, len_dataset= len_dataset, transform=transform)
        dataloader = dl.DataLoader(train_dataset, batch_size= batch_size, shuffle=True)

        #train the lodestar
        lodestar = dl.LodeSTAR(n_transforms=4, optimizer=dl.Adam(lr=1e-4)).build()
        trainer = dl.Trainer(max_epochs=max_epochs)
        trainer.fit(lodestar, dataloader)
        torch.save(lodestar.model.state_dict(), checkpth)

    def catch_frame(self, video_pth):
        image_list = []
        video = cv2.VideoCapture(video_pth)
        while True:
            ret, frame = video.read()
            if not ret:
                break
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)/256
            image_list.append(np.array(gray_frame))
            # if len(image_list) > 20:
            #     break

        video.release()
        cv2.destroyAllWindows()
        return image_list

    def crop_particle(self, image_list):
        region = []
        image_0 = image_list[0]
        region.append(image_0[470:510, 110:150])
        region.append(image_0[700:740, 30:70])
        region.append(image_0[300:340, 620:660])
        region.append(image_0[730:770, 515:555])
        region.append(image_0[100:140, 1020:1060])
        region.append(image_0[265:305, 35:75])
        return region
    
    @torch.no_grad
    def gen_detection_df(self, image_list, checkpt_pth, df_pth, batch_size=8):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        lodestar = dl.LodeSTAR(n_transforms=4, optimizer=dl.Adam(lr=1e-4)).build()
        lodestar.model.load_state_dict(torch.load(checkpt_pth))
        lodestar.to(device)
        lodestar.eval()

        detection_df = pd.DataFrame(columns=["frame", "centroid-0", "centroid-1"])

        num_images = len(image_list)
        num_batches = (num_images + batch_size - 1) // batch_size
        with tqdm(total=num_batches) as pbar:
            for batch_start in range(0, num_images, batch_size):
                batch_images = image_list[batch_start:batch_start + batch_size]
                # Convert batch to tensor and move to GPU
                batch_tensors = [torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(device) for img in batch_images]
                torch_batch = torch.cat(batch_tensors, dim=0)
                detections_batch = lodestar.detect(torch_batch, alpha=0.25, beta=0.75, mode="constant", cutoff=0.3)

                # Process detections and append to dataframe
                for i, detections in enumerate(detections_batch):
                    img_idx = batch_start + i
                    for j in range(detections.shape[0]):
                        detection_df.loc[len(detection_df.index)] = [int(img_idx), detections[j, 1], detections[j, 0]]
                del batch_tensors
                torch.cuda.empty_cache()
                pbar.update(1)
                
        detection_df.to_csv(df_pth, index=False)

class ImageDataset(Dataset):    
    def __init__(self, image_list, len_dataset:int, transform=None):
        self.image_list = image_list
        self.len_dataset = len_dataset
        self.transform = transform

    def __len__(self):
        return (len(self.image_list) * self.len_dataset)

    def __getitem__(self, idx):
        idx = idx % len(self.image_list)
        image = self.image_list[idx]
        if self.transform:
            image = self.transform(image)
        return image.float()
    

if __name__ == "__main__":
    gen_detection = gen_detect()
    gen_detection(
        len_dataset= 500, batch_size= 20, max_epochs= 30,
        video_pth= "/home/user/Project_thesis/Particle_Hana/Video/01_18_Cell7_PC3_cropped3_1_1000ms.avi", 
        checkpth= "/home/user/Project_thesis/Particle_Hana/Cell7__ground_truth/lodestar_larger.pt", 
        detect_pth= "/home/user/Project_thesis/Particle_Hana/Cell7__ground_truth/lodestar_detection(larger).csv"
    )
    