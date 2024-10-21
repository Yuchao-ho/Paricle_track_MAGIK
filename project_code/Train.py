from MAGIK_model import Classifier_model
from deeplay import BinaryClassifier, Adam, Trainer
import os
import torch
from Build_graph import Graph_Generator
from Dataset import Tracing_Dataset, RandomRotation, RandomFlip
from torchvision import transforms
from torch_geometric.loader import DataLoader


class tracker_train:
  def __init__(
      self, connectivity_radius=None, num_particle_sim=None, 
      len_frame_sim=None, num_frame_sim=None, 
      D_sim=None, max_gap=None, prob_noise=None
      ):
    self.connectivity_radius = connectivity_radius
    self.num_particle_sim = num_particle_sim
    self.len_frame_sim = len_frame_sim
    self.num_frame_sim = num_frame_sim
    self.D_sim = D_sim
    self.max_gap = max_gap
    self.prob_noise = prob_noise
    
  def __call__(
      self, window_size, data_size, batch_size, 
      max_epochs, particle_feature_path, checkpoint_pth) :

    ## Generate train graph
    graph_Generator = Graph_Generator(
        connectivity_radius=self.connectivity_radius, num_particle_sim= self.num_particle_sim, 
        len_frame_sim= self.len_frame_sim, num_frame_sim= self.num_frame_sim, 
        D_sim= self.D_sim, max_gap= self.max_gap, prob_noise= self.prob_noise
    )
    train_graph = graph_Generator(particle_feature_path)

    ## build data loader
    train_dataset = Tracing_Dataset(
      train_graph,
      window_size= window_size,    ## could influence capacity for linking
      dataset_size= data_size, 
      transform=transforms.Compose([RandomRotation(), RandomFlip()]),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    ## train the model
    new_model = Classifier_model()
    classifier = BinaryClassifier(model=new_model, optimizer=Adam(lr=1e-3))
    classifier = classifier.create()
    trainer = Trainer(max_epochs=max_epochs)
    trainer.fit(classifier, train_loader)
    if not os.path.exists(checkpoint_pth):
      torch.save(classifier.model.state_dict(), checkpoint_pth)


if __name__ == "__main__":
    trainer = tracker_train(
        connectivity_radius=0.02,num_particle_sim= 100, 
        len_frame_sim= 500, num_frame_sim= 60, 
        D_sim= 0.1, max_gap= 4, prob_noise= 0.05
    )

    trainer(
        window_size= 6, data_size= 624, batch_size=30, max_epochs= 30, 
        particle_feature_path= "/home/user/Project_thesis/Particle_Hana/Cell7__ground_truth/particle_fea(weighted).csv", 
        checkpoint_pth= "/home/user/Project_thesis/Particle_Hana/Cell7__ground_truth/model_(Consec(weighted), num=100, gap=4).pt"
    )