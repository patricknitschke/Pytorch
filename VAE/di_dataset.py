import torch
from torch.utils.data import Dataset

import os
import numpy as np
import pickle
from tqdm import tqdm

batch_size = 4

# saves_folders = "../../../rl_data"
# load_paths = [os.path.join(saves_folders, saves_folder) for saves_folder in os.listdir(saves_folders)]

class DepthImageDataset(Dataset):
    def __init__(self, load_paths) -> None:
        super(DepthImageDataset, self).__init__()
        
        self.x = []
        self.nb_images = 0
        for load_path in load_paths: 
            nb_files = int(len([f for f in os.listdir(load_path) if f.endswith('.p') and os.path.isfile(os.path.join(load_path, f))]) / 5) # five dicts
            print(f"Loading from {load_path}")
            for i in tqdm(range(nb_files)):
                di_load           = pickle.load(open(load_path + "/di_dump" + str(i) + ".p", "rb"))
                # obs_load          = pickle.load(open(load_path + "/obs_dump" +str(i) + ".p", "rb"))
                # action_load       = pickle.load(open(load_path + "/action_dump" + str(i) + ".p", "rb"))
                # action_index_load = pickle.load(open(load_path + "/action_index_dump" + str(i) + ".p", "rb"))
                # collision_load    = pickle.load(open(load_path + "/collision_dump" + str(i) + ".p", "rb"))
         
            for ep in di_load.values():
                for image in ep:
                    image_flipped = np.flip(image, 1)
                    self.x.append(image.transpose(2,0,1))
                    self.x.append(image_flipped.transpose(2,0,1))
                    
                    self.nb_images += 2
        
        self.x = torch.Tensor(np.array(self.x))
    
    def __getitem__(self, index: int):
        return self.x[index]
    
    def __len__(self):
        return self.nb_images