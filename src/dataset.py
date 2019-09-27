import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from torch.utils.data import Dataset, DataLoader
from .utils import get_file_paths, get_user_dict

class TripletDataset():
    def __init__(self, metadata, user_dict, track_sample_nb = 4, data_width = 128, data_height = 370, 
                 ignore_color_channel=True, transformer=None, use_cuda = False):
        self.metadata = metadata
        self.user_dict = user_dict
        self.user_list = list(user_dict.keys())
        self.track_sample_nb = track_sample_nb
        self.data_width = data_width
        self.data_height = data_height
        self.ignore_color_channel = ignore_color_channel
        self.use_cuda = use_cuda
        
    def __len__(self):
        return len(self.user_list)
    
    def load_img(self, fpath):
        if self.ignore_color_channel:
            return torch.tensor((io.imread(fpath)[:,:,0])/255.0)
        else:
            return torch.tensor((io.imread(fpath))/255.0)
    
    def __getitem__(self, user_index):

        if isinstance(user_index, int):
            user = self.user_list[user_index]
            instances_idx = np.random.choice(self.user_dict[user], size = self.track_sample_nb, 
                                        replace = False)
            
            result_array = torch.empty((self.track_sample_nb, self.data_width, self.data_height))
            
            if self.use_cuda:
                result_array = result_array.cuda()

            for i, f in enumerate(instances_idx):
                result_array[i] = self.load_img(f)
            
            return result_array
        
        if isinstance(user_index, tuple):
            users_index, user_index = user_index, None
            result_array = torch.empty((len(users_index), self.track_sample_nb, self.data_width,
            self.data_height))

            if self.use_cuda:
                result_array = result_array.cuda()

            for index_number, user_index in enumerate(users_index):
                user = self.user_list[user_index]
                instances_idx = np.random.choice(self.user_dict[user], size = self.track_sample_nb, 
                                            replace = False)
                                
                for i, f in enumerate(instances_idx):
                    result_array[index_number, i] = self.load_img(f)
            
            return result_array
