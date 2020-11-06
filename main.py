from data import *
from model import *
from generators import *
import os
import matplotlib.pyplot as plt
from vtk.util import numpy_support
import SimpleITK as sitk

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""

data_gen_args_dict = dict(shear_range=20,
                    rotation_range=20,
                    horizontal_flip=True,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    fill_mode='nearest')
data_path ={"train":['G:/Datasets/elderlymen1/2d/train_frames', 'G:/Datasets/elderlymen1/2d/train_masks'],
            "val":['G:/Datasets/elderlymen1/2d/val_frames', 'G:/Datasets/elderlymen1/2d/val_masks'],
            "train3d": ['G:/Datasets/elderlymen1/3ddownsampled/train/images', 'G:/Datasets/elderlymen1/3ddownsampled/train/masks'],
            "val3d": ['G:/Datasets/elderlymen1/3ddownsampled/val/images', 'G:/Datasets/elderlymen1/3ddownsampled/val/masks']
            }

test_paths = {1:['G:/Datasets/elderlymen1/2d/test_frames','G:/Datasets/elderlymen1/2d/test_masks'],
              2:['G:/Datasets/elderlymen2/2d/images','G:/Datasets/elderlymen2/2d/FASCIA_FINAL'],
              3:['G:/Datasets/youngmen/2d/images','G:/Datasets/youngmen/2d/FASCIA_FINAL'],
              4:['G:/Datasets/elderlywomen/2d/images','G:/Datasets/elderlywomen/2d/FASCIA_FINAL']}

save_paths ={1:['C:/final_results/elderlymen1/3d', 'G:/Datasets/elderlymen1/3ddownsampled/test/images',
               'G:/Datasets/elderlymen1/2d/images','C:/final_results/elderlymen1/3doverlaydown', 'C:/final_results/elderlymen2/3doverlay'],
            2:['C:/final_results/elderlymen2/3d', 'G:/Datasets/elderlymen2/3ddownsampled/image',
               'G:/Datasets/elderlymen2/2d/images','C:/final_results/elderlymen2/3doverlaydown', 'C:/final_results/elderlymen2/3doverlay'],
            3:['C:/final_results/youngmen/3d', 'G:/Datasets/youngmen/3ddownsampled/image',
               'G:/Datasets/youngmen/2d/images','C:/final_results/youngmen/3doverlaydown', 'C:/final_results/youngmen/3doverlay'],
            4:['C:/final_results/elderlywomen/3d', 'G:/Datasets/elderlywomen/3ddownsampled/image',
               'G:/Datasets/elderlywomen/2d/images', 'C:/final_results/elderlywomen/3doverlaydown',
               'C:/final_results/elderlywomen/3doverlay']}


list_IDs = os.listdir(data_path["train"][0])
gen = DataGenerator(list_IDs, data_path["train"][0],data_path["train"][1], to_fit=True, batch_size=2, dim=(512, 512),dimy=(512, 512), n_channels=1, n_classes=2, shuffle=True)
genaug =DataGenerator2(list_IDs, data_path["train"][0],data_path["train"][1], to_fit=True, batch_size=2, dim=(512, 512),dimy=(512, 512), n_channels=1, n_classes=2, shuffle=True, data_gen_args =data_gen_args_dict)

list_IDs = os.listdir(data_path["train3d"][0])
gen3d = generator3d(list_IDs, data_path["train3d"][0],data_path["train3d"][1], to_fit=True, batch_size=1, patch_size=8, dim=(128, 128), dimy=(128,128), n_channels=1, n_classes=2, shuffle=True)
gen3da = generator3da(list_IDs, data_path["train3d"][0],data_path["train3d"][1], to_fit=True, batch_size=1, patch_size=8, dim=(128, 128), dimy=(128,128), n_channels=1, n_classes=2, shuffle=True, data_gen_args=data_gen_args_dict)

gen3d.plotFromGenerator3d()