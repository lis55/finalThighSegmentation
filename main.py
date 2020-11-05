from data import *
from model import *
from generators import *
import os
import matplotlib.pyplot as plt
from vtk.util import numpy_support
import SimpleITK as sitk

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""

data_path ={"train":['G:/Datasets/elderlymen1/2d/train_frames', 'G:/Datasets/elderlymen1/2d/train_masks'],
            "aug":['G:/Datasets/elderlymen1/2d/val_frames', 'G:/Datasets/elderlymen1/2d/val_masks']}

test_paths = {1:['G:/Datasets/elderlymen1/2d/test_frames','G:/Datasets/elderlymen1/2d/test_masks'],
              2:['G:/Datasets/elderlymen2/2d/images','G:/Datasets/elderlymen2/2d/FASCIA_FINAL'],
              3:['G:/Datasets/youngmen/2d/images','G:/Datasets/youngmen/2d/FASCIA_FINAL'],
              4:['G:/Datasets/elderlywomen/2d/images','G:/Datasets/elderlywomen/2d/FASCIA_FINAL']}
save_paths ={1:[],
            2:[],
            3:[],
            4:[]}

save_paths = [['C:/final_results/elderlymen1/3d', 'G:/Datasets/elderlymen1/3ddownsampled/test/images',
               'G:/Datasets/elderlymen1/2d/images','C:/final_results/elderlymen1/3doverlaydown', 'C:/final_results/elderlymen2/3doverlay'],
              ['C:/final_results/elderlymen2/3d', 'G:/Datasets/elderlymen2/3ddownsampled/image',
               'G:/Datasets/elderlymen2/2d/images','C:/final_results/elderlymen2/3doverlaydown', 'C:/final_results/elderlymen2/3doverlay'],
              ['C:/final_results/youngmen/3d', 'G:/Datasets/youngmen/3ddownsampled/image',
               'G:/Datasets/youngmen/2d/images','C:/final_results/youngmen/3doverlaydown', 'C:/final_results/youngmen/3doverlay'],
              ['C:/final_results/elderlywomen/3d', 'G:/Datasets/elderlywomen/3ddownsampled/image',
               'G:/Datasets/elderlywomen/2d/images', 'C:/final_results/elderlywomen/3doverlaydown',
               'C:/final_results/elderlywomen/3doverlay']]


list_IDs = os.listdir(data_path[1][0])

gen = DataGenerator(list_IDs, data_path[1][0],data_path[1][1], to_fit=True, batch_size=2, dim=(512, 512), n_channels=1, n_classes=2, shuffle=True)

gen.plotFromGenerator()