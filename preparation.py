from data import *
from metrics import *
from model import *
import shutil


#predict for all the datasets
paths1 = [save_paths3d[2][2],save_paths3d[3][2],save_paths3d[4][2]]## todo
paths2 = [test_paths[3][1],test_paths[4][1],test_paths[5][1]]## todo
for path, path2 in zip(paths1,paths2):
    print()
####### print statictics ############
#comparison 2d model
paths1 = [save_paths3d[2][2],save_paths3d[3][2],save_paths3d[4][2]]
paths2 = [test_paths[3][1],test_paths[4][1],test_paths[5][1]]
for path, path2 in zip(paths1,paths2):
    calculate_statistics(path, path2)
#comparison 3d model
paths1 = [save_paths3d[2][0],save_paths3d[3][0],save_paths3d[4][0]]
paths2 = [test_paths[3][1],test_paths[4][1],test_paths[5][1]]
for path, path2 in zip(paths1,paths2):
    calculate_statistics(path, path2)