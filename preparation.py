from data import *
from metrics import *
from model import *
import shutil

def reorderframes(d2_frame_path,d2_mask_path,d3_frame_path,d3_mask_path,slices,patients):
    all_frames = os.listdir(d2_frame_path)
    all_masks = os.listdir(d2_mask_path)
    for i in range(1, patients+1):
        if not os.path.isdir(d3_frame_path + '/' + str(i)):
            os.makedirs(d3_frame_path + '/' + str(i))
    for i in range(1, patients+1):
        if not os.path.isdir(d3_mask_path + '/' + str(i)):
            os.makedirs(d3_mask_path + '/' + str(i))
    count = 1
    for i in range(0, len(all_frames), slices):
        print(i)
        for j in range(i, i + slices):
            shutil.move(d2_frame_path + '/' + all_frames[j], d3_frame_path + '/' + str(count) + '/' + all_frames[j])
            shutil.move(d2_mask_path + '/' + all_masks[j], d3_mask_path + '/' + str(count) + '/' + all_masks[j])
        count += 1

#predict for all the datasets
paths1 = [save_paths[2][2],save_paths[3][2],save_paths[4][2]]## todo
paths2 = [test_paths[3][1],test_paths[4][1],test_paths[5][1]]## todo
for path, path2 in zip(paths1,paths2):
    print()
####### print statictics ############
#comparison 2d model
paths1 = [save_paths[2][2],save_paths[3][2],save_paths[4][2]]
paths2 = [test_paths[3][1],test_paths[4][1],test_paths[5][1]]
for path, path2 in zip(paths1,paths2):
    calculate_statistics(path, path2)
#comparison 3d model
paths1 = [save_paths[2][0],save_paths[3][0],save_paths[4][0]]
paths2 = [test_paths[3][1],test_paths[4][1],test_paths[5][1]]
for path, path2 in zip(paths1,paths2):
    calculate_statistics(path, path2)