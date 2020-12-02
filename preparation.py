from data import *
from metrics import *
from model import *
import shutil
from generators import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""



sample = open('metrics.txt', '+r')
model = unet(pretrained_weights='unet_ThighOuterSurface.hdf5')

for i in (3,4,5):
    list_IDs = os.listdir(test_paths[i][0])
    testgen = DataGenerator(list_IDs, test_paths[i][0],test_paths[i][1], to_fit=True, batch_size=1, dim=(512, 512),dimy=(512, 512), n_channels=1, n_classes=2, shuffle=False)

    results = model.predict_generator(testgen, len(list_IDs), verbose=1)
    saveResult(save_path2d[i-1][0], results, test_frames_path=test_paths[i][0],overlay=True,overlay_path=save_path2d[i-1][1])
    loss, acc = model.evaluate_generator(testgen, len(testgen), verbose=0)
    print(loss,acc)
    calculate_statistics(save_path2d[i-1][0], test_paths[i][1],sample)
    print('Test loss: {} Test accuracy {}'.format(loss, acc),file=sample)
sample.close()
