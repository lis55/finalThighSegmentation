from model import *
from generators import *


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""

model = unet(pretrained_weights='final/unet_ThighOuterSurface.hdf5')
model.compile(optimizer=tf.keras.optimizers.Adam(lr=3e-5), loss=combo_loss(alpha=0.2, beta=0.4),
              metrics=[dice_accuracy])
sample = open('metrics2.txt', '+r')

i=1

list_IDs = os.listdir(test_paths[i][0])
testgen = DataGenerator(list_IDs, test_paths[i][0], test_paths[i][1], to_fit=True, batch_size=1, dim=(512, 512),
                        dimy=(512, 512), n_channels=1, n_classes=2, shuffle=False)

results = model.predict(testgen, len(list_IDs), verbose=1)
saveResult(save_path2d[i][0], results, test_frames_path=test_paths[i][0], overlay=True,
           overlay_path=save_path2d[i][1])
loss, acc = model.evaluate_generator(testgen, len(testgen), verbose=0)
print(loss, acc)
calculate_statistics(save_path2d[i][0], test_paths[i][1], sample)
print('Test loss: {} Test accuracy {}'.format(loss, acc), file=sample)
#export_as_mhd(save_path2d[i][0], save_path2d[i][3])


for i in (3,4,5):

    list_IDs = os.listdir(test_paths[i][0])
    testgen = DataGenerator(list_IDs, test_paths[i][0],test_paths[i][1], to_fit=True, batch_size=1, dim=(512, 512),dimy=(512, 512), n_channels=1, n_classes=2, shuffle=False)

    results = model.predict(testgen, len(list_IDs), verbose=1)
    saveResult(save_path2d[i-1][0], results, test_frames_path=test_paths[i][0],overlay=True,overlay_path=save_path2d[i-1][1])
    loss, acc = model.evaluate_generator(testgen, len(testgen), verbose=0)
    print(loss,acc)
    calculate_statistics(save_path2d[i-1][0], test_paths[i][1],sample)
    print('Test loss: {} Test accuracy {}'.format(loss, acc),file=sample)
    export_as_mhd(save_path2d[i-1][0], save_path2d[i-1][3])
sample.close()


sample = open('metrics.txt', '+r')

model = unet3d(pretrained_weights='unet_ThighOuterSurface.hdf5',input_size=(256, 256, 4,1))

test_paths = {1:['C:/Datasets/elderlymen1/2d/test_frames','C:/Datasets/elderlymen1/2d/test_masks'],
              2:['C:/Datasets/elderlymen1/3d/test/images','C:/Datasets/elderlymen1/3d/test/masks'],
              3:['C:/Datasets/elderlymen2/2d/images','C:/Datasets/elderlymen2/2d/FASCIA_FINAL'],
              4:['C:/Datasets/youngmen/2d/images','C:/Datasets/youngmen/2d/FASCIA_FINAL'],
              5:['C:/Datasets/elderlywomen/2d/images','C:/Datasets/elderlywomen/2d/FASCIA_FINAL'],
              6:['C:/Datasets/elderlymen1/3dmedium/test/images','C:/Datasets/elderlymen1/3dmedium/test/masks'],
              7:['C:/Datasets/elderlymen2/3dmedium/image','C:/Datasets/elderlymen2/3dmedium/mask'],
              8:['C:/Datasets/youngmen/3dmedium/image','C:/Datasets/youngmen/3dmedium/mask'],
              9:['C:/Datasets/elderlywomen/3dmedium/image','C:/Datasets/elderlywomen/3dmedium/mask']}

i=6
j=i-5
list_IDs = os.listdir(test_paths[i][0])
testgen3d = generator3d(list_IDs, test_paths[i][0], test_paths[i][1], to_fit=True, batch_size=1,
                        patch_size=4, dim=(256, 256), dimy=(256, 256), n_channels=1, n_classes=2, shuffle=False)
results = model.predict_generator(testgen3d, len(testgen3d), verbose=1)
saveResult3d(results, patch_size=4,save_path=save_paths3d[j][0],test_frames_path=save_paths3d[j][1],framepath2=save_paths3d[j][2],overlay_path=save_paths3d[j][3],overlay_path2=save_paths3d[j][4])
calculate_statistics(save_paths3d[j][0], "C:/Datasets/elderlymen1/2d/FASCIA_FILLED", sample, size = (512,512))
loss, acc = model.evaluate_generator(testgen3d, len(testgen3d), verbose=0)
print(loss,acc)
print('Test loss: {} Test accuracy {}'.format(loss, acc),file=sample)


for i in (7,8,9):
    j=i-5
    list_IDs = os.listdir(test_paths[i][0])
    testgen3d = generator3d(list_IDs, test_paths[i][0], test_paths[i][1], to_fit=True, batch_size=1,
                            patch_size=4, dim=(256, 256), dimy=(256, 256), n_channels=1, n_classes=2, shuffle=False)
    results = model.predict_generator(testgen3d, len(testgen3d), verbose=1)
    saveResult3d(results, patch_size=4,save_path=save_paths3d[j][0],test_frames_path=save_paths3d[j][1],framepath2=save_paths3d[j][2],overlay_path=save_paths3d[j][3],overlay_path2=save_paths3d[j][4])
    calculate_statistics(save_paths3d[j][0], test_paths[j+1][1], sample, size = (512,512))
    loss, acc = model.evaluate_generator(testgen3d, len(testgen3d), verbose=0)
    print(loss,acc)
    print('Test loss: {} Test accuracy {}'.format(loss, acc),file=sample)

sample.close()


