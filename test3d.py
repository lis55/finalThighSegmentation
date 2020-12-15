from data import *
from model import *
from generators import *
import os
import matplotlib.pyplot as plt
from vtk.util import numpy_support
import SimpleITK as sitk
from datetime import datetime
from datetime import date
import tensorflow
from tuning import *
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""

'''i=4
list_IDs = test_paths[i][0]
gen = DataGenerator(list_IDs, test_paths[i][0],test_paths[i][1], to_fit=True, batch_size=2, dim=(512, 512),dimy=(512, 512), n_channels=1, n_classes=2, shuffle=True)
gen.downsample((256,256),"C:/Datasets/youngmen/3dmedium")
reorderframes('C:/Datasets/youngmen/3dmedium/images','C:/Datasets/youngmen/3dmedium/masks',
              'C:/Datasets/youngmen/3dmedium/image','C:/Datasets/youngmen/3dmedium/mask',26,17)
'''


'''frame_path = "C:/smol/train_frames"
mask_path = "C:/smol/train_masks2"
val_frames ="C:/smol/val_frames"
val_masks = "C:/smol/val_masks2"
test_frames = "C:/fasciafilled/test_frames"
test_masks = "C:/fasciafilled/test_masks2"'''
######### 2D generators ###########
'''list_IDs = os.listdir(frame_path)
genaug =DataGenerator2(list_IDs, frame_path,mask_path, to_fit=True, batch_size=2, dim=(512, 512),dimy=(512, 512), n_channels=1, n_classes=2, shuffle=True, data_gen_args =data_gen_args_dict)
list_IDs = os.listdir(val_frames)
genval =DataGenerator(list_IDs, val_frames,val_masks, to_fit=True, batch_size=2, dim=(512, 512),dimy=(512, 512), n_channels=1, n_classes=2, shuffle=True)
list_IDs = os.listdir(test_frames)
testgen = DataGenerator(list_IDs, test_frames,test_masks, to_fit=True, batch_size=1, dim=(512, 512),dimy=(512, 512), n_channels=1, n_classes=2, shuffle=False)
'''

######### 3D generators ###########
list_IDs = os.listdir(data_path["train3d"][0])
gen3da = generator3da(list_IDs, data_path["train3d"][0],data_path["train3d"][1], to_fit=True, batch_size=1, patch_size=4, dim=(256, 256), dimy=(256,256), n_channels=1, n_classes=2, shuffle=True, data_gen_args=data_gen_args_dict)
list_IDs = os.listdir(data_path["val3d"][0])
gen3d = generator3d(list_IDs, data_path["val3d"][0],data_path["val3d"][1], to_fit=True, batch_size=1, patch_size=4, dim=(256, 256), dimy=(256,256), n_channels=1, n_classes=2, shuffle=True)
list_IDs = os.listdir(test_paths['test3d'][0])
testgen3d = generator3d(list_IDs, test_paths['test3d'][0],test_paths['test3d'][1], to_fit=True, batch_size=1, patch_size=4, dim=(256, 256), dimy=(256,256), n_channels=1, n_classes=2, shuffle=False)

'''list_IDs = os.listdir('C:/Datasets/elderlymen1/3dmediumcsa/train/images')
gen3da = generator3da(list_IDs, 'C:/Datasets/elderlymen1/3dmediumcsa/train/images','C:/Datasets/elderlymen1/3dmediumcsa/train/masks', to_fit=True, batch_size=1, patch_size=4, dim=(256, 256), dimy=(256,256), n_channels=1, n_classes=2, shuffle=True, data_gen_args=data_gen_args_dict)
list_IDs = os.listdir('C:/Datasets/elderlymen1/3dmediumcsa/val/images')
gen3d = generator3d(list_IDs,'C:/Datasets/elderlymen1/3dmediumcsa/val/images','C:/Datasets/elderlymen1/3dmediumcsa/val/masks', to_fit=True, batch_size=1, patch_size=4, dim=(256, 256), dimy=(256,256), n_channels=1, n_classes=2, shuffle=True)
'''
#plotFromGenerator3d(gen3da)

############# model ##############
try:
    sample = open('metrics.txt', '+r')
except:
    sample = open('metrics.txt', 'x')
    sample = open('metrics.txt', '+r')

model = unet3d(pretrained_weights='unet_ThighOuterSurface.hdf5',input_size=(256, 256,4, 1))

'''for i in (5,3,4):
    list_IDs = os.listdir(test_paths[i+3][0])
    testgen3d = generator3d(list_IDs, test_paths[i+3][0], test_paths[i+3][1], to_fit=True, batch_size=1,
                            patch_size=4, dim=(256, 256), dimy=(256, 256), n_channels=1, n_classes=2, shuffle=False)
    results = model.predict_generator(testgen3d, len(testgen3d), verbose=1)
    saveResult3d(results, patch_size=4,save_path=save_paths3d[i-1][0],test_frames_path=save_paths3d[i-1][1],framepath2=save_paths3d[i-1][2],overlay_path=save_paths3d[i-1][3],overlay_path2=save_paths3d[i-1][4])

    loss, acc = model.evaluate_generator(testgen3d, len(testgen3d), verbose=0)
    print(loss,acc)
    print('Test loss: {} Test accuracy {}'.format(loss, acc),file=sample)
    sample.close()'''

print('{} {}'.format(date.today(),datetime.now()),file=sample)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('unet_ThighOuterSurfaceval.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
model_checkpoint2 = tf.keras.callbacks.ModelCheckpoint('unet_ThighOuterSurface.hdf5', monitor='loss', verbose=1, save_best_only=True)

clr_step_size = int(2 * 798)
base_lr = 1e-5
max_lr = 1e-4
mode='exp_range'
clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, step_size=clr_step_size, mode=mode)
#model = unet3d(pretrained_weights='unet_ThighOuterSurface.hdf5', input_size=(512, 512,2, 1))
#model = unet3d( pretrained_weights='unet_ThighOuterSurface.hdf5',input_size=(512, 512,2, 1))
model = unet3d(pretrained_weights='3d/unet_ThighOuterSurface.hdf5',input_size=(256, 256, 4,1))
#model = unet3d(pretrained_weights='unet_ThighOuterSurface.hdf5',input_size=(256, 256, 4,1))
history = model.fit_generator(generator=gen3da, validation_data=gen3d, validation_steps=len(gen3d), steps_per_epoch=len(gen3da), epochs=50, callbacks=[model_checkpoint2, model_checkpoint])
######### predict and save ###########

print(history.history.keys(),file = sample)

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training loss', 'validation loss'], loc='upper left')
plt.savefig('loss.png')
plt.figure()
plt.plot(history.history['dice_accuracy'])
plt.plot(history.history['val_dice_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train accuracy', 'validation accuracy'], loc='upper left')
plt.savefig('accuracy.png')
print('{} {}'.format(date.today(),datetime.now()),file=sample)

#3D

'''model = unet3d(pretrained_weights='unet_ThighOuterSurface.hdf5',input_size=(256, 256,4, 1))
results = model.predict_generator(testgen3d, 28*len(list_IDs), verbose=1)
saveResult3d(results, patch_size=4,save_path=save_paths3d[1][0],test_frames_path=save_paths3d[1][1],framepath2=save_paths3d[1][2],overlay_path=save_paths3d[1][3],overlay_path2=save_paths3d[1][4])

loss, acc = model.evaluate_generator(testgen3d, len(testgen3d), verbose=0)
print(loss,acc)
print('Test loss: {} Test accuracy {}'.format(loss, acc),file=sample)'''

sample.close()