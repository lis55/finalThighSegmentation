from data import *
from model import *
from generators import *
import os
import matplotlib.pyplot as plt
from vtk.util import numpy_support
import SimpleITK as sitk
from datetime import datetime
from datetime import date

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""

######### 2D generators ###########
list_IDs = os.listdir(data_path["train"][0])
genaug =DataGenerator2(list_IDs, data_path["train"][0],data_path["train"][1], to_fit=True, batch_size=2, dim=(512, 512),dimy=(512, 512), n_channels=1, n_classes=2, shuffle=True, data_gen_args =data_gen_args_dict)
list_IDs = os.listdir(data_path["val"][0])
genval =DataGenerator(list_IDs, data_path["val"][0],data_path["val"][1], to_fit=True, batch_size=2, dim=(512, 512),dimy=(512, 512), n_channels=1, n_classes=2, shuffle=True)
list_IDs = os.listdir(test_paths[1][0])
testgen = DataGenerator(list_IDs, test_paths[1][0],test_paths[1][1], to_fit=True, batch_size=1, dim=(512, 512),dimy=(512, 512), n_channels=1, n_classes=2, shuffle=False)
######### 3D generators ###########
list_IDs = os.listdir(data_path["train3d"][0])
gen3da = generator3da(list_IDs, data_path["train3d"][0],data_path["train3d"][1], to_fit=True, batch_size=1, patch_size=8, dim=(128, 128), dimy=(128,128), n_channels=1, n_classes=2, shuffle=True, data_gen_args=data_gen_args_dict)
list_IDs = os.listdir(data_path["val3d"][0])
gen3d = generator3d(list_IDs, data_path["val3d"][0],data_path["val3d"][1], to_fit=True, batch_size=1, patch_size=8, dim=(128, 128), dimy=(128,128), n_channels=1, n_classes=2, shuffle=True)
list_IDs = os.listdir(test_paths[2][0])
testgen3d = generator3d(list_IDs, test_paths[2][0],test_paths[2][1], to_fit=True, batch_size=1, patch_size=8, dim=(128, 128), dimy=(128,128), n_channels=1, n_classes=2, shuffle=False)
############# models ##############
sample = open('metrics.txt', 'w')
print('{} {}'.format(date.today(),datetime.now()))
model_checkpoint = tf.keras.callbacks.ModelCheckpoint.ModelCheckpoint('unet_ThighOuterSurfaceval.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
model_checkpoint2 = tf.keras.callbacks.ModelCheckpoint.ModelCheckpoint('unet_ThighOuterSurface.hdf5', monitor='loss', verbose=1, save_best_only=True)

#2d
model = unet()
history = model.fit_generator(genaug, validation_data=genval, validation_steps=196, steps_per_epoch=500, epochs=100, callbacks=[model_checkpoint, model_checkpoint2])
#3d
#model = unet3d()
#history = model.fit_generator(generator=gen3da, validation_data=gen3d, validation_steps=49, steps_per_epoch=399, epochs=100, callbacks=[model_checkpoint2, model_checkpoint])

######### predict and save ###########


print(history.history.keys(),file = sample)
#2D
loss, acc = model.evaluate_generator(testgen, len(testgen), verbose=0)
#3D
#loss, acc = model.evaluate_generator(testgen3d, len(testgen3d), verbose=0)
print()
print('Test loss: {} Test accuracy {}'.format(loss, acc))
print('{} {}'.format(date.today(), datetime.now()))
sample.close()

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