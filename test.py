
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
import pickle


'''file_to_read1 = open("C:/Users/lis/Desktop/overfitting/relu+dice/trainHistoryDict", "rb")
loaded_dictionary1 = pickle.load(file_to_read1)
file_to_read2 = open("C:/Users/lis/Desktop/overfitting/elu+dice/trainHistoryDict", "rb")
loaded_dictionary2 = pickle.load(file_to_read2)
file_to_read3 = open("C:/Users/lis/Desktop/overfitting/swish+dice/trainHistoryDict", "rb")
loaded_dictionary3 = pickle.load(file_to_read3)



plt.figure()
plt.plot(loaded_dictionary1['val_loss'][0::])
plt.plot(loaded_dictionary2['val_loss'][0::])
plt.plot(loaded_dictionary3['val_loss'][0::])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.yscale('log')
plt.xscale('log')
plt.legend(['ReLU', 'ELU', 'Swish and ELU'], loc='upper right')
plt.savefig('losses.png')

plt.figure()
plt.plot(loaded_dictionary1['val_dice_accuracy'][0::])
plt.plot(loaded_dictionary2['val_dice_accuracy'][0::])
plt.plot(loaded_dictionary3['val_dice_accuracy'][0::])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.yscale('log')
plt.xscale('log')
plt.legend(['ReLU', 'ELU', 'Swish and ELU'], loc='lower right')
plt.savefig('acc.png')
print(np.mean(loaded_dictionary1['val_loss'][200::]))
print(np.mean(loaded_dictionary2['val_loss'][200::]))
print(np.mean(loaded_dictionary3['val_loss'][200::]))'''

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""

os.environ['TF_CUDNN_DETERMINISTIC']='1'
random.seed(123)
np.random.seed(123)
#tf.random.set_seed(1234)
tf.random.set_random_seed(123)

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
######### 2D generators ###########
'''list_IDs = os.listdir(data_path["train"][0])
genaug =DataGenerator2(list_IDs, data_path["train"][0],data_path["train"][1], to_fit=True, batch_size=2, dim=(512, 512),dimy=(512, 512), n_channels=1, n_classes=2, shuffle=True, data_gen_args =data_gen_args_dict)
list_IDs = os.listdir(data_path["val"][0])
genval =DataGenerator(list_IDs, data_path["val"][0],data_path["val"][1], to_fit=True, batch_size=2, dim=(512, 512),dimy=(512, 512), n_channels=1, n_classes=2, shuffle=True)
list_IDs = os.listdir(test_paths[1][0])
testgen = DataGenerator(list_IDs, test_paths[1][0],test_paths[1][1], to_fit=True, batch_size=1, dim=(512, 512),dimy=(512, 512), n_channels=1, n_classes=2, shuffle=False)
'''

#plotFromGenerator3d(gen3da)

############# model ##############
sample = open('splitting.txt', 'r')
sets = sample.readline()
sample.close()
sets = eval(sets)

frames_path = 'C:/Datasets/elderlymen1/2d/images'
masks_path = 'C:/Datasets/elderlymen1/2d/FASCIA_FILLED'
#masks_path = 'C:/CSA/CSA'

list_IDs = sets[0][0][0:137]
#genaug =DataGenerator2(list_IDs, frames_path,masks_path, to_fit=True, batch_size=2, dim=(512, 512),dimy=(512, 512), n_channels=1, n_classes=2, shuffle=True, data_gen_args =data_gen_args_dict)
genaug =DataGenerator(list_IDs, frames_path,masks_path, to_fit=True, batch_size=2, dim=(512, 512),dimy=(512, 512), n_channels=1, n_classes=2, shuffle=True)
list_IDs = sets[1][1]
genval =DataGenerator(list_IDs, frames_path,masks_path, to_fit=True, batch_size=2, dim=(512, 512),dimy=(512, 512), n_channels=1, n_classes=2, shuffle=True)
list_IDs = sets[2][0]
testgen = DataGenerator(list_IDs, frames_path,masks_path, to_fit=True, batch_size=1, dim=(512, 512),dimy=(512, 512), n_channels=1, n_classes=2, shuffle=False)

'''sample = open('metrics.txt', '+r')
model = unet_no_dropout(pretrained_weights='unet_ThighOuterSurface.hdf5')
results = model.predict_generator(testgen, len(list_IDs), verbose=1)
#saveResult(save_path2d[1][0], results, test_frames_path=test_paths[1][0],overlay=True,overlay_path=save_path2d[1][1])
#saveResult(save_path2d[1][0], results, test_frames_path=sets[2][0],overlay=True,overlay_path=save_path2d[1][1])
#saveResult(save_path2d[1][0], results, test_frames_path=frames_path,overlay=True,overlay_path=save_path2d[1][1])
loss, acc = model.evaluate_generator(testgen, len(testgen), verbose=0)
print(loss,acc)
print('Test loss: {} Test accuracy {}'.format(loss, acc),file=sample)
print('{} {}'.format(date.today(),datetime.now()),file=sample)
list_IDs = sets[0][0][0:2]
results = model.predict_generator(genaug, len(list_IDs), verbose=1)
#saveResult(save_path2d[1][0], results, test_frames_path=test_paths[1][0],overlay=True,overlay_path=save_path2d[1][1])
#saveResult(save_path2d[1][0], results, test_frames_path=sets[2][0],overlay=True,overlay_path=save_path2d[1][1])
#saveResult(save_path2d[1][0], results, test_frames_path=frames_path,overlay=True,overlay_path=save_path2d[1][1])
loss, acc = model.evaluate_generator(genaug, len(genaug), verbose=0)
print(loss,acc)
print('Test loss: {} Test accuracy {}'.format(loss, acc),file=sample)
print('{} {}'.format(date.today(),datetime.now()),file=sample)

sample.close()
'''



try:
    sample = open('metrics.txt', '+r')
except:
    sample = open('metrics.txt', 'x')
    sample.close()
    sample = open('metrics.txt', '+r')
print('{} {}'.format(date.today(),datetime.now()),file=sample)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('unet_ThighOuterSurfaceval.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
model_checkpoint2 = tf.keras.callbacks.ModelCheckpoint('unet_ThighOuterSurface.hdf5', monitor='loss', verbose=1, save_best_only=True)
#2d
#model = unet(pretrained_weights='csa/unet_ThighOuterSurface.hdf5')
model = unet_no_dropout('swish2+dice2ndrun/unet_ThighOuterSurface.hdf5')
history = model.fit(genaug, validation_data=genval, validation_steps=len(genval), steps_per_epoch=len(genaug), epochs=1000, callbacks=[model_checkpoint, model_checkpoint2])
#history = model.fit(genaug, validation_data=genval, validation_steps=len(genval), steps_per_epoch=1, epochs=1, callbacks=[clr,model_checkpoint, model_checkpoint2])

######### predict and save ###########

with open('trainHistoryDict', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

print(history.history.keys(),file = sample)

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training loss', 'validation loss'], loc='upper right')
plt.savefig('loss.png')
plt.figure()
plt.plot(history.history['dice_accuracy'])
plt.plot(history.history['val_dice_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train accuracy', 'validation accuracy'], loc='upper right')
plt.savefig('accuracy.png')
print('{} {}'.format(date.today(),datetime.now()),file=sample)
#2D
model = unet_no_dropout(pretrained_weights='unet_ThighOuterSurfaceval.hdf5')
results = model.predict_generator(testgen, len(list_IDs), verbose=1)
#saveResult(save_path2d[1][0], results, test_frames_path=test_paths[1][0],overlay=True,overlay_path=save_path2d[1][1])
#saveResult(save_path2d[1][0], results, test_frames_path=sets[2][0],overlay=True,overlay_path=save_path2d[1][1])
saveResult(save_path2d[1][0], results, test_frames_path=frames_path,overlay=True,overlay_path=save_path2d[1][1])
loss, acc = model.evaluate_generator(testgen, len(testgen), verbose=0)
print(loss,acc)
print('Test loss: {} Test accuracy {}'.format(loss, acc),file=sample)
print('{} {}'.format(date.today(),datetime.now()),file=sample)

sample.close()


