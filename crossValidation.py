import os, random, re, shutil,gc
import tensorflow as tf
from tuning import *
from metrics import *
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
def kfoldsplit(FRAME_PATH, MASK_PATH,k):

    kfold = []
    all_frames = os.listdir(FRAME_PATH)
    all_masks = os.listdir(MASK_PATH)

    all_frames.sort(key=lambda var: [int(x) if x.isdigit() else x
                                     for x in re.findall(r'[^0-9]|[0-9]+', var)])
    all_masks.sort(key=lambda var: [int(x) if x.isdigit() else x
                                    for x in re.findall(r'[^0-9]|[0-9]+', var)])

    random.seed(230)
    random.shuffle(all_frames)

    # Generate train, val, and test sets for frames

    '''    
    train_split = int(0.8 * len(all_frames))
    #val_split = int(0.9 * len(all_frames))
    #test_split = int(0.9 * len(all_frames))

    train_frames = all_frames[:train_split]
    #val_frames = all_frames[train_split:val_split]
    test_frames = all_frames[train_split:]

    # Generate corresponding mask lists for masks

    train_masks = [f for f in all_masks if 'image_' + f[6:16] + 'dcm' in train_frames]
    #val_masks = [f for f in all_masks if 'image_' + f[6:16] + 'dcm' in val_frames]
    test_masks = [f for f in all_masks if 'image_' + f[6:16] + 'dcm' in test_frames]
    size_of_subset =int(len(train_masks)/k)'''
    all_masks = [f for f in all_masks if 'image_' + f[6:16] + 'dcm' in all_frames]
    size_of_subset = int(len(all_frames) / k)
    for i in range (0,k):
        #subset = (train_frames[i*size_of_subset:(i+1)*size_of_subset],train_masks[i*size_of_subset:(i+1)*size_of_subset])
        subset = (all_frames[i * size_of_subset:(i + 1) * size_of_subset],
                  all_masks[i * size_of_subset:(i + 1) * size_of_subset])
        kfold.append(subset)

    #return kfold, (test_frames,test_masks)
    return kfold



def get_model_name(k):
    return 'model_'+str(k)+'.hdf5'

file = open('crossvalsets.txt')
lines = file.read()
lines = eval(lines)
frames_path = 'C:/Datasets/elderlymen1/2d/images'
masks_path = 'C:/Datasets/elderlymen1/2d/FASCIA_FILLED'
kf = kfoldsplit(frames_path, masks_path, 10)

def crossvalidation(epoch,kf, loops):
    VALIDATION_ACCURACY = []
    VALIDATION_LOSS = []
    TRAINING_ACCURACY = []
    TRAINING_LOSS = []
    Params=[]


    save_dir = 'C:/saved_models/'
    fold_var = 1
    values_alpha = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    values_lr = [1e-4,2e-4,3e-4,4e-4,5e-4,6e-4,7e-4,8e-4,9e-4]
    values_alpha = [0.2]
    values_lr = [2e-4]

    for i in values_alpha:
        for j in values_lr:
        #while i <= loops:
            #_alpha = random.uniform(0, 1)
            #lrate = random.uniform(1e-3, 1e-6)
            _alpha = i
            lrate = j
            Params.append([_alpha,lrate])
            for subset in kf:
                frames = os.listdir(frames_path)
                for u in subset[0]:
                    frames.remove(u)

                list_IDs = frames
                train_data_generator = DataGenerator2(list_IDs, frames_path, masks_path, to_fit=True, batch_size=2,
                                        dim=(512, 512), dimy=(512, 512), n_channels=1, n_classes=2, shuffle=True,
                                        data_gen_args=data_gen_args_dict)
                list_IDs = subset[0]
                valid_data_generator = DataGenerator(list_IDs, frames_path, masks_path, to_fit=True, batch_size=2,
                                       dim=(512, 512), dimy=(512, 512), n_channels=1, n_classes=2, shuffle=True)

                # CREATE NEW MODEL
                model = unet(pretrained_weights='csa/unet_ThighOuterSurface.hdf5')
                # COMPILE NEW MODEL
                model.compile(optimizer=tf.keras.optimizers.Adam(lr=lrate), loss=combo_loss(alpha=_alpha, beta=0.4), metrics=[dice_accuracy])

                # CREATE CALLBACKS
                checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir + get_model_name(fold_var),
                                                                monitor='val_loss', verbose=1,
                                                                save_best_only=True, mode='max')
                callbacks_list = [checkpoint]
                # There can be other callbacks, but just showing one because it involves the model name
                # This saves the best model
                # FIT THE MODEL
                history = model.fit(train_data_generator, validation_steps=len(valid_data_generator), steps_per_epoch=len(train_data_generator),
                                    epochs=epoch,
                                    callbacks=callbacks_list,
                                    validation_data=valid_data_generator)
                # PLOT HISTORY
                #		:
                #		:

                # LOAD BEST MODEL to evaluate the performance of the model
                model.load_weights("C:/saved_models/model_" + str(fold_var) + ".hdf5")

                results = model.evaluate(valid_data_generator)
                results = dict(zip(model.metrics_names, results))

                VALIDATION_ACCURACY.append(results['dice_accuracy'])
                VALIDATION_LOSS.append(results['loss'])
                #TRAINING_ACCURACY.append(history['accuracy'])
                #TRAINING_LOSS.append(history['loss'])

                tf.keras.backend.clear_session()

                fold_var += 1
                del model
                gc.collect()
            #i+=1

    print(VALIDATION_ACCURACY)
    print(Params)
    sample = open('crossval.txt', '+r')
    print(VALIDATION_ACCURACY, file=sample)
    print(".......")
    print(".......")
    print(VALIDATION_LOSS, file=sample)
    print(".......")
    print(".......")
    print(TRAINING_ACCURACY, file=sample)
    print(".......")
    print(".......")
    print(TRAINING_LOSS, file=sample)
    print(".......")
    print(".......")
    print(Params, file=sample)
    print('...',file=sample)
    print(np.mean(VALIDATION_ACCURACY),file=sample)
    sample.close()



masks_path = 'C:/final_results/elderlymen2/2d'
img2 = np.zeros((512,512,1))
dice = []
images = os.listdir(masks_path)
for img in images[0:137]:
    img = load_grayscale_image_VTK(os.path.join(masks_path,img))
    dice.append(dice_coef(img, img2))
print(np.mean(dice))

dice = []
for img in images[0:2*137]:
    img = load_grayscale_image_VTK(os.path.join(masks_path,img))
    dice.append(dice_coef(img, img2))
print(np.mean(dice))

dice = []
for img in images[0:3*137]:
    img = load_grayscale_image_VTK(os.path.join(masks_path,img))
    dice.append(dice_coef(img, img2))
print(np.mean(dice))

dice = []
for img in images[0:4*137]:
    img = load_grayscale_image_VTK(os.path.join(masks_path,img))
    dice.append(dice_coef(img, img2))
print(np.mean(dice))

dice = []
for img in images[0:5*137]:
    img = load_grayscale_image_VTK(os.path.join(masks_path,img))
    dice.append(dice_coef(img, img2))
print(np.mean(dice))

dice = []
for img in images[0:6*137]:
    img = load_grayscale_image_VTK(os.path.join(masks_path,img))
    dice.append(dice_coef(img, img2))
print(np.mean(dice))

dice = []
for img in images[0:7*137]:
    img = load_grayscale_image_VTK(os.path.join(masks_path,img))
    dice.append(dice_coef(img, img2))
print(np.mean(dice))

dice = []
for img in images[0:8*137]:
    img = load_grayscale_image_VTK(os.path.join(masks_path,img))
    dice.append(dice_coef(img, img2))
print(np.mean(dice))



frames_path = 'C:/Datasets/elderlymen1/2d/train_frames'
masks_path = 'C:/Datasets/elderlymen1/2d/train_masks'
VALIDATION_ACCURACY = []
VALIDATION_LOSS = []
list_IDs = kf[0][0]
valid_data_generator = DataGenerator(list_IDs, frames_path, masks_path, to_fit=True, batch_size=2,
                                     dim=(512, 512), dimy=(512, 512), n_channels=1, n_classes=2, shuffle=True)
model = unet(pretrained_weights='final/unet_ThighOuterSurface.hdf5')
# COMPILE NEW MODEL
model.compile(optimizer=tf.keras.optimizers.Adam(lr=2e-4), loss=combo_loss(alpha=0.2, beta=0.4),
              metrics=[dice_accuracy])
results = model.evaluate(valid_data_generator)
results = dict(zip(model.metrics_names, results))

VALIDATION_ACCURACY.append(results['dice_accuracy'])
VALIDATION_LOSS.append(results['loss'])
list_IDs = kf[0][0]+kf[1][0]
valid_data_generator = DataGenerator(list_IDs, frames_path, masks_path, to_fit=True, batch_size=2,
                                     dim=(512, 512), dimy=(512, 512), n_channels=1, n_classes=2, shuffle=True)
results = model.evaluate(valid_data_generator)
results = dict(zip(model.metrics_names, results))

VALIDATION_ACCURACY.append(results['dice_accuracy'])
VALIDATION_LOSS.append(results['loss'])
list_IDs = kf[0][0]+kf[1][0]+kf[2][0]
valid_data_generator = DataGenerator(list_IDs, frames_path, masks_path, to_fit=True, batch_size=2,
                                     dim=(512, 512), dimy=(512, 512), n_channels=1, n_classes=2, shuffle=True)
results = model.evaluate(valid_data_generator)
results = dict(zip(model.metrics_names, results))

VALIDATION_ACCURACY.append(results['dice_accuracy'])
VALIDATION_LOSS.append(results['loss'])
list_IDs = kf[0][0]+kf[1][0]+kf[2][0]+kf[3][0]
valid_data_generator = DataGenerator(list_IDs, frames_path, masks_path, to_fit=True, batch_size=2,
                                     dim=(512, 512), dimy=(512, 512), n_channels=1, n_classes=2, shuffle=True)
results = model.evaluate(valid_data_generator)
results = dict(zip(model.metrics_names, results))

VALIDATION_ACCURACY.append(results['dice_accuracy'])
VALIDATION_LOSS.append(results['loss'])
list_IDs = kf[0][0]+kf[1][0]+kf[2][0]+kf[3][0]+kf[4][0]
valid_data_generator = DataGenerator(list_IDs, frames_path, masks_path, to_fit=True, batch_size=2,
                                     dim=(512, 512), dimy=(512, 512), n_channels=1, n_classes=2, shuffle=True)
results = model.evaluate(valid_data_generator)
results = dict(zip(model.metrics_names, results))

VALIDATION_ACCURACY.append(results['dice_accuracy'])
VALIDATION_LOSS.append(results['loss'])
list_IDs = kf[0][0]+kf[1][0]+kf[2][0]+kf[3][0]+kf[4][0]+kf[5][0]
valid_data_generator = DataGenerator(list_IDs, frames_path, masks_path, to_fit=True, batch_size=2,
                                     dim=(512, 512), dimy=(512, 512), n_channels=1, n_classes=2, shuffle=True)
results = model.evaluate(valid_data_generator)
results = dict(zip(model.metrics_names, results))

VALIDATION_ACCURACY.append(results['dice_accuracy'])
VALIDATION_LOSS.append(results['loss'])
list_IDs = kf[0][0]+kf[1][0]+kf[2][0]+kf[3][0]+kf[4][0]+kf[5][0]+kf[6][0]
valid_data_generator = DataGenerator(list_IDs, frames_path, masks_path, to_fit=True, batch_size=2,
                                     dim=(512, 512), dimy=(512, 512), n_channels=1, n_classes=2, shuffle=True)
results = model.evaluate(valid_data_generator)
results = dict(zip(model.metrics_names, results))

VALIDATION_ACCURACY.append(results['dice_accuracy'])
VALIDATION_LOSS.append(results['loss'])
list_IDs = kf[0][0]+kf[1][0]+kf[2][0]+kf[3][0]+kf[4][0]+kf[5][0]+kf[6][0]+kf[7][0]
valid_data_generator = DataGenerator(list_IDs, frames_path, masks_path, to_fit=True, batch_size=2,
                                     dim=(512, 512), dimy=(512, 512), n_channels=1, n_classes=2, shuffle=True)
results = model.evaluate(valid_data_generator)
results = dict(zip(model.metrics_names, results))
VALIDATION_ACCURACY.append(results['dice_accuracy'])
VALIDATION_LOSS.append(results['loss'])
list_IDs = kf[0][0]+kf[1][0]+kf[2][0]+kf[3][0]+kf[4][0]+kf[5][0]+kf[6][0]+kf[7][0]+kf[8][0]
valid_data_generator = DataGenerator(list_IDs, frames_path, masks_path, to_fit=True, batch_size=2,
                                     dim=(512, 512), dimy=(512, 512), n_channels=1, n_classes=2, shuffle=True)
results = model.evaluate(valid_data_generator)
results = dict(zip(model.metrics_names, results))
VALIDATION_ACCURACY.append(results['dice_accuracy'])
VALIDATION_LOSS.append(results['loss'])

list_IDs = kf[0][0]+kf[1][0]+kf[2][0]+kf[3][0]+kf[4][0]+kf[5][0]+kf[6][0]+kf[7][0]+kf[9][0]
valid_data_generator = DataGenerator(list_IDs, frames_path, masks_path, to_fit=True, batch_size=2,
                                     dim=(512, 512), dimy=(512, 512), n_channels=1, n_classes=2, shuffle=True)
results = model.evaluate(valid_data_generator)
results = dict(zip(model.metrics_names, results))
VALIDATION_ACCURACY.append(results['dice_accuracy'])
VALIDATION_LOSS.append(results['loss'])

print(VALIDATION_ACCURACY)
print(VALIDATION_LOSS)

crossvalidation(2,kf, 2)