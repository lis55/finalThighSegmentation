import os, random, re, shutil
import tensorflow as tf
from tuning import *
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""

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
    size_of_subset =int(len(train_masks)/k)
    for i in range (0,k):
        subset = (train_frames[i*size_of_subset:(i+1)*size_of_subset],train_masks[i*size_of_subset:(i+1)*size_of_subset])
        kfold.append(subset)

    return kfold, (test_frames,test_masks)



def get_model_name(k):
    return 'model_'+str(k)+'.hdf5'

def float_range(start, stop, step):
  while start < stop:
    yield float(start)
    start += decimal.Decimal(step)

frames_path = 'C:/Datasets/elderlymen1/2d/images'
masks_path = 'C:/Datasets/elderlymen1/2d/FASCIA_FILLED'
kf = kfoldsplit(frames_path, masks_path, 10)

def crossvalidation(epoch,kf, loops):
    VALIDATION_ACCURACY = []
    VALIDATION_LOSS = []
    Params=[]


    save_dir = 'C:/saved_models/'
    fold_var = 1
    i=0
    for i in float_range(0,1,0.1):
        for j in float_range(1e-6,1e-3,1e-6):

        #while i <= loops:
            #_alpha = random.uniform(0, 1)
            #lrate = random.uniform(1e-3, 1e-6)
            _alpha = i
            lrate = j
            Params.append([_alpha,lrate])
            for subset in kf[0]:
                list_IDs = subset[0]
                train_data_generator = DataGenerator2(list_IDs, frames_path, masks_path, to_fit=True, batch_size=2,
                                        dim=(512, 512), dimy=(512, 512), n_channels=1, n_classes=2, shuffle=True,
                                        data_gen_args=data_gen_args_dict)
                list_IDs = kf[1][0]
                valid_data_generator = DataGenerator(list_IDs, frames_path, masks_path, to_fit=True, batch_size=2,
                                       dim=(512, 512), dimy=(512, 512), n_channels=1, n_classes=2, shuffle=True)

                # CREATE NEW MODEL
                model = unet(pretrained_weights='csa/unet_ThighOuterSurface.hdf5')
                # COMPILE NEW MODEL
                model.compile(optimizer=Adam(lr=lrate), loss=combo_loss(alpha=_alpha, beta=0.4), metrics=[dice_accuracy])

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

                tf.keras.backend.clear_session()

                fold_var += 1
            #i+=1

        print(VALIDATION_ACCURACY)
        print(Params)
        sample = open('metrics.txt', '+r')
        print(VALIDATION_ACCURACY, file=sample)
        print(Params, file=sample)
        print('...',file=sample)
        sample.close()

crossvalidation(15,kf, 2)