import skimage.io as io
import random
import re
import shutil, os
import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
import vtk
from vtk.util import numpy_support
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import SimpleITK as sitk
from keras.callbacks import Callback
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]
COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])

data_gen_args_dict = dict(shear_range=20,
                    rotation_range=20,
                    horizontal_flip=True,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    fill_mode='nearest')
data_path ={"train":['C:/Datasets/elderlymen1/2d/train_frames', 'C:/Datasets/elderlymen1/2d/train_masks'],
            "val":['C:/Datasets/elderlymen1/2d/val_frames', 'C:/Datasets/elderlymen1/2d/val_masks'],
            "train3d": ['C:/Datasets/elderlymen1/3ddownsampled/train/images', 'C:/Datasets/elderlymen1/3ddownsampled/train/masks'],
            "val3d": ['C:/Datasets/elderlymen1/3ddownsampled/val/images', 'C:/Datasets/elderlymen1/3ddownsampled/val/masks']
            }
test_paths = {1:['C:/Datasets/elderlymen1/2d/test_frames','C:/Datasets/elderlymen1/2d/test_masks'],
              2:['C:/Datasets/elderlymen1/3ddownsampled/test/images','C:/Datasets/elderlymen1/3ddownsampled/test/masks'],
              3:['C:/Datasets/elderlymen2/2d/images','C:/Datasets/elderlymen2/2d/FASCIA_FINAL'],
              4:['C:/Datasets/youngmen/2d/images','C:/Datasets/youngmen/2d/FASCIA_FINAL'],
              5:['C:/Datasets/elderlywomen/2d/images','C:/Datasets/elderlywomen/2d/FASCIA_FINAL']
              }
save_paths3d ={1:['C:/final_results/elderlymen1/3d', 'C:/Datasets/elderlymen1/3ddownsampled/test/images',
               'C:/Datasets/elderlymen1/2d/images','C:/final_results/elderlymen1/3doverlaydown', 'C:/final_results/elderlymen2/3doverlay'],
            2:['C:/final_results/elderlymen2/3d', 'C:/Datasets/elderlymen2/3ddownsampled/image',
               'C:/Datasets/elderlymen2/2d/images','C:/final_results/elderlymen2/3doverlaydown', 'C:/final_results/elderlymen2/3doverlay'],
            3:['C:/final_results/youngmen/3d', 'C:/Datasets/youngmen/3ddownsampled/image',
               'C:/Datasets/youngmen/2d/images','C:/final_results/youngmen/3doverlaydown', 'C:/final_results/youngmen/3doverlay'],
            4:['C:/final_results/elderlywomen/3d', 'C:/Datasets/elderlywomen/3ddownsampled/image',
               'C:/Datasets/elderlywomen/2d/images', 'C:/final_results/elderlywomen/3doverlaydown',
               'C:/final_results/elderlywomen/3doverlay']
             }
save_path2d = {1:['C:/final_results/elderlymen1/2d','C:/final_results/elderlymen1/2doverlay1' ,'C:/final_results/elderlymen1/2doverlay2'],
               2:['C:/final_results/elderlymen2/2d','C:/final_results/elderlymen2/2doverlay1' ,'C:/final_results/elderlymen2/2doverlay2'],
               3:['C:/final_results/youngmen/2d','C:/final_results/youngmen/2doverlay1' ,'C:/final_results/youngmen/2doverlay2'],
               4:['C:/final_results/elderlywomen/2d','C:/final_results/elderlywomen/2doverlay1' ,'C:/final_results/elderlywomen/2doverlay2']}

def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255

def load_grayscale_image_VTK(image_path):
    """Load grayscale image
    :param image_path: path to image to load
    :return: loaded image
    """
    img = vtk.vtkPNGReader()
    img.SetFileName(os.path.normpath(image_path))
    img.Update()


    _extent = img.GetDataExtent()
    ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]

    img_data = img.GetOutput()
    datapointer = img_data.GetPointData()
    assert (datapointer.GetNumberOfArrays()==1)
    vtkarray = datapointer.GetArray(0)
    img = vtk.util.numpy_support.vtk_to_numpy(vtkarray)
    img = img.reshape(ConstPixelDims, order='F')

    img = img / np.max(img)
    img = img.astype('float32')

    return img


def load_dicom(foldername, doflipz = True):
    reader = vtk.vtkDICOMImageReader()
    reader.SetFileName(foldername)
    reader.Update()

    #note: It workes when the OS sorts the files correctly by itself. If the files weren’t properly named, lexicographical sorting would have given a messed up array. In that
    # case you need to loop and pass each file to a separate reader through the SetFileName method, or you’d have to create a vtkStringArray, push the sorted filenames, and use
    # the vtkDICOMImageReader.SetFileNames method.

    # Load meta data: dimensions using `GetDataExtent`
    _extent = reader.GetDataExtent()
    ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]

    # Get the 'vtkImageData' object from the reader
    imageData = reader.GetOutput()
    # Get the 'vtkPointData' object from the 'vtkImageData' object
    pointData = imageData.GetPointData()
    # Ensure that only one array exists within the 'vtkPointData' object
    assert (pointData.GetNumberOfArrays()==1)
    # Get the `vtkArray` (or whatever derived type) which is needed for the `numpy_support.vtk_to_numpy` function
    arrayData = pointData.GetArray(0)

    # Convert the `vtkArray` to a NumPy array
    ArrayDicom = numpy_support.vtk_to_numpy(arrayData)
    # Reshape the NumPy array to 3D using 'ConstPixelDims' as a 'shape'
    ArrayDicom = ArrayDicom.reshape(ConstPixelDims, order='F')


    return ArrayDicom

def load_mhd(filename):
    ##ImageDataGenerator expects (z,x,y) order of array
    itk_image = sitk.ReadImage(filename, sitk.sitkFloat32)
    np_array = sitk.GetArrayFromImage(itk_image)
    ##change order of dimensions to (x,y,z)
    # np_array = np.moveaxis(np_array, 0, -1)

    return np_array

def plotFromGenerator3d(gen):
    count=0
    for i in gen:
      #pydicom.dcmread(gen(i))
      for batch in range(0,np.shape(i[0])[0]):
          for k in range(0,np.shape(i[0])[3]):
              background = i[0][batch,:,:,k,0]
              background = background/ np.max(background)
              background = (background * 255).astype('uint8')
              background = Image.fromarray(background)
              background = background.convert("RGBA")
              img = i[1][batch,:,:,k,0]
              overlay = Image.fromarray((img * 255).astype('uint8'))
              overlay = overlay.convert("RGBA")

              # Split into 3 channels
              r, g, b, a = overlay.split()

              # Increase Reds
              g = b.point(lambda i: i * 0)

              # Recombine back to RGB image
              overlay = Image.merge('RGBA', (r, g, b, a))
              new_img = Image.blend(background, overlay, 0.3)
              new_img.save(str(count)+ ".png", "PNG")
              count +=1
              '''  
              plt.imshow(new_img ,cmap=plt.cm.bone)
              plt.imsave("dicom.png", i[0][0, :, :, 0])
              plt.show()
              plt.imshow((i[1][0,:,:,0]),cmap=plt.cm.bone)
              plt.imsave("dicomlabel.png",i[1][0,:,:,0])
              plt.show()
              '''

def saveSitkPng(array, path):
    sitk_output_image = sitk.GetImageFromArray(array)
    sitk_output_image = sitk.Cast(sitk.RescaleIntensity(sitk_output_image), sitk.sitkUInt8)
    sitk.WriteImage(sitk_output_image, path)

def saveSitkMdh(npy_output,save_path):
    sitk_output_image = sitk.GetImageFromArray(npy_output)
    sitk.WriteImage(sitk_output_image, save_path)

def saveCv2Png(array, path):
    array = (array * 255).astype(np.int16)
    cv2.imwrite(path,array)

def export_as_mhd(mask_path,save_path,n_slices = 28):
    all_frames = os.listdir(mask_path)
    shape = load_grayscale_image_VTK(os.path.join(mask_path, all_frames[0])).shape
    mdh_array = np.zeros(shape + (n_slices))
    for i in range(0, len(all_frames), n_slices):
        name = all_frames[i]
        count = 0
        for j in range(i, i + n_slices):
            mdh_array[count]=load_grayscale_image_VTK(os.path.join(mask_path, all_frames[j]))
            count += 1
        saveSitkMdh(mdh_array,os.path.join(save_path,name))


def make_overlay(overlay, background):
    background = background[:, :, 0] / np.max(background[:, :, 0])
    background = Image.fromarray((background * 255).astype('uint8'))
    background = background.convert("RGBA")
    overlay = overlay.convert("RGBA")

    # Split into 3 channels
    r, g, b, a = overlay.split()

    # Increase Reds
    g = b.point(lambda i: i * 0)

    # Recombine back to RGB image
    overlay = Image.merge('RGBA', (r, g, b, a))

    new_img = Image.blend(background, overlay, 0.3)

    return new_img

def overlay3dup(background,overlay, size = (512, 512)):
    img =background[:,:,0]
    img = img/ np.max(img)
    background = Image.fromarray((img * 255).astype('uint8'))
    # background = background.rotate(90, expand=True)
    # background = Image.fromarray((img2).astype('float'))
    overlay = cv2.resize(overlay[:, :], size, interpolation=cv2.INTER_NEAREST)
    overlay = cv2.medianBlur(overlay, 5)
    overlay = Image.fromarray((overlay * 255).astype('uint8'))

    background = background.convert("RGBA")
    overlay = overlay.convert("RGBA")

    # Split into 3 channels
    r, g, b, a = overlay.split()

    # Increase Reds
    g = b.point(lambda i: i * 0)

    # Recombine back to RGB image
    overlay = Image.merge('RGBA', (r, g, b, a))

    new_img = Image.blend(background, overlay, 0.3)
    return new_img

def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2, test_frames_path=None, overlay=False, overlay_path=None):
    '''
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        #io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)
        io.imsave(os.path.join(save_path, os.listdir(test_frames_path)[i][:-4]+".png"), img)
    '''
    if overlay:
        all_frames = os.listdir(test_frames_path)
        for i, item in enumerate(npyfile):
            img = labelVisualize(num_class, COLOR_DICT, item) if flag_multi_class else item[:, :, 0]
            #img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            io.imsave(os.path.join(save_path, os.listdir(test_frames_path)[i][:-4] + ".png"), img)
            '''
            img2 = img.astype(np.float32)
            # --- the following holds the square root of the sum of squares of the image dimensions ---
            # --- this is done so that the entire width/height of the original image is used to express the complete circular range of the resulting polar image ---
            value = np.sqrt(((img2.shape[0] / 2.0) ** 2.0) + ((img2.shape[1] / 2.0) ** 2.0))
            polar_image = cv2.warpPolar(img2,img2.shape, (img2.shape[0] / 2, img2.shape[1] / 2), 800, cv2.WARP_FILL_OUTLIERS)
            polar_image = polar_image.astype(np.uint8)
            img = polar_image
            '''
            overlay = Image.fromarray((img*255).astype('uint8'))
            background = load_dicom(os.path.join(test_frames_path, all_frames[i]))
            #background = cv2.rotate(background, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img = make_overlay(overlay,background)
            img.save(os.path.join(overlay_path, 'image_' + all_frames[i][6:16] + 'png'), "PNG")


def saveResult3d( npyfile, patch_size=8, flag_multi_class=False, num_class=2, save_path = None, test_frames_path=None, framepath2=None, overlay_path=None,overlay_path2 = None):
    '''
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        #io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)
        io.imsave(os.path.join(save_path, os.listdir(test_frames_path)[i][:-4]+".png"), img)
    '''
    all_frames = os.listdir(test_frames_path)
    count = 0
    test_data = []
    number_of_patches = np.floor(len(os.listdir(test_frames_path + '/' + all_frames[0])) / patch_size)
    for i, ID in enumerate(all_frames):
        slices = os.listdir(os.path.join(test_frames_path, ID))
        while count < number_of_patches:
            patch = slices[(count * patch_size):((count + 1) * patch_size)]
            test_data.append([ID, patch])
            count += 1
        count = 0
    for j,item in enumerate(npyfile):
        for i in range(0,np.shape(npyfile)[3]):
            img = labelVisualize(num_class, COLOR_DICT, item) if flag_multi_class else item[:, :, i,0]
            io.imsave(os.path.join(save_path, test_data[j][1][i][:-4] + ".png"), img)
            imagepath = test_frames_path + '/' + test_data[j][0] + '/' + test_data[j][1][i]
            background = load_grayscale_image_VTK(imagepath)
            path = overlay_path+ '/'+ test_data[j][1][i][:-4] + '.png'
            make_overlay(background, img).save(path, "PNG")
            imagepath2 = framepath2 + '/'  + test_data[j][1][i][:-3]+'dcm'
            background = load_dicom(imagepath2)
            path = os.path.join(overlay_path2, test_data[j][1][i][:-4] + '.png')
            overlay3dup(background, img).save(path, "PNG")

class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    For more detail, please see paper.

    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```

    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(
                self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())

class LRFinder(Callback):
    """
    Up-to date version: https://github.com/WittmannF/LRFinder
    Example of usage:
        from keras.models import Sequential
        from keras.layers import Flatten, Dense
        from keras.datasets import fashion_mnist
        !git clone https://github.com/WittmannF/LRFinder.git
        from LRFinder.keras_callback import LRFinder
        # 1. Input Data
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        mean, std = X_train.mean(), X_train.std()
        X_train, X_test = (X_train-mean)/std, (X_test-mean)/std
        # 2. Define and Compile Model
        model = Sequential([Flatten(),
                            Dense(512, activation='relu'),
                            Dense(10, activation='softmax')])
        model.compile(loss='sparse_categorical_crossentropy', \
                      metrics=['accuracy'], optimizer='sgd')
        # 3. Fit using Callback
        lr_finder = LRFinder(min_lr=1e-4, max_lr=1)
        model.fit(X_train, y_train, batch_size=128, callbacks=[lr_finder], epochs=2)
    """

    def __init__(self, min_lr, max_lr, mom=0.9, stop_multiplier=None,
                 reload_weights=True, batches_lr_update=5):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.mom = mom
        self.reload_weights = reload_weights
        self.batches_lr_update = batches_lr_update
        if stop_multiplier is None:
            self.stop_multiplier = -20 * self.mom / 3 + 10  # 4 if mom=0.9
            # 10 if mom=0
        else:
            self.stop_multiplier = stop_multiplier

    def on_train_begin(self, logs={}):
        p = self.params
        try:
            n_iterations = p['epochs'] * p['samples'] // p['batch_size']
        except:
            n_iterations = p['steps'] * p['epochs']

        self.learning_rates = np.geomspace(self.min_lr, self.max_lr, \
                                           num=n_iterations // self.batches_lr_update + 1)
        self.losses = []
        self.iteration = 0
        self.best_loss = 0
        if self.reload_weights:
            self.model.save_weights('tmp.hdf5')

    def on_batch_end(self, batch, logs={}):
        loss = logs.get('loss')

        if self.iteration != 0:  # Make loss smoother using momentum
            loss = self.losses[-1] * self.mom + loss * (1 - self.mom)

        if self.iteration == 0 or loss < self.best_loss:
            self.best_loss = loss

        if self.iteration % self.batches_lr_update == 0:  # Evaluate each lr over 5 epochs

            if self.reload_weights:
                self.model.load_weights('tmp.hdf5')

            lr = self.learning_rates[self.iteration // self.batches_lr_update]
            K.set_value(self.model.optimizer.lr, lr)

            self.losses.append(loss)

        if loss > self.best_loss * self.stop_multiplier:  # Stop criteria
            self.model.stop_training = True

        self.iteration += 1

    def on_train_end(self, logs=None):
        if self.reload_weights:
            self.model.load_weights('tmp.hdf5')

        plt.figure(figsize=(12, 6))
        plt.plot(self.learning_rates[:len(self.losses)], self.losses)
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.xscale('log')
        plt.show()