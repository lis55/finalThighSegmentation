from data import *
import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
import matplotlib.pyplot as plt
import os


class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, list_IDs, image_path,mask_path,
                 to_fit=True, batch_size=32, dim=(512, 512),
                 n_channels=1, n_classes=10, shuffle=True):
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param image_path: path to images location
        :param mask_path: path to masks location
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.list_IDs = list_IDs
        self.image_path = image_path
        self.mask_path = mask_path
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.n = 0
        self.max = self.__len__()
    def __next__(self):
        if self.n >= self.max:
            self.n = 0
        result = self.__getitem__(self.n)
        self.n += 1
        return result

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs

        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self._generate_X(list_IDs_temp)

        if self.to_fit:
            y = self._generate_y(list_IDs_temp)
            return X, y
        else:
            return X

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _generate_X(self, list_IDs_temp):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        X = np.empty((self.batch_size, *self.dim,self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = self._load_dicom_image(self.image_path + '/'+ ID)
        #X=np.expand_dims(X, 4)

        return X

    def _generate_y(self, list_IDs_temp):
        """Generates data containing batch_size masks
        :param list_IDs_temp: list of label ids to load
        :return: batch if masks
        """
        y = np.empty((self.batch_size, *self.dim,self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            #y[i,] = self._load_grayscale_image_VTK(self.mask_path + '/'+'label_'+ID[6:15]+'.png')
            y[i,] = self._load_grayscale_image_VTK(self.mask_path + '/' + 'label_' + ID[6:15] + '.png')

        return y

    def _load_grayscale_image(self, image_path):
        """Load grayscale image
        :param image_path: path to image to load
        :return: loaded image
        """
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = img / 255
        '''
        img2 = img.astype(np.float32)
        # --- the following holds the square root of the sum of squares of the image dimensions ---
        # --- this is done so that the entire width/height of the original image is used to express the complete circular range of the resulting polar image ---
        value = np.sqrt(((img2.shape[0] / 2.0) ** 2.0) + ((img2.shape[1] / 2.0) ** 2.0))
        polar_image = cv2.linearPolar(img2, (img2.shape[0] / 2, img2.shape[1] / 2), value, cv2.WARP_FILL_OUTLIERS)
        polar_image = polar_image.astype(np.uint8)
        img=polar_image
        '''
        img = np.expand_dims(img, axis=2)
        img=img.astype(np.float32)

        return img

    def _load_dicom_image(self, image_path):
        """Load grayscale image
        :param image_path: path to image to load
        :return: loaded image
        """
        img = load_dicom(image_path)
        img = img / np.max(img)

        #self.polar(img)

        return img

    def _load_grayscale_image_VTK(self, image_path):
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

    def polar(self, img):
        img2 = img.astype(np.float32)

        # --- the following holds the square root of the sum of squares of the image dimensions ---
        # --- this is done so that the entire width/height of the original image is used to express the complete circular range of the resulting polar image ---
        value = np.sqrt(((img2.shape[0] / 2.0) ** 2.0) + ((img2.shape[1] / 2.0) ** 2.0))

        polar_image = cv2.linearPolar(img2, (img2.shape[0] / 2, img2.shape[1] / 2), value, cv2.WARP_FILL_OUTLIERS)

        # polar_image = polar_image.astype(np.uint8)
        img = polar_image
        img = np.expand_dims(img, axis=2)

        return img

    def downsample(self,dim,path):
        ids = os.listdir(self.mask_path)
        for i in ids:
            img = self._load_grayscale_image_VTK(self.mask_path + '/' + i)[:, :, 0]
            img = img*255
            #img = img.astype(np.int8)
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(path+'/'+'masks'+'/'+i, img)
        ids=os.listdir(self.image_path)
        for i in ids:
            img = self._load_dicom_image(self.image_path+'/'+i)[:,:,0]
            #img = cv2.resize(img,dim,interpolation=cv2.INTER_AREA)
            img = (img*255).astype(np.int16)
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(path+'/'+'images'+'/'+i[:-3]+'png', img)

    def plotFromGenerator(self,save_path=''):
        "Plots frames and masks, only works if batch size is 1"
        for i in self:
            plt.imshow((i[0][0, :, :, 0]), cmap=plt.cm.bone)
            saveSitkPng(i[0][0, :, :, 0],os.path.join(save_path, "test.png"))
            plt.show()
            plt.imshow((i[1][0, :, :, 0]), cmap=plt.cm.bone)
            saveSitkPng(i[1][0, :, :, 0], os.path.join(save_path, "test.png"))
            plt.show()