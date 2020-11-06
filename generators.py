from data import *
import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
import matplotlib.pyplot as plt
import os
import random
from keras.preprocessing.image import ImageDataGenerator

class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, list_IDs, image_path,mask_path,
                 to_fit=True, batch_size=32, dim=(512, 512), dimy=(512, 512),
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
        self.dimy = dimy
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
        y = np.empty((self.batch_size, *self.dimy,self.n_channels))

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
            ha = Image.fromarray((i[0][0, :, :, 0] * 255).astype('uint8'))
            ha = ha.convert("RGBA")
            r, g, b, a = ha.split()
            g = b.point(lambda i: i * 0)
            # Recombine back to RGB image
            ha = Image.merge('RGBA', (r, g, b, a))
            ha.save("ha.png")
            plt.show()
            plt.imshow((i[1][0, :, :, 0]), cmap=plt.cm.bone)
            saveSitkPng(i[1][0, :, :, 0], os.path.join(save_path, "test.png"))
            plt.show()

class DataGenerator2(DataGenerator):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """

    def __init__(self, list_IDs, image_path, mask_path,
                 to_fit=True, batch_size=32, dim=(512, 512), dimy=(512, 512),
                 n_channels=1, n_classes=10, shuffle=True,data_gen_args=None):
        super().__init__(list_IDs, image_path, mask_path,
                 to_fit=to_fit, batch_size=batch_size, dim=dim, dimy=dimy,
                 n_channels=n_channels, n_classes=n_classes, shuffle=shuffle)
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
        self.bool = False
        if data_gen_args !=None:
            self.trans = ImageDataGenerator(**data_gen_args)

    def _generate_X(self, list_IDs_temp):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        self.param = self.trans.get_random_transform(self.dim)

        if random.uniform(0,1) >= 0.5:
            self.bool = True
        else:
            self.bool = False
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = self._load_dicom_image(self.image_path + '/' + ID)
            # X[i,] = self.apply_transform(X[i,],self.get_random_transform((1,512,512)))
            if self.bool:
                X[i,] = self.trans.apply_transform(X[i,], self.param)


        # X=np.expand_dims(X, 4)

        return X

    def _generate_y(self, list_IDs_temp):
        """Generates data containing batch_size masks
        :param list_IDs_temp: list of label ids to load
        :return: batch if masks
        """
        y = np.empty((self.batch_size, *self.dimy, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            y[i,] = self._load_grayscale_image_VTK(self.mask_path + '/' + 'label_' + ID[6:15] + '.png')
            if self.bool:
                y[i,] = self.trans.apply_transform(y[i,], self.param)

        return y

class generator3d(DataGenerator):

    def __init__(self, list_IDs, image_path, mask_path,
                 to_fit=True, batch_size=32, patch_size=8, dim=(512, 512), dimy=(512, 512),
                 n_channels=1, n_classes=10, shuffle=True):
        super().__init__(list_IDs, image_path, mask_path,
                     to_fit=to_fit, batch_size=batch_size,  dim=dim, dimy=dimy,
                     n_channels=n_channels, n_classes=n_classes, shuffle=shuffle)
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
        self.patch_size = patch_size
        self.number_of_patches = 0
        self.on_epoch_end()
        self.n = 0
        self.max = self.__len__()
        slices = os.listdir(os.path.join(self.image_path, self.list_IDs[0]))
        self.number_of_patches = int(np.floor((len(slices) / self.patch_size)))
        self.patientIDs = list_IDs
        self.list_IDs = []

        count = 0

        for i, ID in enumerate(self.patientIDs):
            slices = os.listdir(os.path.join(self.image_path, ID))
            while count < self.number_of_patches:
                patch = slices[(count * self.patch_size):((count + 1) * self.patch_size)]
                self.list_IDs.append([ID, patch])
                count += 1
            count = 0
        self.indexes = np.arange(len(self.list_IDs))

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        # return int(np.floor(len(self.list_IDs) / self.batch_size) * self.number_of_patches)
        return int(np.floor(len(self.list_IDs)))

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

    def _generate_y(self, list_IDs_temp):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        Y = np.zeros((self.batch_size, *self.dimy, self.patch_size, self.n_channels))
        # Generate data

        for patch in list_IDs_temp:
            for i, ID in enumerate(patch[1]):
                path = self.mask_path + '/' + patch[0] + '/' + 'label_' + ID[6:15] + '.png'
                img = self._load_grayscale_image_VTK(path)[:, :, 0]
                Y[0, :, :, i, 0] = img
        return Y

    def _generate_X(self, list_IDs_temp):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        X = np.zeros((self.batch_size, *self.dim, self.patch_size, self.n_channels))
        # Generate data

        for patch in list_IDs_temp:
            for i, ID in enumerate(patch[1]):
                path = self.image_path + '/' + patch[0] + '/' + ID
                img = self._load_grayscale_image_VTK(path)[:, :, 0]
                X[0, :, :, i, 0] = img

        return X

    def __next__(self):
        if self.n >= self.max:
            self.n = 0
        result = self.__getitem__(self.n)
        self.n += 1
        return result

    def plotFromGenerator3d(self):
        count = 0
        for i in self:
            for batch in range(0, np.shape(i[0])[0]):
                for k in range(0, np.shape(i[0])[3]):
                    background = i[0][batch, :, :, k, 0]
                    background = background / np.max(background)
                    background = (background * 255).astype('uint8')
                    background = Image.fromarray(background)
                    background = background.convert("RGBA")
                    img = i[1][batch, :, :, k, 0]
                    overlay = Image.fromarray((img * 255).astype('uint8'))
                    overlay = overlay.convert("RGBA")

                    # Split into 3 channels
                    r, g, b, a = overlay.split()

                    # Increase Reds
                    g = b.point(lambda i: i * 0)

                    # Recombine back to RGB image
                    overlay = Image.merge('RGBA', (r, g, b, a))
                    new_img = Image.blend(background, overlay, 0.3)
                    new_img.save(str(count) + ".png", "PNG")
                    count += 1

class generator3da(generator3d):

    def __init__(self, list_IDs, image_path, mask_path,
                 to_fit=True, batch_size=32, patch_size=8, dim=(512, 512),dimy=(512,512),
                 n_channels=1, n_classes=10, shuffle=True, data_gen_args=None):
        super().__init__(list_IDs, image_path, mask_path,
                 to_fit=to_fit, batch_size=batch_size, patch_size=patch_size, dim=dim,dimy=dimy,
                 n_channels=n_channels, n_classes=n_classes, shuffle=shuffle)
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param image_path: path to images location
        :param mask_path: path to masks location
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension of the input
        :param dimy: tuple indicating image dimension of the output
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """
        if data_gen_args !=None:
            self.trans = ImageDataGenerator(**data_gen_args)


    def _generate_y(self, list_IDs_temp):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        Y = np.zeros((self.batch_size, *self.dimy, self.patch_size,self.n_channels))
        # Generate data

        for patch in list_IDs_temp:
            for i, ID in enumerate(patch[1]):
                path= self.mask_path + '/' + patch[0] + '/' + 'label_' + ID[6:15] + '.png'
                img = self._load_grayscale_image_VTK(path)[:, :, 0]
                Y[0, :, :, i, 0] = img
                if self.bool:
                    Y[0,:,:,i,:] = self.trans.apply_transform(Y[0,:,:,i,:], self.param)
        return Y

    def _generate_X(self, list_IDs_temp):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        X = np.zeros((self.batch_size, *self.dim, self.patch_size,self.n_channels))
        self.param = self.trans.get_random_transform(self.dim)

        if random.uniform(0,1) >= 0.5:
            self.bool = True
        else:
            self.bool = False
        # Generate data

        for patch in list_IDs_temp:
            for i, ID in enumerate(patch[1]):
                path= self.image_path + '/' + patch[0] + '/' + ID
                img = self._load_grayscale_image_VTK(path)[:, :, 0]
                X[0, :, :, i, 0] = img
                if self.bool:
                    X[0,:,:,i,:] = self.trans.apply_transform(X[0,:,:,i,:], self.param)
        return X

