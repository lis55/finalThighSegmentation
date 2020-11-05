import vtk
from vtk.util import numpy_support
import numpy as np
import SimpleITK as sitk
import cv2


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
    arrayData = np.transpose(arrayData)
    # Reshape the NumPy array to 3D using 'ConstPixelDims' as a 'shape'
    ArrayDicom = ArrayDicom.reshape(ConstPixelDims, order='F')


    return ArrayDicom

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