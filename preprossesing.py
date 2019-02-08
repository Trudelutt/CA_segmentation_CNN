import os
import json
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from skimage.transform import resize
import nibabel as nib



def split_train_val_test(samples):
    n_samples = len(samples)
    sep_train = int(n_samples*0.8)
    sep_val = int(n_samples*0.9)

    train_files = samples[:sep_train]
    val_files = samples[sep_train:sep_val]

    test_files = samples[sep_val:]
    print(len(train_files), len(val_files), len(test_files))
    return train_files, val_files, test_files


def preprosses_images(image, label, tag):
    image -= np.min(image)
    image = image/ np.max(image)
    image -= np.mean(image)
    image = image / np.std(image)
    return image, label



def get_preprossed_numpy_arrays_from_file(image_path, label_path, tag):
    sitk_image  = sitk.ReadImage(image_path, sitk.sitkFloat32)
    numpy_image = sitk.GetArrayFromImage(sitk_image)
    if len(label_path) == 2:
        sitk_label  = sitk.ReadImage(label_path[0], sitk.sitkFloat32)
        numpy_label = sitk.GetArrayFromImage(sitk_label)
        sitk_label  = sitk.ReadImage(label_path[1], sitk.sitkFloat32 )
        numpy_label += sitk.GetArrayFromImage(sitk_label)
        #write_pridiction_to_file(numpy_image, numpy_label, tag="both", path="./predictions/prediction.nii.gz", label_path=image_path)

    else:
        sitk_label  = sitk.ReadImage(label_path )
        numpy_label = sitk.GetArrayFromImage(sitk_label)

    return preprosses_images(numpy_image, numpy_label, tag)

def remove_slices_with_just_background(image, label):
    first_non_backgroud_slice = float('inf')
    last_non_backgroud_slice = -1
    image_list = []
    label_list = []
    for i in range(image.shape[0]):
        if(1 in label[i]):
            if(i < first_non_backgroud_slice):
                first_non_backgroud_slice = i
            last_non_backgroud_slice = i
    if(first_non_backgroud_slice-2 < 0):
        resize_label =  label[first_non_backgroud_slice-1:last_non_backgroud_slice + 1]
        resize_image =  image[first_non_backgroud_slice-1:last_non_backgroud_slice+1]
    elif(first_non_backgroud_slice-1 < 0):
        resize_label =  label[first_non_backgroud_slice:last_non_backgroud_slice + 1]
        resize_image =  image[first_non_backgroud_slice:last_non_backgroud_slice+1]
    else:
        resize_label =  label[first_non_backgroud_slice-2:last_non_backgroud_slice + 1]
        resize_image =  image[first_non_backgroud_slice-2:last_non_backgroud_slice+1]


    return resize_image, resize_label


def add_neighbour_slides_training_data(image, label):
    image_with_channels = np.zeros((image.shape[0], image.shape[1], image.shape[2], 5))
    zeros_image = np.zeros(image[0].shape)
    for i in range(image.shape[0]):
        if(i == 0):
            image_with_channels[i][...,0] = zeros_image
            image_with_channels[i][...,1] = zeros_image
            image_with_channels[i][...,2] = image[i]
            image_with_channels[i][...,3] = image[i+1]
            image_with_channels[i][...,4] = image[i+2]
        elif(i == 1):
            image_with_channels[i][...,0] = zeros_image
            image_with_channels[i][...,1] = image[i-1]
            image_with_channels[i][...,2] = image[i]
            image_with_channels[i][...,3] = image[i+1]
            image_with_channels[i][...,4] = image[i+2]
        elif(i == image.shape[0]-2):
            image_with_channels[i][...,0] = image[i-2]
            image_with_channels[i][...,1] = image[i-1]
            image_with_channels[i][...,2] = image[i]
            image_with_channels[i][...,3] = image[i+1]
            image_with_channels[i][...,4] = zeros_image
        elif(i == image.shape[0]-1):
            image_with_channels[i][...,0] = image[i-2]
            image_with_channels[i][...,1] = image[i-1]
            image_with_channels[i][...,2] = image[i]
            image_with_channels[i][...,3] = zeros_image
            image_with_channels[i][...,4] = zeros_image
        else:
            image_with_channels[i][...,0] = image[i-2]
            image_with_channels[i][...,1] = image[i-1]
            image_with_channels[i][...,2] = image[i]
            image_with_channels[i][...,3] = image[i+1]
            image_with_channels[i][...,4] = image[i+2]
    return image_with_channels, label


    #TODO check if channels becomes right for training 0. 1. and the last ones
    """if np.array_equal(image_with_channels[20][...,0],image[20]):
        print("HURRA channel 0 er riktig")
    if np.array_equal(image_with_channels[20][...,1], image[21]):
        print("HURRA channel 1 er riktig")
    if np.array_equal(image_with_channels[20][...,2], image[22]):
        print("HURRA channel 2 er riktig")
    if np.array_equal(image_with_channels[20][...,3], image[23]):
        print("HURRA channel 3 er riktig")
    if np.array_equal(image_with_channels[20][...,4], image[24]):
        print("HURRA channel 4 er riktig")"""

    return image_with_channels, label



def fetch_training_data_ca_files(data_root_dir,label="LM"):
    #path = glob("../st.Olav/*/*/*/")
    #path = glob("../../st.Olav/*/*/*/")
    if data_root_dir=="../st.Olav":
        data_root_dir += "/*/*/*/"
    path = glob(data_root_dir)
    training_data_files = list()
    for i in range(len(path)):
        try:
            data_path = glob(path[i] + "*CCTA.nii.gz")[0]
            if(label =="both"):
                label_path = [glob(path[i] + "*LM.nii.gz")[0], glob(path[i] + "*RCA.nii.gz")[0]]
            else:
                label_path = glob(path[i] + "*" + label + ".nii.gz")[0]
        except IndexError:
            print("out of range for %s" %(path[i]))
        else:
            training_data_files.append(tuple([data_path, label_path]))
    return training_data_files


def get_train_and_label_numpy(number_of_slices, train_list, label_list):
    train_data = np.zeros((number_of_slices, train_list[0].shape[1], train_list[0].shape[2], 5))
    label_data = np.zeros((number_of_slices, label_list[0].shape[1], label_list[0].shape[2]))
    index = 0
    for i in range(len(train_list)):
        with tqdm(total=train_list[i].shape[0], desc='Adds splice  from image ' + str(i+1) +"/" + str(len(train_list))) as t:
            for k in range(train_list[i].shape[0]):
                train_data[index] = train_list[i][k]
                label_data[index] = label_list[i][k]
                index += 1
                t.update()

    return train_data, label_data


def read_numpyarray_from_file(path):
    image = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(image).astype('float32')

def show_nii_image(path, slice_nr):
    image = read_numpyarray_from_file(path)
    plt.figure()
    plt.imshow(image[slice_nr])

#TODO make sure that index not out of bounds
def get_prediced_image_of_test_files(files, number, tag):
    element = files[number]
    print("Prediction on " + element[0])
    return get_slices(files[number:number+1], tag)


def write_pridiction_to_file(label_array, prediction_array, tag, path="./predictions/prediction.nii.gz", label_path=None):
    meta_sitk = sitk.ReadImage(label_path)
    print(prediction_array.shape)



    sitk_image = sitk.GetImageFromArray(label_array)
    sitk_image.CopyInformation(meta_sitk)
    sitk.WriteImage(sitk_image, path.replace("prediction.nii", "gt.nii"))

    predsitk_image = sitk.GetImageFromArray(prediction_array)
    predsitk_image.CopyInformation(meta_sitk)
    sitk.WriteImage(predsitk_image, path)
    print("Writing prediction is done...")


# Assume to have some sitk image (itk_image) and label (itk_label)
def get_data_files(data_root_dir, label="LM"):
    files = fetch_training_data_ca_files(data_root_dir,label)
    print("files: " + str(len(files)))
    return split_train_val_test(files)


def get_train_data_slices(train_files, tag = "LM"):
    traindata = []
    labeldata = []
    count_slices = 0
    for element in train_files:
        print(element[0])
        numpy_image, numpy_label = get_preprossed_numpy_arrays_from_file(element[0], element[1], tag)
        i, l = add_neighbour_slides_training_data(numpy_image, numpy_label)
        resized_image, resized_label = remove_slices_with_just_background(i, l)

        count_slices += resized_image.shape[0]
        traindata.append(resized_image)
        labeldata.append(resized_label)
    train_data, label_data = get_train_and_label_numpy(count_slices, traindata, labeldata)

    print("min: " + str(np.min(train_data)) +", max: " + str(np.max(train_data)))
    label = label_data.reshape((label_data.shape[0], label_data.shape[1], label_data.shape[2], 1))
    return train_data, label


def get_slices(files, tag="LM"):
    input_data_list = []
    label_data_list = []
    count_slices = 0
    for element in files:
        numpy_image, numpy_label = get_preprossed_numpy_arrays_from_file(element[0], element[1],tag)
        i, l = add_neighbour_slides_training_data(numpy_image, numpy_label)
        count_slices += i.shape[0]
        input_data_list.append(i)
        label_data_list.append(l)
    train_data, label_data = get_train_and_label_numpy(count_slices, input_data_list, label_data_list)

    print("min: " + str(np.min(train_data)) +", max: " + str(np.max(train_data)))
    label = label_data.reshape((label_data.shape[0], label_data.shape[1], label_data.shape[2], 1))
    return train_data, label


def write_all_labels(path):
    image = read_numpyarray_from_file(path+"LM.nii.gz")
    image += read_numpyarray_from_file(path+"Aorta.nii.gz")
    image += read_numpyarray_from_file(path+ "RCA.nii.gz")
    image[image == 2] = 1
    sitk_image = sitk.GetImageFromArray(image)
    sitk.WriteImage(sitk_image, "all_labels.nii.gz")




if __name__ == "__main__":
    train_files, val_files, test_files = get_data_files( label="LM")
    #for i in range(len(train_files)):
    """n= len(test_files)
    print(test_files)
    test_x, test_y = get_prediced_image_of_test_files(test_files, 0, tag="LM")
    train_data, label_data = get_train_data_slices(train_files[:1], tag ="LM")
    write_pridiction_to_file(test_y, label_data, tag="HV", path="./predictions/prediction.nii.gz", label_path=test_files[0][1])"""
