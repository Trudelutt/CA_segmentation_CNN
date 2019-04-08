import os
import json
import csv
import pandas as pd
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
#from batch_generator import generate_train_batches
from glob import glob
from os import mkdir
from os.path import join, basename
from tqdm import tqdm
from keras.utils import to_categorical
#from keras.preprocessing.image import ImageDataGenerator
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import nibabel as nib
#from augmentation import dataaug



def create_split(data_root_dir, label, splits=4):
    print("Create split")
    outdir= "split_lists"
    try:
        mkdir(outdir)
    except:
        pass
    for i in range(splits):
        training_list = fetch_training_data_ca_files(data_root_dir,label)
        training_val_list, test_list = train_test_split(training_list, test_size=0.1, random_state=i)
        #TODO change name
        new_training_list, val_list = train_test_split(training_val_list, test_size=0.1, random_state=i)
        print("trainfiles: " + str(len(new_training_list)) + " val: " + str(len(val_list)) + " test: " + str(len(test_list)))
        split_dir = join(outdir,label +'_' + str(i) + '_split_lists')
        try:
            mkdir(split_dir)
        except:
            print("Could not create foulder")

        with open(join(split_dir,'split_train.csv'), 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for sample in new_training_list:
                writer.writerow([x for x in sample])
        with open(join(split_dir,'split_val.csv'), 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for sample in val_list:
                writer.writerow([x for x in sample])
        with open(join(split_dir,'split_test.csv'), 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for sample in test_list:
                writer.writerow([x for x in sample])

def get_train_val_test(label, split_nr=0):
    train, val, test = [], [], []
    given_split_dir= join("split_lists", label + "_"+str(split_nr) + "_split_lists")
    print(given_split_dir)
    train = pd.read_csv(join(given_split_dir,'split_train.csv'), sep=',',header=None).values
    val = pd.read_csv(join(given_split_dir,'split_val.csv'), sep=',',header=None).values
    test = pd.read_csv(join(given_split_dir,'split_test.csv'), sep=',',header=None).values
    print("trainfiles: " + str(len(train)) + ", valfiles: " + str(len(val)) + ", testfiles: " + str(len(test)))
    return train, val, test


#Both LM and RCA
def make_both_label():
    path = glob("../st.Olav/*/*/*/")
    for i in xrange(len(path)):
        try:
            data_path = glob(path[i] + "*CCTA.nii.gz")[0]
            print(data_path)
            label_path = [glob(path[i] + "*LM.nii.gz")[0], glob(path[i] + "*RCA.nii.gz")[0]]
            sitk_label  = sitk.ReadImage(label_path[0], sitk.sitkFloat32)
            sitk_label  += sitk.ReadImage(label_path[1], sitk.sitkFloat32 )
            sitk.WriteImage(sitk_label, data_path.replace("CCTA", "both"))
        except:
            print("Could not make both file" + str(glob(path[i])))

def write_pridiction_to_file(label_array, prediction_array, tag, path="./predictions/prediction.nii.gz", label_path=None):
    meta_sitk = sitk.ReadImage(label_path)
    print(prediction_array.shape)
    sitk_image = sitk.GetImageFromArray(label_array)
    sitk_image.CopyInformation(meta_sitk)
    sitk.WriteImage(sitk_image, path.replace("nii", "gt.nii"))

    predsitk_image = sitk.GetImageFromArray(prediction_array)
    predsitk_image.CopyInformation(meta_sitk)
    sitk.WriteImage(predsitk_image, path)
    print("Writing prediction is done...")

def write_to_file(numpy_array, meta_path, path):
    print(path)
    meta_sitk = sitk.ReadImage(meta_path)
    sitk_image = sitk.GetImageFromArray(numpy_array[:meta_sitk.GetDepth()])
    sitk_image.CopyInformation(meta_sitk)
    sitk.WriteImage(sitk_image, path)

def write_all_labels(path):
    image = read_numpyarray_from_file(path+"LM.nii.gz")
    image += read_numpyarray_from_file(path+"Aorta.nii.gz")
    image += read_numpyarray_from_file(path+ "RCA.nii.gz")
    image[image == 2] = 1
    sitk_image = sitk.GetImageFromArray(image)
    sitk.WriteImage(sitk_image, "all_labels.nii.gz")


def preprosses_images(image, label_data):
    image -= np.min(image)
    image = image/ np.max(image)
    image -= np.mean(image)
    image = image / np.std(image)
    #image *= 255
    label = label_data.reshape((label_data.shape[0], label_data.shape[1], label_data.shape[2], 1))
    return image, label


def get_preprossed_numpy_arrays_from_file(image_path, label_path):
    sitk_image  = sitk.ReadImage(image_path, sitk.sitkFloat32)
    numpy_image = sitk.GetArrayFromImage(sitk_image)
    sitk_label  = sitk.ReadImage(label_path )
    numpy_label = sitk.GetArrayFromImage(sitk_label)
    if not np.array_equal(np.unique(numpy_label), np.array([0.,1.])):
        print("numpy is not binary mask")
        #numpy_label = numpy_label / np.max(numpy_label)
        #threshold = np.median(numpy_label)
        print("UNique values")
        print(np.unique(numpy_label))
        frangi_with_threshold = np.zeros(numpy_label.shape)
        frangi_with_threshold[np.where(numpy_label > 1)] = 1.0
        print("it is suposed to be binary now")
        print(np.unique(frangi_with_threshold))
        frangi_sitk = sitk.GetImageFromArray(frangi_with_threshold)
        frangi_sitk.CopyInformation(sitk_image)
        sitk.WriteImage(frangi_sitk, join('logs','frangi_test', image_path.split("/")[-1][:-7] + '_frangi_mask' + image_path[-7:]))
        return preprosses_images(frangi_with_threshold, numpy_label)


    return preprosses_images(numpy_image, numpy_label)


def remove_slices_with_just_background(image, label):
    first_non_backgroud_slice = float('inf')
    last_non_backgroud_slice = -1
    image_list = []
    label_list = []
    for i in xrange(image.shape[0]):
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

    #Channels must be an odd numbers
def add_neighbour_slides_training_data(image, label, stride=5, channels=5):
    padd = channels//2
    image_with_channels = np.zeros((image.shape[0], image.shape[1], image.shape[2], channels))
    zeros_image = np.zeros(image[0].shape)
    for i in range(image.shape[0]):
        if(i< padd * stride):
            count = padd * stride
            for channel in range(channels -(padd -i//stride)):
                #print(channels-channel-1, i + count)
                image_with_channels[i][...,channels - channel-1] = image[i+count]
                count -= stride

        elif i >= (image.shape[0]-(padd*stride)):
            count = - (padd * stride)
            for channel in range((image.shape[0]-i) +padd ):
                if (i + count) >= (image.shape[0]):
                    break
            #    print(channel, count+i)
                image_with_channels[i][...,channel] = image[i + count]
                count += stride
            #print("nFinished")
        else:
            count = - (padd * stride)
            for channel in range(channels):
                image_with_channels[i][...,channel] = image[i + count]
                count += stride
    return image_with_channels, label


def fetch_training_data_ca_files(data_root_dir,label="LM"):
    #path = glob("../st.Olav/*/*/*/")
    #path = glob("../../st.Olav/*/*/*/")
    if data_root_dir=="../st.Olav":
        data_root_dir += "/*/*/*/"
    path = glob(data_root_dir)
    training_data_files= list()
    for i in xrange(len(path)):
        try:
            data_path = glob(path[i] + "*CCTA.nii.gz")[0]
            label_path = glob(path[i] + "*" + label + ".nii.gz")[0]
        except IndexError:
            if label=="both" and i == 0:
                print("Makes both labels")
                make_both_label()
                label_path = glob(path[i] + "*" + label + ".nii.gz")[0]
            else:
                print("out of xrange for %s" %(path[i]))
        else:
            training_data_files.append(tuple([data_path, label_path]))
    return training_data_files


def get_train_and_label_numpy(number_of_slices, train_list, label_list, channels=5):
    train_data = np.zeros((number_of_slices, train_list[0].shape[1], train_list[0].shape[2], channels))
    label_data = np.zeros((number_of_slices, label_list[0].shape[1], label_list[0].shape[2], 1))
    index = 0
    for i in xrange(len(train_list)):
        with tqdm(total=train_list[i].shape[0], desc='Adds splice  from image ' + str(i+1) +"/" + str(len(train_list))) as t:
            for k in xrange(train_list[i].shape[0]):
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
def get_prediced_image_of_test_files(args,files, number, tag):
    element = files[number]
    print("Prediction on " + element[0])
    return get_slices(args,files[number:number+1], tag)



# Assume to have some sitk image (itk_image) and label (itk_label)
"""def get_data_files(data_root_dir, label="LM"):
    files = fetch_training_data_ca_files(data_root_dir,label)
    print("files: " + str(len(files)))
    return split_train_val_test(files)"""


def get_train_data_slices(args, train_files, tag = "LM"):
    traindata = []
    labeldata = []
    count_slices = 0
    for element in train_files:
        print(element[0])
        numpy_image, numpy_label = get_preprossed_numpy_arrays_from_file(element[0], element[1])
        i, l = add_neighbour_slides_training_data(numpy_image, numpy_label, stride= args.stride, channels=args.channels)
        resized_image, resized_label = remove_slices_with_just_background(i, l)

        count_slices += resized_image.shape[0]
        traindata.append(resized_image)
        labeldata.append(resized_label)
        #aug_img, mask = dataaug(resized_image, resized_label, intensityinterval= [0.8, 1.2], print_aug_images= True)
    train_data, label_data = get_train_and_label_numpy(count_slices, traindata, labeldata, channels=channels)

    print("min: " + str(np.min(train_data)) +", max: " + str(np.max(train_data)))
    #label = label_data.reshape((label_data.shape[0], label_data.shape[1], label_data.shape[2], 1))
    return train_data, label_data


def get_slices(args, files, tag="LM"):
    input_data_list = []
    label_data_list = []
    count_slices = 0
    for element in files:
        numpy_image, numpy_label = get_preprossed_numpy_arrays_from_file(element[0], element[1])
        numpy_image = np.float32(numpy_image)
        numpy_image -= np.mean(numpy_image)
        numpy_image = numpy_image / np.std(numpy_image)
        i, l = add_neighbour_slides_training_data(numpy_image, numpy_label,stride=args.stride, channels= args.channels)
        count_slices += i.shape[0]
        input_data_list.append(i)
        label_data_list.append(l)
    train_data, label_data = get_train_and_label_numpy(count_slices, input_data_list, label_data_list)

    print("min: " + str(np.min(train_data)) +", max: " + str(np.max(train_data)))
    #label = label_data.reshape((label_data.shape[0], label_data.shape[1], label_data.shape[2], 1))
    return train_data, label_data




def get_patches(image_numpy, label_numpy, remove_only_background_patches=False):
    image_patch_list = []
    label_patch_list = []
    orginal_shape = image_numpy.shape
    #print(orginal_shape)
    #print(orginal_shape)
    image_numpy_padded = np.zeros((orginal_shape[0] + (orginal_shape[0] % 64), orginal_shape[1] + (orginal_shape[1] % 64), orginal_shape[2] + (orginal_shape[2] % 64)))
    #print(image_numpy_padded.shape)
    image_numpy_padded[0:image_numpy.shape[0], 0:image_numpy.shape[1], 0:image_numpy.shape[2]] = image_numpy
    #print(image_numpy_padded.shape)
    mask_padded = np.zeros((image_numpy_padded.shape[0], image_numpy_padded.shape[1], image_numpy_padded.shape[2], 1))
    #print(mask_padded.shape)
    #print(label_numpy.shape)
    mask_padded[0:image_numpy.shape[0], 0:image_numpy.shape[1], 0:image_numpy.shape[2]] = label_numpy
    for z in xrange(64, image_numpy_padded.shape[0],64):
        for y in xrange(64, image_numpy_padded.shape[1],64):
            for x in xrange(64,image_numpy_padded.shape[2],64):
                mask_patch = mask_padded[z-64:z, y-64:y, x-64:x]
                if remove_only_background_patches:
                    if np.all(mask_patch == 0):
                        continue
                image_patch_list.append(image_numpy_padded[z-64:z, y-64:y, x-64:x])
                label_patch_list.append(mask_patch)
    return image_patch_list, label_patch_list, mask_padded.shape


def get_training_patches(train_files, label = "LM", remove_only_background_patches=False, return_shape=False):
    training_patches = []
    mask_patches = []
    count = 1
    with tqdm(total=len(train_files), desc='Adds patches  from image ' + str(count) +"/" + str(len(train_files))) as t:
        for element in train_files:
            #print(element[0])
            numpy_image, numpy_label = get_preprossed_numpy_arrays_from_file(element[0], element[1])
            img_patch, mask_patch, padded_shape  = get_patches(numpy_image, numpy_label, remove_only_background_patches)
            training_patches.extend(img_patch)
            mask_patches.extend(mask_patch)
            count += 1
            t.update()
    training_patch_numpy = np.array(training_patches)
    new_shape = (training_patch_numpy.shape[0],training_patch_numpy.shape[1],training_patch_numpy.shape[2],training_patch_numpy.shape[3], 1)
    new_shape_training_patch = training_patch_numpy.reshape(new_shape)
    if return_shape:
        return new_shape_training_patch, np.array(mask_patches).reshape(new_shape), padded_shape
    else:
        return new_shape_training_patch, np.array(mask_patches).reshape(new_shape)

def get_prediced_patches_of_test_file(test_files, i, label):
    element = test_files[i]
    print("Prediction on " + element[0])
    return get_training_patches(test_files[i:i+1], label, return_shape=True)

def from_patches_to_numpy(patches, shape):
    print(shape)
    reshape_patches = patches[...,0]
    print(reshape_patches.shape)
    image_numpy = np.zeros(shape[:-1])
    i = 0
    for z in xrange(64, shape[0],64):
        for y in xrange(64, shape[1],64):
            for x in xrange(64, shape[2],64):
                image_numpy[z-64:z, y-64:y, x-64:x] = reshape_patches[i]
                i += 1
    if(i != patches.shape[0]):
        print("something is wrong with the patches to numpy converting")
        print(i, patches.shape[0])
    return image_numpy








if __name__ == "__main__":
    create_split('../st.Olav', 'both')
    #get_data_files("../st.Olav", label="both")
    #train_files, val, test = get_train_val_test("both")
    #pred, lab = get_prediced_image_of_test_files(test, 0, "both")
    #img_slices, lab_slices = get_train_data_slices(train[:1])
    #print(len(fetch_training_data_ca_files("../st.Olav",label="both")))
    #get_training_patches(train_files[:2], label = "LM", remove_only_background_patches=False, return_shape=False)
    """print(test[0])
    x, y, orgshape = get_prediced_patches_of_test_file(test, 0, "both")
    label = from_patches_to_numpy(y, orgshape)
    img, org_label = get_preprossed_numpy_arrays_from_file(test[0][0], test[0][1])
    print(np.unique(np.equal(org_label, label[:org_label.shape[0]])))
    write_to_file(label, meta_path=test[0][0], path="./results/14.feb/" +str(basename(test[0][1])))"""
