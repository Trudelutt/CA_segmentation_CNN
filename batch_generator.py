from __future__ import print_function

import threading
from os.path import join, basename
from os import mkdir
from glob import glob
import csv
from sklearn.model_selection import KFold
import numpy as np
from numpy.random import rand, shuffle
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import keras
from augmentation import augmentImages
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.ioff()

from keras.preprocessing.image import *
from preprossesing import *


def convert_data_to_numpy(args, img_name, no_masks=False, overwrite=False, train=False):
    print("Converting numpy")
    fname = basename(img_name[1])[:-7]
    numpy_path = join('np_files', "numpy_3D") if args.model == 'BVNet3D' else join('np_files', "numpy_2D_channels" + str(args.channels) + "_stride" + str(args.stride))
    print(numpy_path)
    img_path = img_name[0]
    mask_path = img_name[1]
    try:
        mkdir(numpy_path)
    except:
        pass

    ct_min = -1024
    ct_max = 3072

    if not overwrite:
        try:
            with np.load(join(numpy_path, fname + '.npz')) as data:
                return data['img'], data['mask']
        except:
            print("Something went wrong")
            print(join(numpy_path, fname + '.npz'))
            pass

    try:
        if args.model =="BVNet3D":
            img, mask = get_training_patches([[img_path, mask_path]], args.label, remove_only_background_patches=train)
        else:
            numpy_image, numpy_label = get_preprossed_numpy_arrays_from_file(img_path, mask_path)
            img, mask = add_neighbour_slides_training_data(numpy_image, numpy_label, args.stride, args.channels)
            if train:
                img, mask = remove_slices_with_just_background(img, mask)

        if not no_masks:
            np.savez_compressed(join(numpy_path, fname + '.npz'), img=img, mask=mask)
        else:
            np.savez_compressed(join(numpy_path, fname + '.npz'), img=img)

        if not no_masks:
            return img, mask
        else:
            return img

    except Exception as e:
        print('\n'+'-'*100)
        print('Unable to load img or masks for {}'.format(fname))
        print(e)
        print('Skipping file')
        print('-'*100+'\n')

        return np.zeros(1), np.zeros(1)


''' Make the generators threadsafe in case of multiple threads '''
class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()





def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def generate_train_batches(args,train_list, net_input_shape=(512,512,5), batchSize=1, numSlices=1, subSampAmt=-1,
                           stride=1, downSampAmt=1, shuff=1, aug_data=0):
    # Create placeholders for training and numpy path
    if args.model =="BVNet3D":
        numpy_path = join('np_files', "numpy_3D")
        img_batch = np.zeros((np.concatenate(((batchSize,), (64,64,64,args.channels)))), dtype=np.float32)
        mask_batch = np.zeros((np.concatenate(((batchSize,), (64,64,64,1)))), dtype=np.uint8)
        print("MAKE PLACEHOLDERS")

    else:
        numpy_path = join('np_files', "numpy_2D_channels" + str(args.channels) + "_stride" + str(args.stride))
        img_batch = np.zeros((np.concatenate(((batchSize,), (512,512,args.channels)))), dtype=np.float32)
        mask_batch = np.zeros((np.concatenate(((batchSize,), (512,512,1)))), dtype=np.uint8)
    #print("INSIDE train")
    while True:
        if shuff:
            shuffle(train_list)
        count = 0
        for i, scan_name in enumerate(train_list):
            try:
                path_to_np = join(numpy_path,basename(scan_name[1])[:-7]+'.npz')
                with np.load(path_to_np) as data:
                    print(path_to_np)
                    train_img = data['img']
                    train_mask = data['mask']
            except:
                print('\nPre-made numpy array not found for {}.\nCreating now...'.format(join(numpy_path,basename(scan_name[1])[:-7]+'.npz')))
                train_img, train_mask = convert_data_to_numpy(args, scan_name, train=True)
                if np.array_equal(train_img,np.zeros(1)):
                    continue
                else:
                    print('\nFinished making npz file.')
            indicies = np.arange(train_img.shape[0])
            if shuff:
                shuffle(indicies)
            #print(img_batch.shape)
            for j in indicies:
                #if not np.any(train_mask[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]):
                    #continue
                if img_batch.ndim == 4:

                    img_batch[count:,:,:] = train_img[j:j+1]
                    mask_batch[count:,:,:] = train_mask[j:j+1]

                if img_batch.ndim == 5:
                    img_batch[count:,:,:,:] = train_img[j:j+1]
                    mask_batch[count:,:,:,:] = train_mask[j:j+1]

                else:
                    print(img_batch.ndim)
                    print('Error this function currently only supports 2D and 3D data.')
                    exit(0)

                count += 1
                if count % batchSize == 0:
                    count = 0
                    if aug_data:
                        img_batch, mask_batch = augmentImages(img_batch, mask_batch, debugg= False)
                    yield (img_batch, mask_batch)

@threadsafe_generator
def generate_val_batches(args, train_list, net_input_shape=(512,512,5), batchSize=1, numSlices=1, subSampAmt=-1,
                           stride=1, downSampAmt=1, shuff=1, aug_data=0):
    if args.model == 'BVNet3D':
        numpy_path = join('np_files', "numpy_3D")
        img_batch = np.zeros((np.concatenate(((batchSize,), (64,64,64,args.channels)))), dtype=np.float32)
        mask_batch = np.zeros((np.concatenate(((batchSize,), (64,64,64,1)))), dtype=np.uint8)

    else:
        numpy_path = join('np_files', "numpy_2D_channels" + str(args.channels) + "_stride" + str(args.stride))
        img_batch = np.zeros((np.concatenate(((batchSize,), (512,512,args.channels)))), dtype=np.float32)
        mask_batch = np.zeros((np.concatenate(((batchSize,), (512,512,1)))), dtype=np.uint8)
    while True:
        if shuff:
            shuffle(train_list)
        count = 0
        for i, scan_name in enumerate(train_list):
            try:
                scan_name = scan_name
                path_to_np = join(numpy_path,basename(scan_name[1])[:-7]+'.npz')
                with np.load(path_to_np) as data:
                    print(path_to_np)
                    train_img = data['img']
                    train_mask = data['mask']
            except:
                print('\nPre-made numpy array not found for {}.\nCreating now...'.format(join(numpy_path,basename(scan_name[1])[:-7]+'.npz')))
                train_img, train_mask = convert_data_to_numpy(args,scan_name, train=False)
                if np.array_equal(train_img,np.zeros(1)):
                    continue
                else:
                    print('\nFinished making npz file.')

            indicies = np.arange(train_img.shape[0])
            if shuff:
                shuffle(indicies)
            for j in indicies:
                #if not np.any(train_mask[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]):
                    #continue
                if img_batch.ndim == 4:
                    img_batch[count:,:,:] = train_img[j:j+1]
                    mask_batch[count:,:,:] = train_mask[j:j+1]
                if img_batch.ndim == 5:
                    img_batch[count:,:,:,:] = train_img[j:j+1]
                    mask_batch[count:,:,:,:] = train_mask[j:j+1]
                else:
                    print('Error this function currently only supports 2D and 3D data.')
                    exit(0)

                count += 1
                if count % batchSize == 0:
                    count = 0
                    yield (img_batch, mask_batch)





if __name__ == "__main__":
    train, val, test = get_train_val_test("both")
    #pred, lab = get_prediced_image_of_test_files(test, 0, "both")
    #img_slices, lab_slices = get_train_data_slices(train[2:4])
    #print(img_slices.shape)

    traingen= generate_train_batches(args,train[2:3],net_input_shape=(512,512,5), batchSize=32)
    i, l = traingen.next()
    count= 0
    #for img in i:
        #scipy.misc.imsave("./logs/aug_image_" + str(count)  +".png", img[...,2])
        #count += 1
    #traingen.next()
    #i, l = traingen.next()
    """print("###")
    img_1, lab_1 = traingen.next()

    print("###")
    img_2, lab_2 = traingen.next()
    print("AFTER 2")"""
    """print(i.shape)
    print(np.unique(np.equal(i[0], img_slices[0])))
    print(np.unique(np.equal(l[0], lab_slices[0])))
    print(np.unique(l[0]))"""
    #print(np.unique(lab_slices[0]))
    #print(lab_slices[0].shape)
    #print(np.unique(np.equal(lab_slices[0], lab_slices[1])))
    #for i in xrange(lab_slices.shape[0]):
        #print(np.array_equal(lab_slices[i], traingen.next()[1]))
    #l= traingen.__next__()[1]
    """print("iteration")
    #print(np.unique(np.equal(l, lab_slices[:1])))
    for i in xrange(0,lab_slices.shape[0] - lab_slices.shape[0]%4,4):
    #for i in xrange(lab_slices.shape[0]):
        print(i)

        l= traingen.next()[1]
        print(l.shape)
        print(np.unique(l))
        #print(l.shape)
        print(np.unique(np.equal(l, lab_slices[i:i+4])))"""
