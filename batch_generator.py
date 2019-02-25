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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

from keras.preprocessing.image import *
from preprossesing import *

"""def image_generator(files,label_file, batch_size = 64):

    while True:
          # Select files (paths/indices) for the batch
          batch_paths = np.random.choice(a = files,
                                         size = batch_size)
          batch_input = []
          batch_output = []

          # Read in each input, perform preprocessing and get labels
          for input_path in batch_paths:
              input = get_input(input_path )
              output = get_output(input_path,label_file=label_file )

              input = preprocess_input(image=input)
              batch_input += [ input ]
              batch_output += [ output ]
          # Return a tuple of (input,output) to feed the network
          batch_x = np.array( batch_input )
          batch_y = np.array( batch_output )

          yield( batch_x, batch_y )"""

def convert_data_to_numpy(img_name, no_masks=False, overwrite=False, train=False):
    print("Converting numpy")
    fname = basename(img_name[1])[:-7]
    numpy_path = 'np_files'
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
        numpy_image, numpy_label = get_preprossed_numpy_arrays_from_file(img_path, mask_path)
        img, mask = add_neighbour_slides_training_data(numpy_image, numpy_label)
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
def generate_train_batches(train_list, net_input_shape=(512,512,5), batchSize=1, numSlices=1, subSampAmt=-1,
                           stride=1, downSampAmt=1, shuff=0, aug_data=1):
    # Create placeholders for training
    #print("HEHE")
    #print(net_input_shape)
    img_batch = np.zeros((np.concatenate(((batchSize,), (512,512,5)))), dtype=np.float32)
    mask_batch = np.zeros((np.concatenate(((batchSize,), (512,512,1)))), dtype=np.uint8)
    #print("INSIDE train")
    while True:
        if shuff:
            shuffle(train_list)
        count = 0
        for i, scan_name in enumerate(train_list):
            try:
                scan_name = scan_name
                path_to_np = join('np_files',basename(scan_name[1])[:-7]+'.npz')
                with np.load(path_to_np) as data:
                    print(path_to_np)
                    train_img = data['img']
                    train_mask = data['mask']
            except:
                print('\nPre-made numpy array not found for {}.\nCreating now...'.format(join('np_files',basename(scan_name[1])[:-7]+'.npz')))
                train_img, train_mask = convert_data_to_numpy(scan_name, train=True)
                if np.array_equal(train_img,np.zeros(1)):
                    continue
                else:
                    print('\nFinished making npz file.')

            indicies = np.arange(train_img.shape[0])
            if shuff:
                shuffle(indicies)
            print(img_batch.shape)
            for j in indicies:
                #if not np.any(train_mask[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]):
                    #continue
                if img_batch.ndim == 4:
                    img_batch[count:,:,:] = train_img[j:j+1]
                    mask_batch[count:,:,:] = train_mask[j:j+1]

                else:
                    print('Error this function currently only supports 2D and 3D data.')
                    exit(0)

                count += 1
                if count % batchSize == 0:
                    count = 0
                    yield (img_batch, mask_batch)

@threadsafe_generator
def generate_val_batches(train_list, net_input_shape=(512,512,5), batchSize=1, numSlices=1, subSampAmt=-1,
                           stride=1, downSampAmt=1, shuff=0, aug_data=1):
    # Create placeholders for training
    #print("HEHE")
    #print(net_input_shape)
    img_batch = np.zeros((np.concatenate(((batchSize,), (512,512,5)))), dtype=np.float32)
    mask_batch = np.zeros((np.concatenate(((batchSize,), (512,512,1)))), dtype=np.uint8)
    #print("INSIDE train")
    while True:
        if shuff:
            shuffle(train_list)
        count = 0
        for i, scan_name in enumerate(train_list):
            try:
                scan_name = scan_name
                path_to_np = join('np_files',basename(scan_name[1])[:-7]+'.npz')
                with np.load(path_to_np) as data:
                    print(path_to_np)
                    train_img = data['img']
                    train_mask = data['mask']
            except:
                print('\nPre-made numpy array not found for {}.\nCreating now...'.format(join('np_files',basename(scan_name[1])[:-7]+'.npz')))
                train_img, train_mask = convert_data_to_numpy(scan_name, train=False)
                if np.array_equal(train_img,np.zeros(1)):
                    continue
                else:
                    print('\nFinished making npz file.')

            indicies = np.arange(train_img.shape[0])
            if shuff:
                shuffle(indicies)
            print(img_batch.shape)
            for j in indicies:
                #if not np.any(train_mask[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]):
                    #continue
                if img_batch.ndim == 4:
                    img_batch[count:,:,:] = train_img[j:j+1]
                    mask_batch[count:,:,:] = train_mask[j:j+1]

                else:
                    print('Error this function currently only supports 2D and 3D data.')
                    exit(0)

                count += 1
                if count % batchSize == 0:
                    count = 0
                    yield (img_batch, mask_batch)



"""class generate_train_batches(keras.utils.Sequence):
    #Generates data for Keras
    def __init__(self, train_list, batch_size=1, dim=(512,515), n_channels=5, shuffle=True):
        'Initialization'
        self.dim = dim
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.train_list = train_list
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        #'Denotes the number of batches per epoch'
        return int(np.floor(10000 / self.batch_size))

    def __getitem__(self,i):
        #'Generate one batch of data'
        # Generate indexes of the batch
        #indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        #train_list_temp = [self.train_list[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(self.train_list)

        return X, y

    def on_epoch_end(self):
        #'Updates indexes after each epoch'
        self.indexes = np.axrange(len(self.train_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        #'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        img_batch = np.zeros((np.concatenate(((self.batch_size,), (512,512,5)))), dtype=np.float32)
        mask_batch = np.zeros((np.concatenate(((self.batch_size,), (512,512,1)))), dtype=np.uint8)

        # Generate data
        # Create placeholders for training
        #print(list_IDs_temp)
        #print(self.indexes)

        #print("INSIDE train")
        count = 0
        for i, scan_name in enumerate(self.train_list):
            #scan_name = self.train_list[i]
            #print("SCanname")
            #print(scan_name)
            try:
                #scan_name = scan_name
                path_to_np = join('np_files',basename(scan_name[1])[:-7]+'.npz')
                with np.load(path_to_np) as data:
                    train_img = data['img']
                    train_mask = data['mask']
            except:
                #print("inside except")
                #print(scan_name)
                print('\nPre-made numpy array not found for {}.\nCreating now...'.format(join('np_files',basename(scan_name[1])[:-7]+'.npz')))
                train_img, train_mask = convert_data_to_numpy(scan_name, train=True)
                if np.array_equal(train_img,np.zeros(1)):
                    continue
                else:
                    print('\nFinished making npz file.')
            indicies = np.axrange(train_img.shape[0])
            #if shuff:
                #shuffle(indicies)
            #print(img_batch.shape)
            for j in indicies:
                #if not np.any(train_mask[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]):
                    #continue
                if img_batch.ndim == 4:
                    img_batch[count:,:,:] = train_img[j:j+1]
                    mask_batch[count:,:,:] = train_mask[j:j+1]
                else:
                    print('Error this function currently only supports 2D and 3D data.')
                    exit(0)

                count += 1
                if count % self.batch_size == 0:
                    count = 0
                    #print("RETURNS")
                    return (img_batch, mask_batch)

class generate_val_batches(keras.utils.Sequence):
        #Generates data for Keras
        def __init__(self, train_list, batch_size=1, dim=(512,515), n_channels=5, shuffle=True):
            'Initialization'
            self.dim = dim
            self.n_channels = n_channels
            self.batch_size = batch_size
            self.train_list = train_list
            self.shuffle = shuffle
            self.on_epoch_end()

        def __len__(self):
            #'Denotes the number of batches per epoch'
            return int(np.floor(500 / self.batch_size))

        def __getitem__(self,i):
            #'Generate one batch of data'
            # Generate indexes of the batch
            #indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

            # Find list of IDs
            #train_list_temp = [self.train_list[k] for k in indexes]

            # Generate data
            X, y = self.__data_generation(self.train_list)

            return X, y

        def on_epoch_end(self):
            #'Updates indexes after each epoch'
            self.indexes = np.axrange(len(self.train_list))
            if self.shuffle == True:
                np.random.shuffle(self.indexes)

        def __data_generation(self, list_IDs_temp):
            #'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
            # Initialization
            img_batch = np.zeros((np.concatenate(((self.batch_size,), (512,512,5)))), dtype=np.float32)
            mask_batch = np.zeros((np.concatenate(((self.batch_size,), (512,512,1)))), dtype=np.uint8)

            # Generate data
            # Create placeholders for training
            #print(list_IDs_temp)
            #print(self.indexes)

            #print("INSIDE train")
            count = 0
            for i, scan_name in enumerate(self.train_list):
                #scan_name = self.train_list[i]
                #print("SCanname")
                print(scan_name)
                try:
                    #scan_name = scan_name
                    path_to_np = join('np_files',basename(scan_name[1])[:-7]+'.npz')
                    with np.load(path_to_np) as data:
                        train_img = data['img']
                        train_mask = data['mask']
                except:
                    #print("inside except")
                    #print(scan_name)
                    print('\nPre-made numpy array not found for {}.\nCreating now...'.format(join('np_files',basename(scan_name[1])[:-7]+'.npz')))
                    train_img, train_mask = convert_data_to_numpy(scan_name, train=False)
                    if np.array_equal(train_img,np.zeros(1)):
                        continue
                    else:
                        print('\nFinished making npz file.')
                indicies = np.axrange(train_img.shape[0])
                #if shuff:
                    #shuffle(indicies)
                print("shape inside generator")
                print(train_img.shape)
                for j in indicies:
                    #if not np.any(train_mask[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]):
                        #continue
                    if img_batch.ndim == 4:
                        img_batch[count:,:,:] = train_img[j:j+1]
                        mask_batch[count:,:,:] = train_mask[j:j+1]
                    else:
                        print('Error this function currently only supports 2D and 3D data.')
                        exit(0)

                    count += 1
                    if count % self.batch_size == 0:
                        count = 0
                        #print("RETURNS")
                        return (img_batch, mask_batch)"""

if __name__ == "__main__":
    train, val, test = get_train_val_test("both")
    #pred, lab = get_prediced_image_of_test_files(test, 0, "both")
    img_slices, lab_slices = get_train_data_slices(train[2:4])
    print(img_slices.shape)

    traingen= generate_train_batches(train[2:4],net_input_shape=(512,512,5), batchSize=4)
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
    print(np.unique(lab_slices[0]))
    #print(lab_slices[0].shape)
    #print(np.unique(np.equal(lab_slices[0], lab_slices[1])))
    #for i in xrange(lab_slices.shape[0]):
        #print(np.array_equal(lab_slices[i], traingen.next()[1]))
    #l= traingen.__next__()[1]
    print("iteration")
    #print(np.unique(np.equal(l, lab_slices[:1])))
    for i in xrange(0,lab_slices.shape[0] - lab_slices.shape[0]%4,4):
    #for i in xrange(lab_slices.shape[0]):
        print(i)

        l= traingen.next()[1]
        print(l.shape)
        print(np.unique(l))
        #print(l.shape)
        print(np.unique(np.equal(l, lab_slices[i:i+4])))
