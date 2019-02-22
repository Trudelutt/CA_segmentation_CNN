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
    fig_path = 'figs'
    try:
        mkdir(numpy_path)
    except:
        pass
    try:
        mkdir(fig_path)
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


        """if not no_masks:
            itk_mask = sitk.ReadImage(img_name)
            mask = sitk.GetArrayFromImage(itk_mask)
            mask = np.rollaxis(mask, 0, 3)
            mask[mask > 250] = 1 # In case using 255 instead of 1
            mask[mask > 4.5] = 0 # Trachea = 5
            mask[mask >= 1] = 1 # Left lung = 3, Right lung = 4
            mask[mask != 1] = 0 # Non-Lung/Background
            mask = mask.astype(np.uint8)"""

        """try:
            f, ax = plt.subplots(1, 3, figsize=(15, 5))

            ax[0].imshow(img[:, :, img.shape[2] // 3], cmap='gray')
            if not no_masks:
                ax[0].imshow(mask[:, :, img.shape[2] // 3], alpha=0.15)
            ax[0].set_title('Slice {}/{}'.format(img.shape[2] // 3, img.shape[2]))
            ax[0].axis('off')

            ax[1].imshow(img[:, :, img.shape[2] // 2], cmap='gray')
            if not no_masks:
                ax[1].imshow(mask[:, :, img.shape[2] // 2], alpha=0.15)
            ax[1].set_title('Slice {}/{}'.format(img.shape[2] // 2, img.shape[2]))
            ax[1].axis('off')

            ax[2].imshow(img[:, :, img.shape[2] // 2 + img.shape[2] // 4], cmap='gray')
            if not no_masks:
                ax[2].imshow(mask[:, :, img.shape[2] // 2 + img.shape[2] // 4], alpha=0.15)
            ax[2].set_title('Slice {}/{}'.format(img.shape[2] // 2 + img.shape[2] // 4, img.shape[2]))
            ax[2].axis('off')

            fig = plt.gcf()
            fig.suptitle(fname)

            plt.savefig(join(fig_path, fname + '.png'), format='png', bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print('\n'+'-'*100)
            print('Error creating qualitative figure for {}'.format(fname))
            print(e)
            print('-'*100+'\n')"""

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

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def generate_train_batches(label,train_list, net_input_shape,  batchSize=1, numSlices=1, subSampAmt=-1,
                           stride=1, downSampAmt=1, shuff=1, aug_data=1):
    # Create placeholders for training
    img_batch = np.zeros((np.concatenate(((batchSize,), net_input_shape))), dtype=np.float32)
    mask_batch = np.zeros((np.concatenate(((batchSize,), net_input_shape))), dtype=np.uint8)
    print("INSIDE train")
    while True:
        if shuff:
            shuffle(train_list)
        count = 0
        for i, scan_name in enumerate(train_list):
            try:
                scan_name = scan_name
                path_to_np = join('np_files',basename(scan_name[1])[:-7]+'.npz')
                with np.load(path_to_np) as data:
                    train_img = data['img']
                    train_mask = data['mask']
            except:
                print('\nPre-made numpy array not found for {}.\nCreating now...'.format(join('np_files',basename(scan_name[1])[:-7]+'.npz')))
                train_img, train_mask = convert_data_to_numpy(scan_name, train=True)
                if np.array_equal(train_img,np.zeros(1)):
                    continue
                else:
                    print('\nFinished making npz file.')
            if numSlices == 1:
                subSampAmt = 0
            """elif subSampAmt == -1 and numSlices > 1:
                np.random.seed(None)
                subSampAmt = int(rand(1)*(train_img.shape[2]*0.05))"""

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
                elif img_batch.ndim == 5:
                    # Assumes img and mask are single channel. Replace 0 with : if multi-channel.
                    img_batch[:count] = train_img[j:j + numSlices * (subSampAmt+1):subSampAmt+1,:, :]
                    mask_batch[:count] = train_mask[j:j + numSlices * (subSampAmt+1):subSampAmt+1,:, :]
                else:
                    print('Error this function currently only supports 2D and 3D data.')
                    exit(0)

                count += 1
                if count % batchSize == 0:
                    count = 0
                    yield (img_batch, mask_batch)
                    """if aug_data:
                        img_batch, mask_batch = augmentImages(img_batch, mask_batch)
                    if debug:
                        if img_batch.ndim == 4:
                            plt.imshow(np.squeeze(img_batch[0, :, :, 0]), cmap='gray')
                            plt.imshow(np.squeeze(mask_batch[0, :, :, 0]), alpha=0.15)
                        elif img_batch.ndim == 5:
                            plt.imshow(np.squeeze(img_batch[0, :, :, 0, 0]), cmap='gray')
                            plt.imshow(np.squeeze(mask_batch[0, :, :, 0, 0]), alpha=0.15)
                        plt.savefig(join( 'logs', 'ex_train.png'), format='png', bbox_inches='tight')
                        plt.close()
                    if net.find('caps') != -1:
                        yield ([img_batch, mask_batch], [mask_batch, mask_batch*img_batch])
                    else:
                        yield (img_batch, mask_batch)"""

        """if count != 0:
            if aug_data:
                img_batch[:count,...], mask_batch[:count,...] = augmentImages(img_batch[:count,...],
                                                                              mask_batch[:count,...])
            if net.find('caps') != -1:
                yield ([img_batch[:count, ...], mask_batch[:count, ...]],
                       [mask_batch[:count, ...], mask_batch[:count, ...] * img_batch[:count, ...]])
            else:
                yield (img_batch[:count,...], mask_batch[:count,...])"""

@threadsafe_generator
def generate_val_batches(label,val_list, net_input_shape=(512,512,5),  batchSize=1, numSlices=1, subSampAmt=-1,
                           stride=1, downSampAmt=1, shuff=1, aug_data=1):
    img_batch = np.zeros((np.concatenate(((batchSize,), net_input_shape))), dtype=np.float32)
    mask_batch = np.zeros((np.concatenate(((batchSize,), net_input_shape))), dtype=np.uint8)
    while True:
        if shuff:
            shuffle(val_list)
        count = 0
        for i, scan_name in enumerate(train_list):
            try:
                scan_name = scan_name
                path_to_np = join('np_files',basename(scan_name[1])[:-7]+'.npz')
                with np.load(path_to_np) as data:
                    val_img = data['img']
                    val_mask = data['mask']
            except:
                print('\nPre-made numpy array not found for {}.\nCreating now...'.format(join('np_files',basename(scan_name[1])[:-7]+'.npz')))
                val_img, val_mask = convert_data_to_numpy(scan_name)
                if np.array_equal(val_img,np.zeros(1)):
                    continue
                else:
                    print('\nFinished making npz file.')
            if numSlices == 1:
                subSampAmt = 0
            """elif subSampAmt == -1 and numSlices > 1:
                np.random.seed(None)
                subSampAmt = int(rand(1)*(train_img.shape[2]*0.05))"""

            indicies = np.arange(val_img.shape[0])
            if shuff:
                shuffle(indicies)
            print(img_batch.shape)
            for j in indicies:
                #if not np.any(train_mask[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]):
                    #continue
                if img_batch.ndim == 4:
                    img_batch[count:,:,:] = val_img[j:j+1]
                    mask_batch[count:,:,:] = val_mask[j:j+1]
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
    img_slices, lab_slices = get_train_data_slices(train[:2])
    print(img_slices.shape)

    traingen= generate_train_batches("both",train[:2], net_input_shape=(512,512,5), batchSize=4, numSlices=1, subSampAmt=-1,
                               stride=1, downSampAmt=1, shuff=0, aug_data=1)
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
    print(np.unique(np.equal(lab_slices[0], lab_slices[1])))
    #for i in range(lab_slices.shape[0]):
        #print(np.array_equal(lab_slices[i], traingen.next()[1]))
    print("iteration")
    for i in range(0,lab_slices.shape[0] - lab_slices.shape[0]%4,4):
        print(i)

        l= traingen.next()[1]
        print(l.shape)
        print(np.unique(np.equal(l, lab_slices[i:i+4])))
