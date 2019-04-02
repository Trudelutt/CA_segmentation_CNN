import scipy.misc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import numpy as np
import tensorflow as tf
import skimage
import random
from imgaug import augmenters as iaa
import imgaug as ia
from keras.preprocessing.image import *
from os.path import join, basename

def get_ploting_read_mask(mask):
    print(mask.shape)
    new_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
    #new_mask[mask == 1] = np.array([1,0,0])
    #new_mask[mask == 0] = np.array([0,0,0])
    new_mask[...,0] = mask[...,0]
    return new_mask

def convert_to_uint8(numpy_array):
    conv_numpy = np.uint8(numpy_array)
    for batch in range(numpy_array.shape[0]):
        for channel in range(numpy_array.shape[-1]):
            conv_numpy[batch][...,channel] -= np.min(conv_numpy[batch][...,channel])
            conv_numpy[batch][...,channel] = conv_numpy[batch][...,channel] / np.max(conv_numpy[batch][...,channel])
            conv_numpy[batch][...,channel] *= 255
    return conv_numpy

def convert_to_float32(numpy_array):
    conv_numpy = np.float32(numpy_array)
    for batch in range(numpy_array.shape[0]):
        for channel in range(numpy_array.shape[-1]):
            conv_numpy[batch][...,channel] -= np.mean(conv_numpy[batch][...,channel])
            conv_numpy[batch][...,channel] = conv_numpy[batch][...,channel] / np.std(conv_numpy[batch][...,channel])
    return conv_numpy


def augmentImages(batch_of_images, batch_of_mask, debugg= False):
    uint8_batch_of_images = convert_to_uint8(batch_of_images)
    images_mask_batch = np.concatenate((np.uint8(uint8_batch_of_images), batch_of_mask), axis=3)
    #print(images_mask_batch.shape)
    aug_images, aug_mask = add_afine_transformations(images_mask_batch)
    uint8_aug_images = add_brightness(aug_images)
    #iaa.Sometimes(0.5, aug)
    aug_images = convert_to_float32(uint8_aug_images)
    if(debugg):
        count = 0
        for i in range(len(aug_images)):
            #plot_aug_mask = np.zeros(aug_mask[i].shape)
            plot_aug_mask = aug_mask[i]
            plot_aug_mask = get_ploting_read_mask(plot_aug_mask)
            plt.figure()
            plt.imshow(np.squeeze(aug_images[i][...,2]), cmap='gray')
            plt.imshow(np.squeeze(plot_aug_mask), alpha=0.3, cmap='gray')
            plt.savefig(join( 'logs', 'ex_'+str(i)+'_train.png'), format='png', bbox_inches='tight')
            plt.close()
            plot_gt_mask = batch_of_mask[i]
            plot_gt_mask = get_ploting_read_mask(plot_gt_mask)
            plt.figure()
            plt.imshow(np.squeeze(batch_of_images[i][...,2]), cmap='gray')
            plt.imshow(np.squeeze(plot_gt_mask), alpha=0.3, cmap='gray')
            plt.savefig(join( 'logs', 'ex_'+str(i)+'_gt_train.png'), format='png', bbox_inches='tight')
            plt.close()

            #scipy.misc.imsave("./results/14.feb/aug_image_" + str(i)  +".png", aug_images[i][...,2])
            #aug_image = salt_pepper_noise(i)
            #print("Image shape")
            #scipy.misc.imsave("./results/14.feb/aug_image_gt_" + str(i)  +".png", aug_mask[i][...,0])
            print("Unique mask")
            print(np.unique(aug_mask[i]))
    return aug_images, aug_mask



def add_brightness(images_batch):
    sometimes = lambda aug: iaa.Sometimes(0.1, aug)
    seq = iaa.Sequential([sometimes(iaa.SomeOf((0, None),[
    #seq = iaa.Sequential([iaa.SomeOf(1,[
    iaa.Multiply((0.2, 1.5)),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255))]))])
    aug_images = seq.augment_images(images_batch)
    #print(np.unique(aug_images))
    return aug_images


def add_afine_transformations(images_mask_batch):
    sometimes = lambda aug: iaa.Sometimes(0.1, aug)
    seq = iaa.Sequential([sometimes(iaa.SomeOf((0,None),[
    iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},order=[0],
    cval=0,
    mode='constant'),
    iaa.Affine(translate_percent={"x": (-0.05, 0.05)},order=[0],
    cval=0,
    mode='constant'),
    iaa.Affine(translate_percent={"y": (-0.05, 0.05)},order=[0],
    cval=0,
    mode='constant'),
    iaa.Affine(rotate=(-5, 5),order=[0],
    cval=0,
    mode='constant'),
    iaa.Affine(shear=(-4,4),order=[0],
    cval=0,
    mode='constant'),
    iaa.ElasticTransformation(alpha=(0.8, 1.2), sigma=0.01,
    order=[0],
    cval=0,
    mode='constant')]))])
    aug_images_mask_batch = seq.augment_images(images_mask_batch)
    return aug_images_mask_batch[:,:,:,:-1], aug_images_mask_batch[:,:,:,-1:]









if __name__ == '__main__':
    cup = scipy.misc.imread("./results/14.feb/cup.jpg")
    #cup = cup/ np.std(cup)
    #cup -= np.mean(cup)
    images = np.array(
    [cup for _ in range(32)], dtype=np.uint8)
    #aug_cup = random_shift(cup, wrg=0.2, hrg=0.1, row_axis=0, col_axis=1, channel_axis=2,
    #                                    fill_mode='constant', cval=0.)
    #print(aug_cup)
    #aug_cup = intensity(cup, (5, 20))
    #print(aug_cup)
    #aug_cup = salt_pepper_noise(cup, salt=0.2, amount=0.04)
    #seq = iaa.Multiply((0.4, 1.6))
    #seq = iaa.Sequential([ iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255))])
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    """seq = iaa.Sequential([sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-10, 10),
            shear=(-4, 4),
            order=[0, 1],
            cval=0,
            mode='constant'
        ))])"""
    seq= iaa.Sequential([iaa.SomeOf((0, None),[
    iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},order=[0, 1],
    cval=0,
    mode='constant'),
    iaa.Affine(translate_percent={"x": (-0.05, 0.05)},order=[0, 1],
    cval=0,
    mode='constant'),
    iaa.Affine(translate_percent={"y": (-0.05, 0.05)},order=[0, 1],
    cval=0,
    mode='constant'),
    iaa.Affine(rotate=(-5, 5),order=[0, 1],
    cval=0,
    mode='constant'),
    iaa.Affine(shear=(-4,4),order=[0, 1],
    cval=0,
    mode='constant'),
    iaa.Multiply((0.4, 1.2)),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255))])])
    aug_cup = seq.augment_images(images)
    for i in range(len(aug_cup)):
        print(aug_cup[i].shape)
        print(i)
        #aug_cup = rotation(cup, 90)
        new_aug = np.float32(aug_cup[i])
        new_aug -= np.mean(new_aug)
        new_aug = new_aug / np.std(new_aug)
        scipy.misc.imsave("./results/14.feb/aug_cup" + str(i) + ".png", new_aug)

    #print(cup)
