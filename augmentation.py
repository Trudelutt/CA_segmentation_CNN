import scipy.misc
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import skimage
import random
from imgaug import augmenters as iaa
import imgaug as ia
from keras.preprocessing.image import *
from os.path import join, basename


def augmentImages(batch_of_images, batch_of_mask, debugg= False):
    images_mask_batch = np.concatenate((np.uint8(batch_of_images), batch_of_mask), axis=3)
    #print(images_mask_batch.shape)
    aug_images, aug_mask = add_afine_transformations(images_mask_batch)
    aug_images = add_brightness(aug_images)
    #iaa.Sometimes(0.5, aug)
    aug_images = np.float32(aug_images)
    aug_images -= np.mean(aug_images)
    aug_images = aug_images / np.std(aug_images)
    if(debugg):
        count = 0
        for i in range(len(aug_images)):
            #print(np.unique(np.array_equal(batch_of_images, aug_images)))
            #print("aug")
            #print(aug_images.shape)
            plt.figure()
            plt.imshow(np.squeeze(aug_images[i][...,2]), cmap='gray')
            plt.imshow(np.squeeze(aug_mask[i][...,0]), alpha=0.3, cmap='Reds')
            plt.savefig(join( 'logs', 'ex_'+str(i)+'_train.png'), format='png', bbox_inches='tight')
            plt.close()
            plt.figure()
            plt.imshow(np.squeeze(batch_of_images[i][...,2]), cmap='gray')
            plt.imshow(np.squeeze(batch_of_mask[i][...,0]), alpha=0.3, cmap='Reds')
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
    sometimes = lambda aug: iaa.Sometimes(0.2, aug)
    seq = iaa.Sequential([sometimes(iaa.SomeOf((0, None),[
    iaa.Multiply((0.2, 1.5)),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255))]))])
    aug_images = seq.augment_images(images_batch)
    #print(np.unique(aug_images))
    return aug_images


def add_afine_transformations(images_mask_batch):
    sometimes = lambda aug: iaa.Sometimes(0.2, aug)
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
