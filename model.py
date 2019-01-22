import numpy as np
import os
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from dice_coefficient_loss import dice_coefficient_loss, dice_coefficient
from metric import *


#TODO make BVNet static like unet
def BVNet(pretrained_weights = None,input_size = (256,256, 5)):
    # Build U-Net model
    inputs = Input((input_size))
   # s = Lambda(lambda x: x / 255) (inputs)

    c1 = Conv2D(64, (3, 3), padding='same') (inputs)
    #c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
   # c1 = Dropout(0.1) (c1)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same') (c1)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    c1 = Conv2D(128, (3, 3), padding='same') (c1)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    p1 = MaxPooling2D((2, 2), strides=(2,2)) (c1)

#Encode block 2
    c2 = Conv2D(128, (3, 3), padding='same') (p1)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    #c2 = Dropout(0.1) (c2)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same') (c2)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    c2 = Conv2D(256, (3, 3), activation='relu', padding='same') (c2)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    p2 = MaxPooling2D((2, 2), strides = (2,2)) (c2)

#Encode block 3
    c3 = Conv2D(256, (3, 3), padding='same') (p2)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    #c3 = Dropout(0.2) (c3)
    c3 = Conv2D(256, (3, 3), padding='same') (c3)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    c3 = Conv2D(512, (3, 3), padding='same') (c3)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    p3 = MaxPooling2D((2, 2), strides=(2,2)) (c3)

# Encode block 4
    c4 = Conv2D(512, (3, 3), padding='same') (p3)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)
    #c4 = Dropout(0.2) (c4)
    c4 = Conv2D(512, (3, 3),  padding='same') (c4)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)
    c4 = Conv2D(1024, (3, 3),  padding='same') (c4)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)

#Decode block 1
    u7 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same') (c4)
    u7 = BatchNormalization()(u7)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(512, (3, 3), padding='same') (u7)
    c7 = BatchNormalization()(c7)
    c7 = Activation('relu')(c7)
    #c7 = Dropout(0.2) (c7)
    c7 = Conv2D(512, (3, 3),  padding='same') (c7)
    c7 = BatchNormalization()(c7)
    c7 = Activation('relu')(c7)
    c7 = Conv2D(512, (3, 3),  padding='same') (c7)
    c7 = BatchNormalization()(c7)
    c7 = Activation('relu')(c7)

#Decode block 2
    u8 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = BatchNormalization()(u8)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(256, (3, 3), padding='same') (u8)
    c8 = BatchNormalization()(c8)
    c8 = Activation('relu')(c8)
    c8 = Conv2D(256, (3, 3), padding='same') (u8)
    c8 = BatchNormalization()(c8)
    c8 = Activation('relu')(c8)
    #c8 = Dropout(0.1) (c8)
    c8 = Conv2D(256, (3, 3), padding='same') (c8)
    c8 = BatchNormalization()(c8)
    c8 = Activation('relu')(c8)

#Decode block 3
    u9 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(128, (3, 3), padding='same') (u9)
    c9 = BatchNormalization()(c9)
    c9 = Activation('relu')(c9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same') (u9)
    c9 = BatchNormalization()(c9)
    c9 = Activation('relu')(c9)

#c9 = Dropout(0.1) (c9)
    #c9 = Conv2D(64, (3, 3), padding='same') (c9)
    #c9 = BatchNormalization()(c9)
    #c9 = Activation('relu')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss=dice_coefficient_loss, metrics=[binary_accuracy, dice_coefficient, recall, precision])
    model.summary()
    return model


def unet(pretrained_weights=None, input_size=(256, 256, 5), number_of_classes=1):
    inputs = Input(input_size)
    conv1 = Conv2D(filters = 64, kernel_size = 3, padding='same')(inputs)
    conv2 = Conv2D(filters = 64, kernel_size = 3, padding='same')(conv1)
    # L2
    down1 = MaxPooling2D(pool_size=(2,2), padding='same')(conv2)
    conv3 = Conv2D(filters = 128, kernel_size = 3, padding='same')(down1)
    conv4 = Conv2D(filters = 128, kernel_size = 3, padding='same')(conv3)
    # L3
    down2 = MaxPooling2D(pool_size=(2,2), padding='same')(conv4)
    conv5 = Conv2D(filters = 256, kernel_size = 3, padding='same')(down2)
    conv6 = Conv2D(filters = 256, kernel_size = 3, padding='same')(conv5)
    # L4
    down3 = MaxPooling2D(pool_size=(2,2), padding='same')(conv6)
    conv7 = Conv2D(filters = 512, kernel_size = 3, padding='same')(down3)
    conv8 = Conv2D(filters = 512, kernel_size = 3, padding='same')(conv7)
    # L5
    down4 = MaxPooling2D(pool_size=(2,2), padding='same')(conv8)
    conv9 = Conv2D(filters=1024, kernel_size = 3, padding='same')(down4)
    conv10 = Conv2D(filters=1024, kernel_size=3, padding='same')(conv9)

    # Merge upsampled L5 with output from L4
    upsampled_conv10 = Conv2DTranspose(filters=512, kernel_size=3, strides=2, padding='same')(conv10)
    #print(upsampled_conv10.shape)
    # Up-L4
    up1 = concatenate([conv8, upsampled_conv10])
    conv11 = Conv2D(filters = 512, kernel_size= 3, padding='same')(up1)
    conv12 = Conv2D(filters = 512, kernel_size= 3, padding='same')(conv11)
    # Up-L3 - merge upsampled L4 with output from L3
    upsampled_conv12 = Conv2DTranspose(filters=256, kernel_size = 3, strides=2, padding='same')(conv12)
    up2 = concatenate([conv6, upsampled_conv12])
    conv13 = Conv2D(filters = 256, kernel_size= 3, padding='same')(up2)
    conv14 = Conv2D(filters = 256, kernel_size= 3, padding='same')(conv13)
    # Up-L2 - merge upsampled L3 with output from L2
    upsampled_conv14 = Conv2DTranspose(filters=128, kernel_size = 3, strides=2, padding='same')(conv14)
    up3 = concatenate([conv4, upsampled_conv14])
    conv15 = Conv2D(filters = 128, kernel_size= 3, padding='same')(up3)
    conv16 = Conv2D(filters = 128, kernel_size= 3, padding='same')(conv15)
    # Up-L1 - merge upsampled L2 with output from L1
    upsampled_conv16 =  Conv2DTranspose(filters=64, kernel_size = 3, strides=2, padding='same')(conv16)
    up4 = concatenate([conv2, upsampled_conv16])
    conv17 = Conv2D(filters = 64, kernel_size = 3, padding='same')(up4)
    conv18 = Conv2D(filters = 64, kernel_size= 3, padding='same')(conv17)

    conv19 = Conv2D(filters = 2, kernel_size = 3, padding='same')(conv18)
    # Final layer - makes 768x768x1 image
    segmap = Conv2D(filters = 1, kernel_size = 1, activation = 'sigmoid')(conv19)

    model = Model(inputs=inputs, outputs=segmap)

    model.compile(optimizer='adam', loss=dice_coefficient_loss, metrics=[binary_accuracy, dice_coefficient, recall, precision])
    model.summary()
    return model

if __name__ == "__main__":
    unet()
