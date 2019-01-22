import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.misc
import json
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, TerminateOnNaN
from keras.models import load_model
from model import unet, BVNet
from preprossesing import *
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from metric import *
from dice_coefficient_loss import dice_coefficient_loss, dice_coefficient


def gpu_config():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    #set_session(tf.Session(config = config))
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())


def train_model(model, input, target, val_x, val_y, modelpath):
    print("Inside training")

    model_checkpoint = ModelCheckpoint("./models/"+ modelpath +".hdf5", monitor='val_loss',verbose=1, save_best_only=True)
    model_earlyStopp = EarlyStopping(monitor='val_loss', min_delta=0, patience=12, verbose=1, mode='min', baseline=None, restore_best_weights=False)
    history = model.fit(x=input, y= target, validation_data=(val_x, val_y), batch_size=4, epochs=500, verbose=1, callbacks=[model_checkpoint, model_earlyStopp, TerminateOnNaN()])
    with open('./history/'+ modelpath + '.json', 'w') as f:
        json.dump(history.history, f)
        print("Saved history....")

def predict_model(model, input, target, name='LM_01', label="LM", label_path=None):
    print("Starting predictions")
    p = model.predict(input,  batch_size=1, verbose=1)
    write_pridiction_to_file(target, p, label, path="./predictions/" +label + "/" + name + "prediction.nii.gz", label_path=label_path)


def evaluation(model, test_files, label):
    test_x, test_y = get_slices(test_files, label)
    print("Starting evaluation.....")
    print(model.evaluate(test_x, test_y, batch_size=1, verbose=1))
    print(model.metrics_names)
    print("Evaluation done..")


if __name__ == "__main__":
    overwrite = False
    gpu_config()
    model_name = "BVNet"
    #Hepatic Vessel has label HV
    label = "HV"
    modelpath = model_name+ "_"+ label
    custom_objects = custom_objects={ 'binary_accuracy':binary_accuracy, 'recall':recall,
    'precision':precision, 'dice_coefficient': dice_coefficient, 'dice_coefficient_loss': dice_coefficient_loss}

    train_files, val_files, test_files = get_data_files(data="HV", label=label)
    if  not overwrite:
        prediction_model= load_model('./models/' + modelpath +'.hdf5', custom_objects=custom_objects)
    else:
        train_data, label_data = get_train_data_slices(train_files, tag=label)
        print("Done geting training slices...")
        val_data, val_label = get_slices(val_files, label)
        print("Done geting validation slices...")

        if model_name == "BVNet":
            model = BVNet(input_size =train_data.shape[1:])
        else:
            model_name="unet"
            model = unet(input_size =train_data.shape[1:])
        train_model(model, train_data, label_data, val_data, val_label, modelpath=modelpath)
        prediction_model = load_model('./models/' + modelpath +'.hdf5', custom_objects=custom_objects)
    for i in range(len(test_files)):
        pred_sample, pred_label = get_prediced_image_of_test_files(test_files, i, tag=label)
        predict_model(prediction_model, pred_sample, pred_label, name=modelpath+"_"+str(i)+"_", label=label, label_path=test_files[i][1])
    evaluation(prediction_model, test_files, label)
