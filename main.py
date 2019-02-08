from __future__ import print_function

from os.path import join
from os import makedirs
from os import environ
import argparse
import SimpleITK as sitk
import tensorflow as tf
from time import gmtime, strftime
time = strftime("%Y-%m-%d-%H:%M:%S", gmtime())

from keras.utils import print_summary
from keras.models import load_model

from model import unet, BVNet
from preprossesing import *
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from metric import *
from loss_function import dice_coefficient_loss, dice_coefficient
from test import test

def gpu_config():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    #set_session(tf.Session(config = config))
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

def main(args):
    #gpu_config()
    # Ensure training, testing, and manip are not all turned off
    assert ((args.train or args.test) and args.label ), 'Cannot have train, test, and label all set to 0, Nothing to do.'
    #overwrite = False
    gpu_config()
    model_name = "BVNet"
    # label must be noe of the coronary arteries
    label = "both"
    modelpath = model_name+ "_"+ label
    custom_objects = custom_objects={ 'binary_accuracy':binary_accuracy, 'recall':recall,
    'precision':precision, 'dice_coefficient': dice_coefficient, 'dice_coefficient_loss': dice_coefficient_loss}

    train_files, val_files, test_files = get_data_files(args.data_root_dir, label=args.label)
    if args.modelweights != None:
        prediction_model= load_model('./models/' + modelpath +'.hdf5', custom_objects=custom_objects)

    if args.train:
        #from train import train
        # Run training

        train_data, label_data = get_train_data_slices(train_files, tag=label)
        print("Done geting training slices...")
        val_data, val_label = get_slices(val_files, label)
        print("Done geting validation slices...")
        if model_name == "BVNet":
            model = BVNet(args.modelweights, input_size =train_data.shape[1:])
        train_model(model, train_data, label_data, val_data, val_label, modelpath=modelpath)
        prediction_model = load_model('./models/' + modelpath +'.hdf5', custom_objects=custom_objects)


    if args.test:
        #from test import test
        # Run testing
        test(test_files, args.label, prediction_model, modelpath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train on Medical Data')
    parser.add_argument('--data_root_dir', type=str, required=True,
                        help='The root directory for your data.')
    parser.add_argument('--weights_path', type=str, default='',
                        help='/path/to/trained_model.hdf5 from root. Set to "" for none.')
    parser.add_argument('--train', type=int, default=1, choices=[0,1],
                        help='Set to 1 to enable training.')
    parser.add_argument('--modelweights', type=str, default=None,
                        help='Set to the path for the  weights of the model check the model folder ex ./model/BVNet_LM.hdf5.')
    parser.add_argument('--test', type=int, default=1, choices=[0,1],
                        help='Set to 1 to enable testing.')
    parser.add_argument('--label',type=str, default='RCA', choices=['RCA', 'LM', 'Aorta', 'both'],
                        help='Which loss to use. "bce" and "w_bce": unweighted and weighted binary cross entropy'
                             '"dice": soft dice coefficient, "mar" and "w_mar": unweighted and weighted margin loss.')
    parser.add_argument('--loss', type=str.lower, default='w_bce', choices=['bce', 'w_bce', 'dice', 'mar', 'w_mar'],
                        help='Which loss to use. "bce" and "w_bce": unweighted and weighted binary cross entropy'
                             '"dice": soft dice coefficient, "mar" and "w_mar": unweighted and weighted margin loss.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training/testing.')
    parser.add_argument('--initial_lr', type=float, default=0.0001,
                        help='Initial learning rate for Adam.')
    parser.add_argument('--thresh_level', type=float, default=0.,
                        help='Enter 0.0 for otsu thresholding, else set value')
    parser.add_argument('--which_gpus', type=str, default="0",
                        help='Enter "-2" for CPU only, "-1" for all GPUs available, '
                             'or a comma separated list of GPU id numbers ex: "0,1,4".')
    parser.add_argument('--gpus', type=int, default=-1,
                        help='Number of GPUs you have available for training. '
                             'If entering specific GPU ids under the --which_gpus arg or if using CPU, '
                             'then this number will be inferred, else this argument must be included.')

    arguments = parser.parse_args()

    #
    if arguments.which_gpus == -2:
        environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        environ["CUDA_VISIBLE_DEVICES"] = ""
    elif arguments.which_gpus == '-1':
        assert (arguments.gpus != -1), 'Use all GPUs option selected under --which_gpus, with this option the user MUST ' \
                                  'specify the number of GPUs available with the --gpus option.'
    else:
        arguments.gpus = len(arguments.which_gpus.split(','))
        environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        environ["CUDA_VISIBLE_DEVICES"] = str(arguments.which_gpus)

    if arguments.gpus > 1:
        assert arguments.batch_size >= arguments.gpus, 'Error: Must have at least as many items per batch as GPUs ' \
                                                       'for multi-GPU training. For model parallelism instead of ' \
                                                       'data parallelism, modifications must be made to the code.'

    main(arguments)
