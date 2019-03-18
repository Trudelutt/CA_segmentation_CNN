from __future__ import print_function

from os.path import join
from os import makedirs
from os import environ
import argparse
import SimpleITK as sitk
import tensorflow as tf
from time import gmtime, strftime
time = strftime("%Y-%m-%d-%H:%M:%S", gmtime())

#from keras.utils import print_summary
from keras.models import load_model

from model import unet, BVNet, BVNet3D
from preprossesing import *
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from metric import *
from loss_function import dice_coefficient_loss, dice_coefficient
from test import test
from train import train_model
#from augmentation import augmentImages

def gpu_config():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    #set_session(tf.Session(config = config))
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

def get_loss(loss):
    if loss == 'dice':
        return dice_coefficient_loss

def get_model(args, train_files, val_files,input_shape=(512,512,5)):
    if args.modelweights != None:
        custom_objects = custom_objects={ 'binary_accuracy':binary_accuracy, 'recall':recall,
        'precision':precision, 'dice_coefficient': dice_coefficient, 'dice_coefficient_loss': dice_coefficient_loss}
        return load_model(args.modelweights, custom_objects=custom_objects)
    #else:
        #if(args.model=="BVNet3D"):
            #train_data, label_data = get_training_patches(train_files, args.label, remove_only_background_patches=True)
            #val_data, val_label = get_training_patches(train_files, args.label)
            #model =  BVNet3D(input_size =train_data.shape[1:], loss=get_loss(args.loss))
            #return model, train_data, label_data, val_data, val_label

    #train_data, label_data = get_train_data_slices(train_files, tag=args.label)
    #print("Done geting training slices...")
    #val_data, val_label = get_slices(val_files, args.label)
    #print("Done geting validation slices...")
    else:
        if(args.model=="BVNet3D"):
            return BVNet3D(input_size =(64,64, 64, 1), loss=get_loss(args.loss))
        if(args.model=="BVNet"):
            return BVNet(input_size =input_shape, loss=get_loss(args.loss))
        if(args.model == "unet"):
            return unet(input_size=input_shape, loss= get_loss(args.loss))
    #return model, None, None, None, None

def main(args):
    gpu_config()
    # Ensure training, testing, and manip are not all turned off
    assert ((args.train or args.test) and args.label ), 'Cannot have train, test, and label all set to 0, Nothing to do.'
    #overwrite = False
    gpu_config()
    # label must be noe of the coronary arteries
    custom_objects = custom_objects={ 'binary_accuracy':binary_accuracy, 'recall':recall,
    'precision':precision, 'dice_coefficient': dice_coefficient, 'dice_coefficient_loss': dice_coefficient_loss}

    if args.modelweights != None:
        modelpath = args.modelweights
    else:
        modelpath = "./models/" +args.model+ "_"+ args.label + "_"+ args.loss + '_batch'+ str(args.batch_size)\
        +"_channels"+ str(args.channels) + "_stride"+str(args.stride)+ "_aug" +str(args.aug) +".hdf5"

    #train_files, val_files, test_files = get_data_files(args.data_root_dir, label=args.label)
    try:
        train_files, val_files, test_files = get_train_val_test(args.label)
    except:
        create_split(args.data_root_dir, args.label)
        train_files, val_files, test_files = get_train_val_test(args.label)

        #prediction_model = get_model(args.model, args.modelweights, train_data.shape[1:], args.loss)

    if args.train:
        prediction_model = get_model(args,train_files[:1], val_files[:1], input_shape=(512,512, args.channels))
        train_model(args,prediction_model, train_files, val_files, modelpath=modelpath)
        #prediction_model = load_model('./models/' + modelpath +'.hdf5', custom_objects=custom_objects)


    if args.test:
        print("Loading model")
        prediction_model= load_model(modelpath, custom_objects=custom_objects)
        test(args, test_files, args.label, prediction_model, modelpath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train on Medical Data')
    parser.add_argument('--data_root_dir', type=str, required=True,
                        help='The root directory for your data.')
    parser.add_argument('--weights_path', type=str, default='',
                        help='/path/to/trained_model.hdf5 from root. Set to "" for none.')
    parser.add_argument('--train', type=int, default=1, choices=[0,1],
                        help='Set to 1 to enable training.')
    parser.add_argument('--model', type=str, default="BVNet", choices=["BVNet","BVNet3D", "unet"],
                        help='Set to 1 to enable training.')
    parser.add_argument('--modelweights', type=str, default=None,
                        help='Set to the path for the  weights of the model check the model folder ex ./model/BVNet_LM.hdf5.')
    parser.add_argument('--test', type=int, default=1, choices=[0,1],
                        help='Set to 1 to enable testing.')
    parser.add_argument('--label',type=str, default='RCA', choices=['RCA', 'LM', 'Aorta', 'both'],
                        help='Which loss to use. "bce" and "w_bce": unweighted and weighted binary cross entropy'
                             '"dice": soft dice coefficient, "mar" and "w_mar": unweighted and weighted margin loss.')
    parser.add_argument('--loss', type=str.lower, default='dice', choices=['bce', 'w_bce', 'dice', 'mar', 'w_mar'],
                        help='Which loss to use. "bce" and "w_bce": unweighted and weighted binary cross entropy'
                             '"dice": soft dice coefficient, "mar" and "w_mar": unweighted and weighted margin loss.')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training.')
    parser.add_argument('--channels', type=int, default=5,
                        help='Number of channels to take in the model.')
    parser.add_argument('--stride', type=int, default=1,
                        help='Stride of the slides to take in the channels.')
    parser.add_argument('--aug', type=int, default=0, choices=[0,1],
                        help='Set to 1 to enable augmentation.')
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
