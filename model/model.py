import sys
sys.path.append('..')

import os
import pickle
import operator
import numpy as np

import theano
import theano.tensor as T

from lasagne import layers as l
from lasagne import nonlinearities as nl

import config as cfg
from utils import log

NONLINEARITY = 'relu'
FILTERS = [8, 16, 32, 64, 128]
KERNEL_SIZES = [(5, 5), (3, 3), (3, 3), (3, 3), (3, 3)]
RESNET_K = 4
RESNET_N = 3

def loadSnapshot(path):

    log.p(('LOADING SNAPSHOT', path.split(os.sep)[-1], '...'), new_line=False)

    with open(path, 'rb') as f:
        try:
            model = pickle.load(f, encoding='latin1') 
        except:
            model = pickle.load(f) 

    cfg.setModelSettings(model)

    log.p('DONE!')
    
    return model

def loadParams(net, params):

    log.p('IMPORTING MODEL PARAMS...', new_line=False)

    l.set_all_param_values(net, params)

    log.p('DONE!')
    
    return net

def logmeanexp(x, axis=None, keepdims=False, sharpness=5):
    # in between maximum (high sharpness) and mean (low sharpness)
    # https://arxiv.org/abs/1411.6228, Eq. 6
    # return T.log(T.mean(T.exp(sharpness * x), axis, keepdims=keepdims)) / sharpness
    # more stable version (Theano can only stabilize the plain logsumexp)
    xmax = T.max(x, axis, keepdims=True)
    xmax2 = T.max(x, axis, keepdims=keepdims)
    x = sharpness * (x - xmax)
    y = T.log(T.mean(T.exp(x), axis, keepdims=keepdims))
    y = y / sharpness + xmax2
    return y

def resblock(net_in, filters, kernel_size, stride=1, preactivated=True, block_id=1, name=''):

    # Show input shape
    #log.p(("\t\t" + name + " IN SHAPE:", l.get_output_shape(net_in)), new_line=False)

    # Pre-activation
    if block_id > 1:
        net_pre = l.NonlinearityLayer(net_in, nonlinearity=nl.rectify)
    else:
        net_pre = net_in

    # Pre-activated shortcut?
    if preactivated:
        net_in = net_pre

    # Bottleneck Convolution
    if stride > 1:
        net_pre = l.batch_norm(l.Conv2DLayer(net_pre,
                                            num_filters=l.get_output_shape(net_pre)[1],
                                            filter_size=1,
                                            pad='same',
                                            stride=1,
                                            nonlinearity=nl.rectify))
    
    # First Convolution     
    net = l.batch_norm(l.Conv2DLayer(net_pre,
                                   num_filters=l.get_output_shape(net_pre)[1],
                                   filter_size=kernel_size,
                                   pad='same',
                                   stride=1,
                                   nonlinearity=nl.rectify))

    # Pooling layer
    if stride > 1:
        net = l.MaxPool2DLayer(net, pool_size=(stride, stride))

    # Dropout Layer
    net = l.DropoutLayer(net)        

    # Second Convolution
    net = l.batch_norm(l.Conv2DLayer(net,
                        num_filters=filters,
                        filter_size=kernel_size,
                        pad='same',
                        stride=1,
                        nonlinearity=None))

    # Shortcut Layer
    if not l.get_output_shape(net) == l.get_output_shape(net_in):

        # Average pooling
        shortcut = l.Pool2DLayer(net_in, pool_size=(stride, stride), stride=stride, mode='average_exc_pad')

        # Shortcut convolution
        shortcut = l.batch_norm(l.Conv2DLayer(shortcut,
                                 num_filters=filters,
                                 filter_size=1,
                                 pad='same',
                                 stride=1,
                                 nonlinearity=None))        
        
    else:

        # Shortcut = input
        shortcut = net_in
    
    # Merge Layer
    out = l.ElemwiseSumLayer([net, shortcut])

    # Show output shape
    #log.p(("OUT SHAPE:", l.get_output_shape(out), "LAYER:", len(l.get_all_layers(out)) - 1))

    return out

def classificationBranch(net, kernel_size):

    # Post Convolution
    branch = l.batch_norm(l.Conv2DLayer(net,
                        num_filters=int(FILTERS[-1] * RESNET_K),
                        filter_size=kernel_size,
                        nonlinearity=nl.rectify))

    #log.p(("\t\tPOST  CONV SHAPE:", l.get_output_shape(branch), "LAYER:", len(l.get_all_layers(branch)) - 1))

    # Dropout Layer
    branch = l.DropoutLayer(branch)
    
    # Dense Convolution
    branch = l.batch_norm(l.Conv2DLayer(branch,
                        num_filters=int(FILTERS[-1] * RESNET_K * 2),
                        filter_size=1,
                        nonlinearity=nl.rectify))

    #log.p(("\t\tDENSE CONV SHAPE:", l.get_output_shape(branch), "LAYER:", len(l.get_all_layers(branch)) - 1))
    
    # Dropout Layer
    branch = l.DropoutLayer(branch)
    
    # Class Convolution
    branch = l.Conv2DLayer(branch,
                        num_filters=len(cfg.CLASSES),
                        filter_size=1,
                        nonlinearity=None)
    return branch

def buildNet():

    log.p('BUILDING BirdNET MODEL...', new_line=False)

    # Input layer for images
    net = l.InputLayer((None, cfg.IM_DIM, cfg.IM_SIZE[1], cfg.IM_SIZE[0]))    

    # Pre-processing stage
    #log.p(("\tPRE-PROCESSING STAGE:"))
    net = l.batch_norm(l.Conv2DLayer(net,
                    num_filters=int(FILTERS[0] * RESNET_K),
                    filter_size=(5, 5),
                    pad='same',
                    nonlinearity=nl.rectify))
    
    #log.p(("\t\tFIRST  CONV OUT SHAPE:", l.get_output_shape(net), "LAYER:", len(l.get_all_layers(net)) - 1))

    # Max pooling
    net = l.MaxPool2DLayer(net, pool_size=(1, 2))
    #log.p(("\t\tPRE-MAXPOOL OUT SHAPE:", l.get_output_shape(net), "LAYER:", len(l.get_all_layers(net)) - 1))
    
    # Residual Stacks
    for i in range(1, len(FILTERS)):
        #log.p(("\tRES STACK", i, ':'))
        net = resblock(net,
                       filters=int(FILTERS[i] * RESNET_K),
                       kernel_size=KERNEL_SIZES[i],
                       stride=2,
                       preactivated=True,
                       block_id=i,
                       name='BLOCK ' + str(i) + '-1')
        
        for j in range(1, RESNET_N):
            net = resblock(net,
                           filters=int(FILTERS[i] * RESNET_K),
                           kernel_size=KERNEL_SIZES[i],
                           preactivated=False,
                           block_id=i+j,
                           name='BLOCK ' + str(i) + '-' + str(j + 1))
        
    # Post Activation
    net = l.batch_norm(net)
    net = l.NonlinearityLayer(net, nonlinearity=nl.rectify)
    
    # Classification branch
    #log.p(("\tCLASS BRANCH:"))
    net = classificationBranch(net,  (4, 10)) 
    #log.p(("\t\tBRANCH OUT SHAPE:", l.get_output_shape(net), "LAYER:", len(l.get_all_layers(net)) - 1))

    # Pooling
    net = l.GlobalPoolLayer(net, pool_function=logmeanexp)
    #log.p(("\tGLOBAL POOLING SHAPE:", l.get_output_shape(net), "LAYER:", len(l.get_all_layers(net)) - 1))

    # Sigmoid output
    net = l.NonlinearityLayer(net, nonlinearity=nl.sigmoid)

    #log.p(("\tFINAL NET OUT SHAPE:", l.get_output_shape(net), "LAYER:", len(l.get_all_layers(net))))
    log.p("DONE!")

    # Model stats
    #log.p(("MODEL HAS", (sum(hasattr(layer, 'W') for layer in l.get_all_layers(net))), "WEIGHTED LAYERS"))
    #log.p(("MODEL HAS", l.count_params(net), "PARAMS"))

    return net

def test_function(net, layer_index=-1):

    log.p('COMPILING THEANO TEST FUNCTION FUNCTION...', new_line=False)    

    prediction = l.get_output(l.get_all_layers(net)[layer_index], deterministic=True)    
    test_function = theano.function([l.get_all_layers(net)[0].input_var], prediction, allow_input_downcast=True)        

    log.p('DONE!')

    return test_function

def prepareInput(spec):

    # ConvNet inputs in Theano are 4D-vectors: (batch size, channels, height, width)
    
    # Add axis if grayscale image
    if len(spec.shape) == 2:
        spec = spec[:, :, np.newaxis]

    # Transpose axis, channels = axis 0
    spec = np.transpose(spec, (2, 0, 1))

    # Add new dimension
    spec = np.expand_dims(spec, 0)

    return spec

def flat_sigmoid(x, sensitivity=-1):
    return 1 / (1.0 + np.exp(sensitivity * x))

def predictionPooling(p, sensitivity=-1, mode='avg'):

    # Apply sigmoid function
    p = flat_sigmoid(p, sensitivity)

    # Mean exponential pooling for monophonic recordings
    if mode == 'mexp':
        p_pool = np.mean((p * 2.0) ** 2, axis=0)

    # Simple average pooling
    else:        
        p_pool = np.mean(p, axis=0)
    
    p_pool[p_pool > 1.0] = 1.0

    return p_pool

def predict(spec_batch, test_function):

    # Prediction
    prediction = test_function(spec_batch)

    # Prediction pooling
    p_pool = predictionPooling(prediction, cfg.SENSITIVITY)

    # Get label and scores for pooled predictions  
    p_labels = {}  
    for i in range(p_pool.shape[0]):
        label = cfg.CLASSES[i]
        if cfg.CLASSES[i] in cfg.WHITE_LIST:
            p_labels[label] = p_pool[i]
        else:
            p_labels[label] = 0.0

    # Sort by score
    p_sorted =  sorted(p_labels.items(), key=operator.itemgetter(1), reverse=True)
        
    return p_sorted, p_pool