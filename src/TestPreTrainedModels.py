"""
Test a neural network model 
""" 
import torch
import torchvision.transforms as transforms
import torchvision.models as models, torchvision.datasets as datasets
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights, mobilenet_v2, MobileNet_V2_Weights, mobilenet_v3_large, MobileNet_V3_Large_Weights 
import tensorflow as tf
from tensorflow.keras.applications import imagenet_utils
from PIL import Image
import os, scipy, pickle, numpy as np

import Quantizer, settings
from settings import *

def extractWeightsOfResnetModel (
        model,
        verbose = []
        ) -> np.array:
    """
    """
    vec2quantize = np.array(model.layer1[0].bn1.running_var) # Get the weights for a specific layer (e.g., layer 3) # Get 1K weights.
    if settings.VERBOSE_DEBUG in verbose:
        return vec2quantize
    vec2quantize = np.append (vec2quantize, np.array(model.layer2[0].bn1.running_var))
    vec2quantize = np.append (vec2quantize, np.array(model.layer3[0].bn1.running_var))
    vec2quantize = np.append (vec2quantize, np.array(model.layer4[0].bn1.running_var))
    return vec2quantize 

def calcQuantRoundErrOfModel (
        modelStr, # a string defining the model
        vec2quantize,
        verbose   # enum detailing which outputs to write. The enums are defined at settings.py
        ):
    """
    calculate the quantization round error obtained for the given model.
    Output the results as detailed in verbose. 
    """

    for cntrSize in [8, 16, 19]:
        Quantizer.calcQuantRoundErr(
            cntrSize        = cntrSize,
            signed          = False,
            dist            = modelStr,
            modes           = settings.modesOfCntrSize(cntrSize),
            vec2quantize    = vec2quantize[:1000000],  
            verbose         = verbose,
        )  

def ModelsQuantRoundErr (modelStrs=[]):
    """
    calculate the quantization round error obtained by several models and counter sizes. 
    """
    # weights = get_weight("MobileNet_V3_Large_QuantizedWeights.DEFAULT")
    # model    = MobileNet_V3 (weights=ResNet50_Weights.IMAGENET1K_V2),
    # settings.error (weights)
    verbose = [VERBOSE_RES, VERBOSE_PCL] #[VERBOSE_RES, VERBOSE_PCL]
    for modelStr in modelStrs:
        model = None
        if modelStr=='Resnet18':
            model    = resnet18 (weights=ResNet18_Weights.IMAGENET1K_V1)
            vec2quantize = extractWeightsOfResnetModel (model, verbose=verbose)
            weights  = extractWeightsOfResnetModel(model)
        elif modelStr=='Resnet50':
            model    = resnet50 (weights=ResNet50_Weights.IMAGENET1K_V2)
            vec2quantize = extractWeightsOfResnetModel (model, verbose=verbose)
        elif modelStr=='MobileNet_V2':
            model = tf.keras.applications.mobilenet_v2.MobileNetV2()
            vec2quantize = np.array (model.layers[1].weights).flatten() # Get the weights for a specific layer (e.g., layer 3) # Get 1K weights.
            for i in range(2, 100): #133): # 100: 374720 weights. 133: 1040064 weights
                vec2quantize = np.append (vec2quantize, np.array (model.layers[i].weights).flatten()) 
        elif modelStr=='MobileNet_V3':
            model = tf.keras.applications.MobileNetV3Large()
            vec2quantize = np.array (model.layers[2].weights).flatten()
            for layerNum in range (100): # 100: 112976 weights
                for i in range(len(model.layers[layerNum].weights)):
                    vec2quantize = np.append (vec2quantize, np.array (model.layers[layerNum].weights[i]).flatten()) 
        else:
            print ('In TestQauntModels.ModelsQuantRoundErr(). Sorry, the model {modelStr} you choose is not support yet.')
        calcQuantRoundErrOfModel (
            vec2quantize = vec2quantize,
            modelStr     = modelStr,
            verbose      = verbose, 
            )   

if __name__ == '__main__':
    try:
        ModelsQuantRoundErr (['MobileNet_V2', 'MobileNet_V3', 'Resnet18', 'Resnet50']) #'MobileNet_V2', 'Resnet18', 'Resnet50'])
    except KeyboardInterrupt:
        print('Keyboard interrupt.')

# img = Image.open("dog.jpg")
# model.eval ()
# imagenet_data = datasets.ImageNet(`ImageNet <http://image-net.org/>`)
