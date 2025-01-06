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

import torch
import torchvision.models as models
import numpy as np

def extractWeightsOfModel(
        model, # A PyTorch model.
        verbose      : list = [],
    ) -> np.array: # A 1D NumPy array of flattened weights, or None on error.
    """
    Extracts, flattens, and optionally clamps weights from a PyTorch model.
    """

    all_weights = []

    if isinstance(model, torch.nn.Module):  # PyTorch model
        for module in model.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.Embedding)):
                try:
                    all_weights.append(module.weight.detach().cpu().numpy().flatten())
                    if module.bias is not None:
                        all_weights.append(module.bias.detach().cpu().numpy().flatten())
                except AttributeError:
                    pass
            elif isinstance(module, torch.nn.BatchNorm2d):
                try:
                    all_weights.append(module.weight.detach().cpu().numpy().flatten())
                    all_weights.append(module.bias.detach().cpu().numpy().flatten())
                    all_weights.append(module.running_mean.detach().cpu().numpy().flatten())
                    all_weights.append(module.running_var.detach().cpu().numpy().flatten())
                except AttributeError:
                    pass
    elif isinstance(model, tf.keras.Model):  # TensorFlow/Keras model
        for layer in model.layers:
            if hasattr(layer, 'weights'):  # Check if the layer has weights
                for weight in layer.weights:
                    all_weights.append(weight.numpy().flatten())
    else:
        error ("In TestPrestrainedModels.extractWeightsOfModel(). Input model is of unsupported type.")

    if all_weights:
        return np.concatenate(all_weights)
    
    error ("In TestPrestrainedModels.extractWeightsOfModel(). Failed to extract weights from the model.")


def ModelsQuantRoundErr (
        modelStrs=[], 
        vec2quantLen : INF_INT  = None, # Maximum number of elements in the output vector (clamping). If None, all weights are kept.
    ):
    """
    calculate the quantization round error obtained by several models and counter sizes. 
    """
    verbose = [] #$$ [VERBOSE_RES, VERBOSE_PCL]
    for modelStr in modelStrs:
        model = None
        match modelStr:
            case 'Resnet18':
                model    = resnet18 (weights=ResNet18_Weights.IMAGENET1K_V1)
            case 'Resnet50':
                model    = resnet50 (weights=ResNet50_Weights.IMAGENET1K_V2)
            case 'MobileNet_V2':
                model = tf.keras.applications.mobilenet_v2.MobileNetV2()
            case 'MobileNet_V3':
                model = tf.keras.applications.MobileNetV3Large()
            case _:
                print ('In TestQauntModels.ModelsQuantRoundErr(). Sorry, the model {modelStr} you choose is not support yet.')
                
        vec2quantize = extractWeightsOfModel (model, verbose=verbose)
        for cntrSize in [16]: #$ [8, 16, 19]:
            Quantizer.calcQuantRoundErr(
                cntrSize        = cntrSize,
                signed          = True,
                modes           = settings.modesOfCntrSize(cntrSize),
                vec2quantize    = vec2quantize[:vec2quantLen],  
                inputFrom       = modelStr,       
                verbose         = verbose,
            )  

if __name__ == '__main__':
    try:
        ModelsQuantRoundErr (
            ['MobileNet_V3', 'MobileNet_V2', 'MobileNet_V3', 'Resnet18', 'Resnet50'],
            vec2quantLen = INF_INT) 
    except KeyboardInterrupt:
        print('Keyboard interrupt.')

# img = Image.open("dog.jpg")
# model.eval ()
# imagenet_data = datasets.ImageNet(`ImageNet <http://image-net.org/>`)
