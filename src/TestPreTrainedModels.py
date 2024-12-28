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
        vec2quantLen : int  = None, # Maximum number of elements in the output vector (clamping). If None, all weights are kept.
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
        print("Error: Input model must be a torch.nn.Module or a tf.keras.Model.")
        return None
    
    if all_weights:
        flattened_weights = np.concatenate(all_weights)
    else:
        return np.array([])

    return flattened_weights
    
    
    
    # for module in model.modules():  # Iterate through all modules
    #     if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.Embedding)):
    #         try:
    #             all_weights.append(module.weight.detach().cpu().numpy().flatten())
    #             if module.bias is not None:
    #                 all_weights.append(module.bias.detach().cpu().numpy().flatten())
    #         except AttributeError:
    #             pass
    #     elif isinstance(model, tf.keras.Model):  # TensorFlow/Keras model
    #         for layer in model.layers:
    #             if hasattr(layer, 'weights'):  # Check if the layer has weights
    #                 for weight in layer.weights:
    #                     all_weights.append(weight.numpy().flatten())        
    #     elif isinstance(module, torch.nn.BatchNorm2d):
    #         try:
    #             all_weights.append(module.weight.detach().cpu().numpy().flatten())
    #             all_weights.append(module.bias.detach().cpu().numpy().flatten())
    #             all_weights.append(module.running_mean.detach().cpu().numpy().flatten())
    #             all_weights.append(module.running_var.detach().cpu().numpy().flatten())
    #         except AttributeError:
    #             pass
    #     else:
    #         error ("In TestPreTrainedModels.extract_and_flatten_weights(). Input model is not a torch.nn.Module.")

    # Concatenate all flattened arrays into a single 1D array
    if all_weights:
        flattened_weights = np.concatenate(all_weights)
    else:
        return np.array([])
    
    # Clamp the number of weights if num_of_weights is provided
    if vec2quantLen is not None:
        flattened_weights = flattened_weights[:vec2quantLen]

    return flattened_weights


# # Example usage:
# resnet18 = models.resnet18(pretrained=True)
#
# # Extract all weights:
# all_flattened_weights = extract_and_flatten_weights(resnet18)
# if all_flattened_weights is not None:
#     print("All Flattened Weights Shape:", all_flattened_weights.shape)
#
# # Extract a maximum of 1000 weights:
# limited_flattened_weights = extract_and_flatten_weights(resnet18, num_of_weights=1000)
# if limited_flattened_weights is not None:
#     print("Limited Flattened Weights Shape:", limited_flattened_weights.shape)
#
# empty_model = torch.nn.Sequential()
# empty_weights = extract_and_flatten_weights(empty_model)
# print(f"Empty weights shape: {empty_weights.shape}") # Output: Empty weights shape: (0,)



# def extractWeightsOfResnetModel (
#         model,
#         verbose = []
#         ) -> np.array:
#     """
#     """
#     vec2quantize = np.array(model.layer1[0].bn1.running_var) # Get the weights for a specific layer (e.g., layer 3) # Get 1K weights.
#     if settings.VERBOSE_DEBUG in verbose:
#         return vec2quantize
#     vec2quantize = np.append (vec2quantize, np.array(model.layer2[0].bn1.running_var))
#     vec2quantize = np.append (vec2quantize, np.array(model.layer3[0].bn1.running_var))
#     vec2quantize = np.append (vec2quantize, np.array(model.layer4[0].bn1.running_var))
#     return vec2quantize 

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

def ModelsQuantRoundErr (
        modelStrs=[], 
        vec2quantLen = None
    ):
    """
    calculate the quantization round error obtained by several models and counter sizes. 
    """
    # model    = MobileNet_V3 (weights=ResNet50_Weights.IMAGENET1K_V2),
    verbose = [VERBOSE_RES] #$$$$, VERBOSE_PCL] #[VERBOSE_RES, VERBOSE_PCL]
    for modelStr in modelStrs:
        model = None
        match modelStr:
            case 'Resnet18':
                model    = resnet18 (weights=ResNet18_Weights.IMAGENET1K_V1)
                vec2quantize = extractWeightsOfModel (model, vec2quantLen=vec2quantLen, verbose=verbose)
            case 'Resnet50':
                model    = resnet50 (weights=ResNet50_Weights.IMAGENET1K_V2)
                vec2quantize = extractWeightsOfModel (model, verbose=verbose)
            case 'MobileNet_V2':
                model = tf.keras.applications.mobilenet_v2.MobileNetV2()
                vec2quantize = extractWeightsOfModel (model, verbose=verbose)
                # vec2quantize = np.array (model.layers[1].weights).flatten() # Get the weights for a specific layer (e.g., layer 3) # Get 1K weights.
                # for i in range(2, 100): #133): # 100: 374720 weights. 133: 1040064 weights
                #     vec2quantize = np.append (vec2quantize, np.array (model.layers[i].weights).flatten()) 
            case 'MobileNet_V3':
                model = tf.keras.applications.MobileNetV3Large()
                vec2quantize = extractWeightsOfModel (model, verbose=verbose)
                # vec2quantize = np.array (model.layers[2].weights).flatten()
                # for layerNum in range (100): # 100: 112976 weights
                #     for i in range(len(model.layers[layerNum].weights)):
                #         vec2quantize = np.append (vec2quantize, np.array (model.layers[layerNum].weights[i]).flatten()) 
            case _:
                print ('In TestQauntModels.ModelsQuantRoundErr(). Sorry, the model {modelStr} you choose is not support yet.')
        vec2quantize = np.array(vec2quantize[:vec2quantLen])
        calcQuantRoundErrOfModel (
            vec2quantize = vec2quantize,
            modelStr     = modelStr,
            verbose      = verbose, 
        )   

if __name__ == '__main__':
    try:
         ModelsQuantRoundErr (
            ['MobileNet_V3'], #, 'MobileNet_V2', 'MobileNet_V3', 'Resnet18', 'Resnet50'],
            vec2quantLen = 10) 
    except KeyboardInterrupt:
        print('Keyboard interrupt.')

# img = Image.open("dog.jpg")
# model.eval ()
# imagenet_data = datasets.ImageNet(`ImageNet <http://image-net.org/>`)
