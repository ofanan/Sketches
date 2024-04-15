"""
Test a model 
""" 
import torch
import torchvision.transforms as transforms
import torchvision.models as models, torchvision.datasets as datasets
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights 
from PIL import Image
import os, scipy, pickle, numpy as np

import Quantizer, settings

def preprocessImage(imagePath):
    # Define the transformations to be applied to the input image
    transform = transforms.Compose([
        transforms.Resize((255, 255)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # Load the image and apply the defined transformations
    settings.checkIfInputFileExists (imagePath)
    image = Image.open(imagePath)
    image = transform(image).unsqueeze(0)
    return image

def quantizeByPyTorch (model):
    """
    Quantized the model using PyTorach's quantize_dynamic function
    """
    quantizedModel = torch.ao.quantization.quantize_dynamic(
    model,  # the original model  # a set of layers to dynamically quantize
    dtype=torch.qint8)  # the target dtype for quantized weights
    return quantizedModel

# def testModel (model, filesToTest):
#     preds = []
#     runner = 0
#     for f in filesToTest:
#         try:
#             runner += 1
#             with torch.no_grad():
#                 inputImage = self.preprocess_image(f)
#                 output = model(inputImage)
#                 probabilities = torch.nn.functional.softmax(output[0], dim=0)
#             classIdx = torch.argmax(probabilities).item()
#             idx = self.class_dict[str(classIdx)]
#             preds.append(idx[0])
#             print(f'{runner}/50000 Predicted class: {idx[1]}, Probability: {probabilities[classIdx].item()}')
#         except:
#             preds.append(1002)
#             print(f'{runner}/50000 Predicted class: 1002, Probability: CHANNELS ERROR')
#     prec = classification_report(preds, self.true_labels, output_dict=True)['accuracy']
#     print(prec*100,"%")
#     return prec     


def quantizeModels (models):
    """
    Quanitze the model using my quantization function
    """
    # weights = resnet18.layer4[0].bn1.running_var[:settings.VECTOR_SIZE] # Get the weights for a specific layer (e.g., layer 3) # Get 1K weights.
    model        = resnet18 (weights=ResNet18_Weights.IMAGENET1K_V1)
    vec2quantize = resnet18.layer4[0].bn1.running_var[:settings.VECTOR_SIZE] # Get the weights for a specific layer (e.g., layer 3) # Get 1K weights.
    settings.error (f'len={len(vec2quantize)}')
    # vec2quantize = model.layer1[0].bn1.running_var[:settings.VECTOR_SIZE] # Get the weights for a specific layer (e.g., layer 3)
    # verbose = [settings.VERBOSE_RES] #, settings.VERBOSE_PLOT]

    for cntrSize in [8]:
        Quantizer.simQuantRndErr(
            cntrSize        = cntrSize,
            modes           = ['F2P_si_h1'],
            vec2quantize    = vec2quantize,  
            verbose         = [settings.VERBOSE_RES, settings.VERBOSE_PLOT],
        )  
    
if __name__ == '__main__':
    try:
        quantizeModels(model)
    except KeyboardInterrupt:
        print('Keyboard interrupt.')
# resnet18 = resnet18 (weights=ResNet18_Weights.IMAGENET1K_V1)
# weights = resnet18.layer4[0].bn1.running_var[:settings.VECTOR_SIZE] # Get the weights for a specific layer (e.g., layer 3) # Get 1K weights. 


# img = Image.open("dog.jpg")
# model.eval ()
# imagenet_data = datasets.ImageNet(`ImageNet <http://image-net.org/>`)
