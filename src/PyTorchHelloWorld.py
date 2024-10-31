import torch
import torchvision.transforms as transforms
import torchvision.models as models, torchvision.datasets as datasets
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights 
from PIL import Image
import os, scipy, pickle, numpy as np

import Quantizer, settings

def preprocessImage(imagePath):
    """
    pre-process images before entering them to the ML model. 
    """
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
    Quantize the given model using PyTorach's quantize_dynamic function
    """
    quantizedModel = torch.ao.quantization.quantize_dynamic(
    model,  # the original model  # a set of layers to dynamically quantize
    dtype=torch.qint8)  # the target dtype for quantized weights
    return quantizedModel

def testModel (model, filesToTest):
    """
    Test the given model using the given input fileToTest.
    """
    preds = []
    runner = 0
    for f in filesToTest:
        try:
            runner += 1
            with torch.no_grad():
                inputImage = self.preprocessImage(f)
                output = model(inputImage)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
            classIdx = torch.argmax(probabilities).item()
            idx = self.class_dict[str(classIdx)]
            preds.append(idx[0])
            print(f'{runner}/50000 Predicted class: {idx[1]}, Probability: {probabilities[classIdx].item()}')
        except:
            preds.append(1002)
            print(f'{runner}/50000 Predicted class: 1002, Probability: CHANNELS ERROR')
    prec = classification_report(preds, self.true_labels, output_dict=True)['accuracy']
    print(prec*100,"%")
    return prec     


def quantizeModel (model):
    """
    Quantize the model using my quantization function
    """
    vec2quantize = model.layer1[0].bn1.running_var[:settings.VECTOR_SIZE] # Get the weights for a specific layer (e.g., layer 3) 
    cntrSize=8
    grid = np.array ([item for item in range (-2**(cntrSize-1)+1, 2**(cntrSize-1))])
    [quantizedVec, scale] = Quantizer.quantize(vec=vec2quantize, grid=grid)                
    dequantizedVec = Quantizer.dequantize(vec=quantizedVec, scale=scale)
    return model

model = resnet18 (weights=ResNet18_Weights.IMAGENET1K_V1)
quantizeModel(model)
