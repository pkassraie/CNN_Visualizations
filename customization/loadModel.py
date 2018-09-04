import torch
import os
from models.resnet import *
from models.vgg import *


def loadModel(training, structure):
    assert os.path.isdir('./customization/trainedmodels'), 'Error: Model not found. Train it first.'

    if structure == 'ResNet50':
        model = ResNet50()
        path = './customization/trainedmodels/'+training+'_'+ structure
        model.load_state_dict(torch.load(path,map_location=lambda storage, loc: storage),strict=False)
    elif structure == 'VGG19':
        model = VGG('VGG19')
        path = './customization/trainedmodels/'+training+ '_'+ structure
        model.load_state_dict(torch.load(path,map_location=lambda storage, loc: storage),strict=False)

    model.eval()

    return model



