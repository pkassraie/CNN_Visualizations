import torch
import os
from models.resnet import *
from models.vgg import *


def loadModel(training, structure):
    assert os.path.isdir('trainedmodels'), 'Error: Model not found. Train it first.'

    if structure == 'ResNet50':
        model = ResNet50()
        path = './trainedmodels/'+training+'_'+ structure
        model.load_state_dict(torch.load(path,map_location=lambda storage, loc: storage))
    elif structure == 'VGG19':
        model = VGG('VGG19')
        path = './trainedmodels/'+training+ '_'+ structure
        model.load_state_dict(torch.load(path))


    return model


#loadModel('Normal','ResNet50')


