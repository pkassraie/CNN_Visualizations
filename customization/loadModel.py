import torch
import os
from models.resnet import *
from models.vgg import *


def loadModel(training, structure):
    assert os.path.isdir('./customization/trainedmodels'), 'Error: Model not found. Train it first.'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if structure == 'ResNet50':
        model = ResNet50()
        path = './customization/trainedmodels/'+training+'_'+ structure
        if device == 'cuda':
            model.load_state_dict(torch.load(path),strict=False)
        else:
            model.load_state_dict(torch.load(path,map_location=lambda storage, loc: storage),strict=False)
    elif structure == 'VGG19':
        model = VGG('VGG19')
        path = './customization/trainedmodels/'+training+ '_'+ structure
        if device == 'cuda':
            model.load_state_dict(torch.load(path),strict=False)
        else:
            model.load_state_dict(torch.load(path,map_location=lambda storage, loc: storage),strict=False)

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    return model



