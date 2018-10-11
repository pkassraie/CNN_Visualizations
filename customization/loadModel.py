import torch
import os
from torchvision import models
import torch.nn as nn
import torch.backends.cudnn as cudnn

def loadModel(training, structure):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    path = './customization/trainedmodels/' + training + '_' + structure
    if training == 'Adversarial':
        cppath = './customization/checkpoint/ckpt_advers_' + structure + '.t7'
    elif training == 'Normal':
        cppath = './customization/checkpoint/ckpt_norm_' + structure + '.t7'

    if structure == 'ResNet50':
        net = models.resnet50()
        num_ftrs = net.fc.in_features
        net.fc = torch.nn.Linear(num_ftrs, 10)
    elif structure == 'VGG19':
        net = models.vgg19()
        num_ftrs = net.classifier[6].in_features
        net.classsifier[6] = torch.nn.Linear(num_ftrs, 10)

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if device == 'cuda':
        checkpoint = torch.load(cppath)
        net.load_state_dict(checkpoint['net'],strict=False) #strict = False ?
    else:
        checkpoint = torch.load(cppath,map_location=lambda storage, loc: storage)
        net.load_state_dict(checkpoint['net'], strict=False) #strict?

    if torch.cuda.is_available():
        net = net.cuda()

    net.eval()

    return net


