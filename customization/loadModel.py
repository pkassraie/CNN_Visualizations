import torch
import os
from models.resnet import *
from models.vgg import *

#net = ResNet18()
#checkpoint = torch.load('ckpt.t7')
#net.load_state_dict(checkpoint['net'])
#print(net.parameters())
#net.eval()
from torchvision.models import resnet50,vgg19


net =vgg19()
#print(list(net.features._modules.items())[0][1])

net2 = VGG('VGG19')
print(list(net.features._modules.items())[0][1])
j = 0
for i in net.features._modules.items():
    #print(j,':',i)
    #print(j'cust:',list(net2.features)[j])
    j += 1


def loadModel(training, structure):
    assert os.path.isdir('trainedmodels'), 'Error: Model not found. Train it first.'
    if training =='Normal':
        if structure == 'ResNet50':
            model = ResNet50()
            path = './trainedmodels/'+'Norm_'+ structure
            model.load_state_dict(torch.load(path))
        elif structure == 'VGG19':
            model = VGG('VGG19')
            path = './trainedmodels/'+'Norm_'+ structure
            model.load_state_dict(torch.load(path))
    elif training == 'Adversarial':
        print('Not Implemented Yet')


    return model


