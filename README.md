

# Visualizing Adversarial Examples on Convolutional Networks
This is my Internship project at [CAMP](http://campar.in.tum.de/WebHome), [TUM](https://www.tum.de/), under supervision of [Professor Nassir Navab](http://campar.in.tum.de/Main/NassirNavab), [Dr. Federico Tombari](http://campar.in.tum.de/Main/FedericoTombari) with [Magda Paschali](http://campar.in.tum.de/Main/MagdaPaschali) as my advisor. It's a package for attacking and visualizing convolutional networks with the purpose of understanding and comparing the effects of adversarial example on such networks.

### Contents
0. [Intro](#intro) 
1. [Tools](#tools)
	1. [Visualization Methods](#visualization-methods)
		* [Guided Back Prop / Vanilla Back Prop](#guided-back-prop)
		* [Smooth Grad](#smooth-grad)
		* [Grad Cam / Guided Gram Cam](#grad-cam)
		* [Interpretable Explanations](#interpretable-explanations)
		*  [Inverted image representations](#inverted-image-representations)
		* [Deep Dream](#deep-dream)
		*  [Deep Image Prior](#deep-image-prior)	
	2. [Attack Types](#attack-types)
		* [FGSM](#fgsm)
		* [PGD / Random Start PGD](#pgd)
		* [Single Pixel](#single-pixel)
		* [Boundary](#boundary)
		* [Deep Fool](#deep-fool)
		* [LBFGS](#lbfgs)
		* [Saliency Map](#saliency-map)
	3. [Convolutional Network & Training Choices](#convolutional-networks)
		* [AlexNet](#alexnet)
		* [VGG19](#vgg19)
			* [Pytorch Pretrained Model on ImageNet](#vgg19)
			* [Custom Training on CIFAR10](#vgg19)
			* [Custom Adversarial Training on CIFAR10](#vgg19)
		* [ResNet50](#resnet50)
			* [Pytorch Pretrained Model on ImageNet](#resnet50)
			* [Custom Training on CIFAR10](#resnet50)
			* [Custom Adversarial Training on CIFAR10](#resnet50)
2. [Code Structure](#code-structure)
	1. [Other Functions](#other-functions)
		* [Comparison Functions](#comparison-functions)
		* [runGradCam2](#rungradcam2)
		* [runExplain2](#runexplain2)
	3. [TL;DR: Step By Step Instructions](#step-by-step-instructions)
4. [Requirements](#requirements)
5. [References](#references)

## Intro
To be updated soon.
## Tools
### Visualization Methods
#### Guided Back Prop
1. Instructions
In `massRun.py` run the function `runGBackProp` for guided backprop method, or `runVanillaBP` for Vanilla Back prop. For instance:

```python
runGBackProp(choose_network = 'ResNet50',
                 isTrained = True,
                 training = "Normal",
                 structure="",
                 target_example = 4,
                 attack_type = 'LBFGS')
```

![Output](https://raw.githubusercontent.com/svarthafnyra/CAMP-Project/master/Concise%20Results/pelican_FGSM_Guided%20Back%20Prop(TrainedResNet50).png)

- For more information on `choose_network`, `isTrained`, `training`, `structure` see [this section](#convolutional-networks).
- For more information on `attack_type` check [the list of attacks](#attack-types). `
- `target_example` let's you choose between 6 sample images drawn from [ImageNet](www.image-net.org) if you are using a pretrained Pytorch network. In case of using a custom network, this argument is redundant, because every time a random image is chosen from CIRFAR10 test set. To change this random setting, you can change `get_params` function in `misc_functions.py`.

2. Reference
>J. T. Springenberg, A. Dosovitskiy, T. Brox, and M. Riedmiller. Striving for Simplicity: The All Convolutional Net, https://arxiv.org/abs/1412.6806
    
>K. Simonyan, A. Vedaldi, A. Zisserman. Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps, https://arxiv.org/abs/1312.6034
 
#### Smooth Grad
1. Instructions
In `massRun.py` run the function `runsmoothGrad` for smooth guided grad method. For instance:

```python
runsmoothGrad(choose_network = 'VGG19',
                 isTrained = True,
                 training = "Normal",
                 target_example = 4,
                 attack_type = 'SalMap')
```
![Output](https://raw.githubusercontent.com/svarthafnyra/CAMP-Project/master/Concise%20Results/pelican_RPGD_SmoothGrad(TrainedResNet50).png)

- For more information on `choose_network`, `isTrained`, `training`,  `structure` see [this section](#convolutional-networks).
- For more information on `attack_type` check [the list of attacks](#attack-types). `
- `target_example` let's you choose between 6 sample images drawn from [ImageNet](www.image-net.org) if you are using a pretrained Pytorch network. In case of using a custom network, this argument is redundant, because every time a random image is chosen from CIRFAR10 test set. To change this random setting, you can change `get_params` function in `misc_functions.py`.

2. Reference
>D. Smilkov, N. Thorat, N. Kim, F. Viégas, M. Wattenberg. SmoothGrad: removing noise by adding noise https://arxiv.org/abs/1706.03825

#### Grad Cam
1. Instructions
In `massRun.py` run the function `runGradCam` for Grad Cam method, or `runGGradCam` for Guided Grad Cam. For instance:

```python
runGradCam(choose_network = 'ResNet50',
                 isTrained = True,
                 target_example = 4,
                 attack_type = 'SalMap)
```
![Output](https://raw.githubusercontent.com/svarthafnyra/CAMP-Project/master/Concise%20Results/pelican_SalMap_GradCam(TrainedResNet50).png)

- For more information on `choose_network`, `isTrained`, `training`, `structure` see [this section](#convolutional-networks).
- For more information on `attack_type` check [the list of attacks](#attack-types). `
- `target_example` let's you choose between 6 sample images drawn from [ImageNet](www.image-net.org) if you are using a pretrained Pytorch network. In case of using a custom network, this argument is redundant, because every time a random image is chosen from CIRFAR10 test set. To change this random setting, you can change `get_params` function in `misc_functions.py`.

2. Reference
>R. R. Selvaraju, A. Das, R. Vedantam, M. Cogswell, D. Parikh, and D. Batra. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization, https://arxiv.org/abs/1610.02391

#### Interpretable Explanations
1. Instructions
In `massRun.py` run the function `runExplain` for Interpretable explanations method. For instance:
```python
runExplain(choose_network='ResNet50',
               isTrained=True,
               target_example=5,
               iters=5,
               attack_type='RPGD')
```
![Output](https://raw.githubusercontent.com/svarthafnyra/CAMP-Project/master/Concise%20Results/admiral_RPGD_InterpExp(TrainedResNet50).png)

- For more information on `choose_network`, `isTrained`, `training`, `structure` see [this section](#convolutional-networks).
- For more information on `attack_type` check [the list of attacks](#attack-types). `
- `target_example` let's you choose between 6 sample images drawn from [ImageNet](www.image-net.org) if you are using a pretrained Pytorch network. In case of using a custom network, this argument is redundant, because every time a random image is chosen from CIRFAR10 test set. To change this random setting, you can change `get_params` function in `misc_functions.py`.
- `iters` sets the number of iterations for optimizing the Interpretable mask. For a clean output, choose a value above 100.

2. Reference
>R. Fong, A. Vedaldi. Interpratable Explanations of Black Boxes by Meaningful Perturbations, https://arxiv.org/abs/1704.03296

#### Inverted Image Representations
1. Instructions
> Note that this method is *only* implemented for Pytorch pretrained AlexNet or VGG19. The method is also not supported by *any of the comparison functions*. Use with caution!

In `massRun.py` run the function `runInvRep` for Inverted Image Representations method. For instance:
```python
runInvRep(choose_network = 'AlexNet',
              isTrained = True,
              target_example = 4,
              target_layer = 10,
              attack_type = 'FGSM')
```
![Output](https://raw.githubusercontent.com/svarthafnyra/CAMP-Project/master/Concise%20Results/pelican_FGSM_InvertedRep(TrainedVGG19).png
)

- For more information on `choose_network`, `isTrained`, `training`, `structure` see [this section](#convolutional-networks).
- For more information on `attack_type` check [the list of attacks](#attack-types). `
- `target_example` let's you choose between 6 sample images drawn from [ImageNet](www.image-net.org) if you are using a pretrained Pytorch network. In case of using a custom network, this argument is redundant, because every time a random image is chosen from CIRFAR10 test set. To change this random setting, you can change `get_params` function in `misc_functions.py`.
- `target_layer` sets the number of the layer you want to start the inverting algorithm from.


2. Reference
>A. Mahendran, A. Vedaldi. Understanding Deep Image Representations by Inverting Them, https://arxiv.org/abs/1412.0035 
>
#### Deep Dream
1. Instructions
> Note that this method is *only* implemented for Pytorch pretrained AlexNet or VGG19. The method is also not supported by *any of the comparison functions*. Use with caution!
In `massRun.py` run the function `runDeepDream` for Inverted Image Representations method. For instance:
```python
runDeepDream(choose_network = 'VGG19',
                 isTrained = True,
                 target_example = 3,
                 attack_type = 'FGSM',
                 cnn_layer = 34,
                 filter_pos = 94,
                 iters = 50)
```
- For more information on `choose_network`, `isTrained`, `training`, `structure` see [this section](#convolutional-networks).
- For more information on `attack_type` check [the list of attacks](#attack-types). `
- `target_example` let's you choose between 6 sample images drawn from [ImageNet](www.image-net.org) if you are using a pretrained Pytorch network. In case of using a custom network, this argument is redundant, because every time a random image is chosen from CIRFAR10 test set. To change this random setting, you can change `get_params` function in `misc_functions.py`.
- `cnn_layer`
- `filter_pos`
- `iters`
2. Reference
> D. Smilkov, N. Thorat, N. Kim, F. Viégas, M. Wattenberg. SmoothGrad: removing noise by adding noise https://arxiv.org/abs/1706.03825
#### Deep Image Prior
To be added soon.
>Dmitry Ulyanov, Andrea Vedaldi, Victor Lempitsky. Deep Image Prior, https://arxiv.org/abs/1711.10925

### Attack Types
All the attacks are implemented using [FoolBox Package](https://foolbox.readthedocs.io/en/latest/).
#### FGSM
>Alexey Kurakin, Ian Goodfellow, Samy Bengio, *Adversarial examples in the physical world*,
https://arxiv.org/abs/1607.02533

#### PGD
>Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, Adrian Vladu, *Towards Deep Learning Models Resistant to Adversarial Attacks*, https://arxiv.org/abs/1706.06083
#### Single Pixel
>Nina Narodytska, Shiva Prasad Kasiviswanathan, *Simple Black-Box Adversarial Perturbations for Deep Networks*, https://arxiv.org/pdf/1612.06299.pdf
#### Boundary
>Wieland Brendel, Jonas Rauber, Matthias Bethge, *Decision-Based Adversarial Attacks: Reliable Attacks Against Black-Box Machine Learning Models*, https://arxiv.org/abs/1712.04248
#### Deep Fool
>Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Pascal Frossard, *DeepFool: a simple and accurate method to fool deep neural networks*, https://arxiv.org/abs/1511.04599
#### LBFGS
> Pedro Tabacof, Eduardo Valle. *Exploring the Space of Adversarial Images*, https://arxiv.org/abs/1510.05328
#### Saliency Map 
>Nicolas Papernot, Patrick McDaniel, Somesh Jha, Matt Fredrikson, Z. Berkay Celik, Ananthram Swami. *The Limitations of Deep Learning in Adversarial Settings*,   https://arxiv.org/abs/1511.07528	

### Convolutional Networks
#### AlexNet
The only available option is the [Pytorch Model](https://pytorch.org/docs/stable/torchvision/models.html), Pretrained on ImageNet.
* Set `choose_network = 'AlexNet'`.
* Set `isTrained = True` if you want to work with the pretrained PyTorch Model. You may set `isTrained = False` to run the model with random weights for sanity check.

#### VGG19
There are 3 available training options.
* [Pytorch Model](https://pytorch.org/docs/stable/torchvision/models.html), Pretrained on ImageNet.
	* Set `choose_network = 'VGG19'`.
	* Set `isTrained = True` if you want to work with the pretrained PyTorch Model. You may set `isTrained = False` to run the model with random weights for sanity check.
* Normal Custom Training on CIFAR10. 
You should train the model by running `normalCifar.py` if the corresponding ckpt file doesn't exist in the `customization/trainedmodels` directory.
	* Set `choose_network = 'Custom'`.
	* Set `structure = 'VGG19'`.
	* Set `training = 'Normal`.

* Custom [Adversarial Training](https://arxiv.org/abs/1412.6572) on CIFAR10.
You should train the model by running `adversCifar.py` if the corresponding ckpt file doesn't exist in the `customization/trainedmodels` directory.
	* Set `choose_network = 'Custom'`.
	* Set `structure = 'VGG19'`.
	* Set `training = 'Adversarial`.

#### ResNet50
There are 3 available training options.
* [Pytorch Model](https://pytorch.org/docs/stable/torchvision/models.html), Pretrained on ImageNet.
	* Set `choose_network = 'ResNet50'`.
	* Set `isTrained = True` if you want to work with the pretrained PyTorch Model. You may set `isTrained = False` to run the model with random weights for sanity check.
* Normal Custom Training on CIFAR10. 
You should train the model by running `normalCifar.py` if the corresponding ckpt file doesn't exist in the `customization/trainedmodels` directory.
	* Set `choose_network = 'Custom'`.
	* Set `structure = 'ResNet50'`.
	* Set `training = 'Normal`.
* Custom [Adversarial Training](https://arxiv.org/abs/1412.6572) on CIFAR10.
You should train the model by running `adversCifar.py` if the corresponding ckpt file doesn't exist in the `customization/trainedmodels` directory.
	* Set `choose_network = 'Custom'`.
	* Set `structure = 'ResNet50'`.
	* Set `training = 'Adversarial`.

## Code Structure
### Other Functions
#### Comparison Functions
There are 4 functions written for making the following comparisons:
* Among Attacks(`CompareAttacks`): For a specific network, one can see how different attacks are visualized using the same visualization method. It is executed from `massRun.py` by entering:
```python
compareAttacks(vizmethod = 'Explain',  
                   choose_network = 'Custom',  
                   image_index = 4,  
                   training='Normal',  
                   structure='ResNet50'):
```
![Output](https://raw.githubusercontent.com/svarthafnyra/CAMP-Project/master/Comparing/AttackComp_pelican_Explain%20(ResNet50__%20).png)

* Among Visualization Methods(`CompareVisualization`): For a specific network and attack type, one can compare chosen visualization methods. Implemented similar to `CompareAttacks`,
```python
compareVisualizations(attackmethod = 'Boundary',  
                          choose_network = 'Custom',  
                          image_index = 5,  
                          training='Adverarial',  
                          structure='VGG19')
```
* Among Networks(`CompareNetworks`): For a specific attack, one can see how different networks are visualized using the same visualization method. Implemented similar to `CompareAttacks`,
```python
compareNetworks(attackmethod = 'PGD,  
                    vizmethod = 'GradCam',  
                    image_index = 3,  
                    training='Normal') # or `Adversarial`
```
* Among Training (`CompareTraining`): For a selected attack and network, one can compare how different training methods affect the chosen visualization. Currently Normal and [adversarial training](https://arxiv.org/abs/1412.6572) are available, distillation will soon be added. In addition for sanity check, visualization with a noisy input as well as untrained network could be shown. Implemented similar to `CompareAttacks`,
```python
compareTraining(attackmethod = 'SinglePixle',  
                    vizmethod = 'VanillaBP',  
                    structure = 'ResNet50',  
                    image_index = 2)
```
#### runGradCam2
An extension to `runGradCam` which allows you to compare the following Grad Cam visualizations:
* Natural Input Image with the correct class prediction (Ground truth)
* Adversarial Input Image with the adversarial class prediction
* Adversarial Input Image with the correct class prediction (Ground truth)
* Natural Input Image with the adversarial class prediction (The wrong network prediction when fed the adversarial image)

The arguments are similar to Grad Cam and the output will look like this:
![Output](https://raw.githubusercontent.com/svarthafnyra/CAMP-Project/master/Concise%20Results/pelican_FGSM_GradCam2(TrainedResNet50).png)

#### runExplain2

An extension to `runExplain` which allows you to compare the following Interpretable Explanations visualizations:
* Natural Input Image with the correct class prediction (Ground truth)
* Adversarial Input Image with the adversarial class prediction
* Natural Input Image with the adversarial class prediction (The wrong network prediction when fed the adversarial image)

The arguments are similar to Explainable Interpretations and the output will look like this:
![Output](https://raw.githubusercontent.com/svarthafnyra/CAMP-Project/master/Concise%20Results/pelican_SalMap_InterpExp2_200iters(TrainedResNet50).png)

### Step By Step Instructions

1. open `massRun.py`
2. Choose your function amongst the following available ones:
* Single Visualization:
>`runGradCam`, `runGradCam2`, `runGGradCam`, `runsmoothGrad`, `runExplain`, `runExplain2`, `runVanillaBP`, `runInvRep`, `runDeepDream`.

* Comparison Visualizations:
> `CompareTraining`, `CompareVisualizaion`, `CompareNetworks`, `CompareAttacks` 
3. As explained above there is a function for each type of visualization or comparison. The common arguments between all functions are:

* *Choose Network*: Currently you can either choose pretrained `ResNet50`, `VGG19` or `AlexNet` or `Custom` network.

* *Training*: Choose either `Normal` or `Adversarial`.

* *Structure*: Having chosen 'Custom' network, choose its structure from 'ResNet50' and 'VGG19'.

* *Attack Type*: Can be chosen from: `FGSM`, `LBFGS`, `PGD`, `RPGD`, `Boundary`, `DeepFool`, `SinglePixel`, `SalMap`

* *Example Index* (Only for ImageNet): Choose a number from 0-6 to choose an image from `input_images`. If you are using a network trained on CIFAR10 the example would be chosen randomly.

## Requirements
```python
python = 3.5
torch >= 0.4.0
torchvision >= 0.1.9
numpy >= 1.13.0
opencv >= 3.1.0
foolbox >= 1.3.1
```
## References
>1. [Convolutional Neural Network Visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations) By [Utku Ozbulak](https://github.com/utkuozbulak).
>2. [CIFAR10 Adversarial Examples Challenge](https://github.com/MadryLab/cifar10_challenge) By [Madry Lab](https://github.com/MadryLab).
>3. [Train CIFAR10 with PyTorch](https://github.com/kuangliu/pytorch-cifar) By [kuangliu](https://github.com/kuangliu).