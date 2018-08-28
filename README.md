# CAMP-Project

This is a package for attacking and visualizing convolutional networks with the purpose of understanding and comparing the effects of adversarial example on such networks.

![Output](https://raw.githubusercontent.com/svarthafnyra/CAMP-Project/master/Concise%20Results/admiral_RPGD_InterpExp(TrainedResNet50).png)

![Output](https://raw.githubusercontent.com/svarthafnyra/CAMP-Project/master/Concise%20Results/pelican_SalMap_GradCam(TrainedResNet50).png)

## Visualization Methods:

* Guided Back Prop / Vanilla Back Prop

    J. T. Springenberg, A. Dosovitskiy, T. Brox, and M. Riedmiller. Striving for Simplicity: The All Convolutional Net, https://arxiv.org/abs/1412.6806
    
    K. Simonyan, A. Vedaldi, A. Zisserman. Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps, https://arxiv.org/abs/1312.6034
    
* Smooth Back Prop

    D. Smilkov, N. Thorat, N. Kim, F. Viégas, M. Wattenberg. SmoothGrad: removing noise by adding noise https://arxiv.org/abs/1706.03825
* Grad Cam / Guided Gram Cam

    R. R. Selvaraju, A. Das, R. Vedantam, M. Cogswell, D. Parikh, and D. Batra. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization, https://arxiv.org/abs/1610.02391
    
* Deep Dream (Only supports AlexNet & VGG19)

    D. Smilkov, N. Thorat, N. Kim, F. Viégas, M. Wattenberg. SmoothGrad: removing noise by adding noise https://arxiv.org/abs/1706.03825

* Inverted image representations (Only supports AlexNet & VGG19)

    A. Mahendran, A. Vedaldi. Understanding Deep Image Representations by Inverting Them, https://arxiv.org/abs/1412.0035 

* Interpretable Explanations

	R. Fong, A. Vedaldi. Interpratable Explanations of Black Boxes by Meaningful Perturbations, https://arxiv.org/abs/1704.03296
    
* Deep Image Prior (To be implemented)

    Dmitry Ulyanov, Andrea Vedaldi, Victor Lempitsky. Deep Image Prior, https://arxiv.org/abs/1711.10925

## Attacks:

* FGSM

    Alexey Kurakin, Ian Goodfellow, Samy Bengio, “Adversarial examples in the physical world”,
https://arxiv.org/abs/1607.02533
* PGD / Random Start PGD

    Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, Adrian Vladu, “Towards Deep Learning Models Resistant to Adversarial Attacks”, https://arxiv.org/abs/1706.06083
* Single Pixel

    Nina Narodytska, Shiva Prasad Kasiviswanathan, “Simple Black-Box Adversarial Perturbations for Deep Networks”, https://arxiv.org/pdf/1612.06299.pdf
* Boundary

    Wieland Brendel (*), Jonas Rauber (*), Matthias Bethge, “Decision-Based Adversarial Attacks: Reliable Attacks Against Black-Box Machine Learning Models”, https://arxiv.org/abs/1712.04248
* Deep Fool

    Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Pascal Frossard, “DeepFool: a simple and accurate method to fool deep neural networks”, https://arxiv.org/abs/1511.04599
    
* LBFGS

    Pedro Tabacof, Eduardo Valle. Exploring the Space of Adversarial Images, https://arxiv.org/abs/1510.05328


* Saliency Map

    Nicolas Papernot, Patrick McDaniel, Somesh Jha, Matt Fredrikson, Z. Berkay Celik, Ananthram Swami. The Limitations of Deep Learning in Adversarial Settings,   https://arxiv.org/abs/1511.07528	
    
## ConvNets:
  * AlexNet, VGG19 & ResNet50 from torch vision library (pre-trained on ImageNet).

  * ResNet50 & VGG19 with custom training on CIFAR10.

  * ResNet50 & VGG19 with adversarial training on CIFAR10.

## Comparing Functions:

There are 4 functions written for making the following comparisons:
  * Among Visualization Methods(CompareVisualization): For a specific network and attack type, one can compare chosen visualization methods.

  * Among Networks(CompareNetworks): For a specific attack, one can see how different networks are visualized using the same visualization method.


  * Among Attacks(CompareAttacks): For a specific network, one can see how different attacks are visualized using the same visualization method.

  * Among Training (CompareTraining): For a selected attack and network, one can compare how different training methods affect the chosen visualization. Currently Normal and [adversarial training](https://arxiv.org/abs/1412.6572) are available, distillation will soon be added. In addition for sanity check, visualization with a noisey input as well as untrained network could be shown. 

## Instructions:

[1] Open 'massRun.py'

[2] There is a function for each type of visualization or comparision. The common arguments between all functions are:

* Choose Network: Currently you can either choose pretrained 'ResNet50', 'VGG19' or 'AlexNet' or 'Custom' network.

* Training: Choose either 'Normal' or 'Adversarial'.

* Structure: Having chosen 'Custom' network, choose its structure from 'ResNet50' and 'VGG19'.

* Attack Type: Can be chosen from: 'FGSM', 'LBFGS', 'PGD', 'RPGD', 'Boundary', 'DeepFool', 'SinglePixel', 'SalMap'

* Example Index: Choose a number from 0-6 to choose an image from 'input_images'.



