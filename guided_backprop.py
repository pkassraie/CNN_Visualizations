"""
Created on Thu Oct 26 11:23:47 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import torch
from torch.nn import ReLU
from matplotlib import pyplot as plt
from attacks import attack
from misc_functions import (get_params,
                            convert_to_grayscale,
                            save_gradient_images,
                            get_positive_negative_saliency)


class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the first layer
        first_layer = list(self.model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that it only returns positive gradients
        """
        def relu_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, changes it to zero
            """
            if isinstance(module, ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)
        # Loop through layers, hook up ReLUs with relu_hook_function
        for pos, module in self.model.features._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_hook_function)

    def generate_gradients(self, input_image, target_class):
        # Forward pass
        model_output = self.model(input_image)
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr


if __name__ == '__main__':
    target_example = 1  # Dog
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_params(target_example,'AlexNet')
    attack_type = 'FGSM'

    # Guided backprop
    GBP = GuidedBackprop(pretrained_model)
    # Get gradients
    guided_grads = GBP.generate_gradients(prep_img, target_class)
    # Save colored gradients
    colorgrads = save_gradient_images(guided_grads, file_name_to_export + '_Guided_BP_color')
    # Convert to grayscale
    grayscale_guided_grads = convert_to_grayscale(guided_grads)
    # Save grayscale gradients
    graygrads = save_gradient_images(grayscale_guided_grads, file_name_to_export + '_Guided_BP_gray')
    # Positive and negative saliency maps
    pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
    possal = save_gradient_images(pos_sal, file_name_to_export + '_pos_sal')
    negsal = save_gradient_images(neg_sal, file_name_to_export + '_neg_sal')

    print('Guided backprop completed')

    plt.subplot(2, 4, 1)
    plt.imshow(colorgrads)
    plt.title('Guided BP Color')
    plt.subplot(2, 4, 2)
    plt.imshow(graygrads[:,:,0])
    plt.title( 'Guided BP Gray')
    plt.subplot(2, 4, 3)
    plt.imshow(possal)
    plt.title('Positive Saliency')
    plt.subplot(2, 4, 4)
    plt.imshow(negsal)
    plt.title('Negative Saliency')

    # Now the attack:
    adversarial,advers_class = attack(attack_type,pretrained_model,original_image,file_name_to_export,target_class)
    # Get gradients
    guided_grads = GBP.generate_gradients(adversarial, advers_class)
    # Save colored gradients
    colorgrads = save_gradient_images(guided_grads, 'Adversarial_'+ file_name_to_export + '_Guided_BP_color_')
    # Convert to grayscale
    grayscale_guided_grads = convert_to_grayscale(guided_grads)
    # Save grayscale gradients
    graygrads = save_gradient_images(grayscale_guided_grads,'Adversarial_'+  file_name_to_export + '_Guided_BP_gray')
    # Positive and negative saliency maps
    pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
    possal = save_gradient_images(pos_sal, 'Adversarial_'+ file_name_to_export + '_pos_sal')
    negsal = save_gradient_images(neg_sal, 'Adversarial_'+ file_name_to_export + '_neg_sal')
    print('Adversary Guided backprop completed')

    plt.subplot(2, 4, 5)
    plt.imshow(colorgrads)
    plt.title('Adversarial' 'Guided BP Color')
    plt.subplot(2, 4, 6)
    plt.imshow(graygrads[:,:,0])
    plt.title('Adversarial'+ 'Guided BP Gray')
    plt.subplot(2, 4, 7)
    plt.imshow(possal)
    plt.title('Adversarial ''Positive Saliency')
    plt.subplot(2, 4, 8)
    plt.imshow(negsal)
    plt.title('Adversarial'+'Negative Saliency')

    plt.show()