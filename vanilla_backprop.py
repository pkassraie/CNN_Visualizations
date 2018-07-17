"""
Created on Thu Oct 26 11:19:58 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import torch
from matplotlib import pyplot as plt
from attacks import attack
from misc_functions import get_params, convert_to_grayscale, save_gradient_images

import cv2
class VanillaBackprop():
    """
        Produces gradients generated with vanilla back propagation from the image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        # Hook the first layer to get the gradient
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the first layer
        first_layer = list(self.model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def generate_gradients(self, input_image, target_class):
        # Forward
        model_output = self.model(input_image)
        # Zero grads
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
    # Get params
    target_example = 1  # Dog
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_params(target_example,'AlexNet')
    attack_type = 'FGSM'

    # Vanilla backprop
    VBP = VanillaBackprop(pretrained_model)
    # Generate gradients
    vanilla_grads = VBP.generate_gradients(prep_img, target_class)
    # Save colored gradients
    vanilbp = save_gradient_images(vanilla_grads, file_name_to_export + '_Vanilla_BP_color')

    # Convert to grayscale
    grayscale_vanilla_grads = convert_to_grayscale(vanilla_grads)
    # Save grayscale gradients
    grayvanilbp = save_gradient_images(grayscale_vanilla_grads, file_name_to_export + '_Vanilla_BP_gray')
    print('Vanilla backprop completed')

    fig = plt.figure()
    fig.suptitle(file_name_to_export+' - '+attack_type+' - Vanilla BackProp')

    ax1 = fig.add_subplot(2,2,1)
    ax1.imshow(vanilbp)
    ax1.set_title('Vanilla BackProp')
    ax2 = fig.add_subplot(2,2,2)
    ax2.imshow(grayvanilbp[:,:,0])
    ax2.set_title('Vanilla BackProp GrayScale')


    adversarial,advers_class = attack(attack_type,pretrained_model,original_image,file_name_to_export,target_class)
    # Generate gradients
    vanilla_grads = VBP.generate_gradients(adversarial, advers_class)
    # Save colored gradients
    vanilbp2 = save_gradient_images(vanilla_grads, 'Adversary_'+ file_name_to_export + '_Vanilla_BP_color')
    # Convert to grayscale
    grayscale_vanilla_grads = convert_to_grayscale(vanilla_grads)
    # Save grayscale gradients
    grayvanilbp2 = save_gradient_images(grayscale_vanilla_grads,'Adversary_'+ file_name_to_export + '_Vanilla_BP_gray')
    print('Adversary Vanilla backprop completed')

    ax3 = fig.add_subplot(2,2,3)
    ax3.imshow(vanilbp2)
    ax3.set_title('Adversary Vanilla BackProp')
    ax4 = fig.add_subplot(2,2,4)
    ax4.imshow(grayvanilbp2[:,:,0])
    ax4.set_title('Adversary Vanilla BackProp GrayScale')

    fig.set_size_inches(18.5, 10.5)
    fig.savefig('Concise Results/'+file_name_to_export+'_'+attack_type+'_VanillaBP',dpi = 100)

