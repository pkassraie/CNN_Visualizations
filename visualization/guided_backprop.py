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
                            save_gradient_images,prediction_reader,
                            get_positive_negative_saliency)
import numpy as np
import cv2
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


def runGBackProp(choose_network = 'AlexNet',
                 isTrained = True,
                 target_example = 3,
                 attack_type = 'FGSM'):

    #if __name__ == '__main__':
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_params(target_example,choose_network,isTrained)


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
    # Now the attack:
    adversarial,advers_class,orig_pred,adver_pred,diff = attack(attack_type,pretrained_model,original_image,file_name_to_export,target_class)

    orig_labs,orig_vals = prediction_reader(orig_pred,10)
    adver_labs,adver_vals = prediction_reader(adver_pred,10)
    indices = np.arange(len(orig_labs))

    # Get gradients
    guided_grads = GBP.generate_gradients(adversarial, advers_class)
    # Save colored gradients
    colorgrads2 = save_gradient_images(guided_grads, 'Adversarial_'+ file_name_to_export + '_Guided_BP_color_')
    # Convert to grayscale
    grayscale_guided_grads = convert_to_grayscale(guided_grads)
    # Save grayscale gradients
    graygrads2 = save_gradient_images(grayscale_guided_grads,'Adversarial_'+  file_name_to_export + '_Guided_BP_gray')
    # Positive and negative saliency maps
    pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
    possal2 = save_gradient_images(pos_sal, 'Adversarial_'+ file_name_to_export + '_pos_sal')
    negsal2 = save_gradient_images(neg_sal, 'Adversarial_'+ file_name_to_export + '_neg_sal')

    print('Adversary Guided backprop completed')


    fig = plt.figure()
    fig.suptitle(file_name_to_export+' - '+attack_type+' - Guided Back Prop')

    ax11 = fig.add_subplot(2,6,1)
    ax11.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    ax11.set_title('Original Image')

    ax1 = fig.add_subplot(2,6,2)
    ax1.imshow(colorgrads)
    ax1.set_title('Guided BP Color')

    ax2 = fig.add_subplot(2, 6, 3)
    ax2.imshow(graygrads[:,:,0])
    ax2.set_title( 'Guided BP Gray')
    ax3 = fig.add_subplot(2, 6, 4)
    ax3.imshow(possal)
    ax3.set_title('Positive Saliency')
    ax4 = fig.add_subplot(2, 6, 5)
    ax4.imshow(negsal)
    ax4.set_title('Negative Saliency')


    ax9 = fig.add_subplot(2,6,6)
    ax9.bar(indices,orig_vals,align='center', alpha=0.5)
    ax9.set_title('Orignial Image Predictions')
    ax9.set_xticks(indices)
    ax9.set_xticklabels(orig_labs,rotation = 45,ha="right")

    adversarial = cv2.imread('results/'+file_name_to_export+'_'+attack_type +'_Attack.jpg')
    ax12 = fig.add_subplot(2,6,7)
    ax12.imshow(cv2.cvtColor(adversarial, cv2.COLOR_BGR2RGB))
    ax12.set_title('Adversary Image(SSIM = '+str(diff)+')')
    ax5 = fig.add_subplot(2, 6, 8)
    ax5.imshow(colorgrads2)
    ax5.set_title('Adversarial Guided BP Color')
    ax6 = fig.add_subplot(2, 6, 9)
    ax6.imshow(graygrads2[:,:,0])
    ax6.set_title('Adversarial'+ 'Guided BP Gray')
    ax7 = fig.add_subplot(2, 6, 10)
    ax7.imshow(possal2)
    ax7.set_title('Adversarial ''Positive Saliency')
    ax8 = fig.add_subplot(2, 6, 11)
    ax8.imshow(negsal2)
    ax8.set_title('Adversarial'+'Negative Saliency')

    ax10 = fig.add_subplot(2,6,12)
    ax10.bar(indices,adver_vals,align='center', alpha=0.5)
    ax10.set_title('Adversary Image Predictions')
    ax10.set_xticks(indices)
    ax10.set_xticklabels(adver_labs,rotation = 45,ha="right")

    fig.set_size_inches(32, 18)
    fig.tight_layout()
    if isTrained:
        train = 'Trained'
    else:
        train = 'UnTrained'
    fig.savefig('Concise Results/'+file_name_to_export+'_'+attack_type+
                '_Guided Back Prop('+train+choose_network+')',dpi = 100)

