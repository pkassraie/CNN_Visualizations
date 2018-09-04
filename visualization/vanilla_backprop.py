"""
Created on Thu Oct 26 11:19:58 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import torch
from matplotlib import pyplot as plt
from attacks import attack
from misc_functions import get_params, convert_to_grayscale, save_gradient_images,prediction_reader
import numpy as np
import cv2
class VanillaBackprop():
    """
        Produces gradients generated with vanilla back propagation from the image
    """
    def __init__(self, model,network,structure):
        self.model = model
        self.network = network
        self.structure = structure
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        # Hook the first layer to get the gradient
        self.hook_layers()

    def hook_layers(self):

        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the first layer
        if self.network == "ResNet50":
            first_layer = list(self.model.children())[0]

        elif self.network == "Custom":
            if self.structure == "ResNet50":
                first_layer = list(self.model.children())[0]
            elif self.structure =='VGG19':
                first_layer = list(self.model.features._modules.items())[0][1]
        else:
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

def runVanillaBP(choose_network = 'AlexNet',
                 isTrained = True,
                 training = 'Normal',
                 structure = 'ResNet50',
                 target_example = 3,
                 attack_type = 'FGSM'):
#if __name__ == '__main__':
    # Get params
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_params(target_example,choose_network,isTrained,training,structure)

    # Vanilla backprop
    VBP = VanillaBackprop(pretrained_model,choose_network,structure)
    # Generate gradients
    vanilla_grads = VBP.generate_gradients(prep_img, target_class)
    # Save colored gradients
    vanilbp = save_gradient_images(vanilla_grads, file_name_to_export + '_Vanilla_BP_color')

    # Convert to grayscale
    grayscale_vanilla_grads = convert_to_grayscale(vanilla_grads)
    # Save grayscale gradients
    grayvanilbp = save_gradient_images(grayscale_vanilla_grads, file_name_to_export + '_Vanilla_BP_gray')
    print('Vanilla backprop completed')


    attack1 = attack(choose_network,attack_type,pretrained_model,
                                      original_image,file_name_to_export,target_class)
    adversarialpic,adversarial,advers_class,orig_pred,adver_pred,diff = attack1.getstuff()

    orig_labs,orig_vals = prediction_reader(orig_pred,10,choose_network)
    adver_labs,adver_vals = prediction_reader(adver_pred,10,choose_network)
    indices = np.arange(len(orig_labs))
    # Generate gradients
    vanilla_grads = VBP.generate_gradients(adversarial, advers_class)
    # Save colored gradients
    vanilbp2 = save_gradient_images(vanilla_grads, 'Adversary_'+ file_name_to_export + '_Vanilla_BP_color')
    # Convert to grayscale
    grayscale_vanilla_grads = convert_to_grayscale(vanilla_grads)
    # Save grayscale gradients
    grayvanilbp2 = save_gradient_images(grayscale_vanilla_grads,'Adversary_'+ file_name_to_export + '_Vanilla_BP_gray')
    print('Adversary Vanilla backprop completed')

    fig = plt.figure()
    fig.suptitle(file_name_to_export+' - '+attack_type+' - Vanilla BackProp')
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    ax0 = fig.add_subplot(2,4,1)
    ax0.imshow(original_image)
    ax0.set_title('Original Image')

    ax1 = fig.add_subplot(2,4,2)
    ax1.imshow(vanilbp)
    ax1.set_title('Vanilla BackProp')
    ax2 = fig.add_subplot(2,4,3)
    ax2.imshow(grayvanilbp[:,:,0])
    ax2.set_title('Vanilla BackProp GrayScale')



    ax9 = fig.add_subplot(2,4,4)
    ax9.bar(indices,orig_vals,align='center', alpha=0.5)
    ax9.set_title('Orignial Image Predictions')
    ax9.set_xticks(indices)
    ax9.set_xticklabels(orig_labs,rotation = 45,ha="right")

    adversarial = cv2.cvtColor(np.uint8(adversarialpic), cv2.COLOR_BGR2RGB)
    ax12 = fig.add_subplot(2,4,5)
    ax12.imshow(adversarial)
    ax12.set_title('Adversary Image(SSIM = '+str(diff)+')')

    ax3 = fig.add_subplot(2,4,6)
    ax3.imshow(vanilbp2)
    ax3.set_title('Adversary Vanilla BackProp')
    ax4 = fig.add_subplot(2,4,7)
    ax4.imshow(grayvanilbp2[:,:,0])
    ax4.set_title('Adversary Vanilla BackProp GrayScale')

    ax10 = fig.add_subplot(2,4,8)
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
    fig.savefig('Concise Results/'+file_name_to_export+'_'+attack_type+'_VanillaBP('+train+choose_network+')',dpi = 100)


    return original_image,vanilbp,grayvanilbp[:,:,0],adversarial,vanilbp2,grayvanilbp2[:,:,0],\
           indices,orig_labs,orig_vals,adver_labs,adver_vals

