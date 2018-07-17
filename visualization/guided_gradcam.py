"""
Created on Thu Oct 23 11:27:15 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import numpy as np
from guided_backprop import GuidedBackprop
from matplotlib import pyplot as plt

from attacks import attack
from misc_functions import (get_params,
                            convert_to_grayscale,
                            save_gradient_images)
from visualization.gradcam import GradCam


def guided_grad_cam(grad_cam_mask, guided_backprop_mask):
    """
        Guided grad cam is just pointwise multiplication of cam mask and
        guided backprop mask

    Args:
        grad_cam_mask (np_arr): Class activation map mask
        guided_backprop_mask (np_arr):Guided backprop mask
    """
    cam_gb = np.multiply(grad_cam_mask, guided_backprop_mask)
    return cam_gb

def runGGradCam(choose_network = 'AlexNet',
                 target_example = 3,
                 attack_type = 'FGSM'):
    #if __name__ == '__main__':
    # Get params
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_params(target_example,choose_network)

    # Grad cam
    gcv2 = GradCam(pretrained_model, target_layer=11)
    # Generate cam mask
    cam = gcv2.generate_cam(prep_img, target_class)
    print('Grad cam completed')

    # Guided backprop
    GBP = GuidedBackprop(pretrained_model)
    # Get gradients
    guided_grads = GBP.generate_gradients(prep_img, target_class)
    print('Guided backpropagation completed')

    # Guided Grad cam
    cam_gb = guided_grad_cam(cam, guided_grads)
    guidedgrad = save_gradient_images(cam_gb, file_name_to_export + '_GGrad_Cam')
    grayscale_cam_gb = convert_to_grayscale(cam_gb)
    grayguidedgrad = save_gradient_images(grayscale_cam_gb, file_name_to_export + '_GGrad_Cam_gray')
    print('Guided grad cam completed')


    fig = plt.figure()
    fig.suptitle(file_name_to_export+' - '+attack_type+' - Guided GradCam')

    ax1 = fig.add_subplot(2,2,1)
    ax1.imshow(guidedgrad)
    ax1.set_title('Guided Grad Cam')
    ax2 = fig.add_subplot(2,2,2)
    ax2.imshow(grayguidedgrad[:,:,0])
    ax2.set_title('Guided Grad Cam Grasycale')


    adversarial,advers_class = attack(attack_type,pretrained_model,original_image,file_name_to_export,target_class)
    cam = gcv2.generate_cam(adversarial, advers_class)
    print('Grad cam completed')

    # Guided backprop
    GBP = GuidedBackprop(pretrained_model)
    # Get gradients
    guided_grads = GBP.generate_gradients(adversarial, advers_class)
    print('Guided backpropagation completed')

    # Guided Grad cam
    cam_gb = guided_grad_cam(cam, guided_grads)
    guidedgrad2 = save_gradient_images(cam_gb, 'Adversary_'+ file_name_to_export + '_GGrad_Cam')
    grayscale_cam_gb = convert_to_grayscale(cam_gb)
    grayguidedgrad2 = save_gradient_images(grayscale_cam_gb,'Adversary_'+ file_name_to_export + '_GGrad_Cam_gray')
    print('Guided grad cam completed')


    ax3 = fig.add_subplot(2,2,3)
    ax3.imshow(guidedgrad2)
    ax3.set_title('Adversary Guided Grad Cam')
    ax4 = fig.add_subplot(2,2,4)
    ax4.imshow(grayguidedgrad2[:,:,0])
    ax4.set_title('Adversary Guided Grad Cam Grasycale')

    fig.set_size_inches(18.5, 10.5)
    fig.savefig('Concise Results/'+file_name_to_export+'_'+attack_type+'_Guided GradCam',dpi = 100)