"""
Created on Thu Oct 23 11:27:15 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import numpy as np
from matplotlib import pyplot as plt
from attacks import attack
from misc_functions import (get_params,
                            convert_to_grayscale,
                            save_gradient_images)
from gradcam import GradCam
from guided_backprop import GuidedBackprop


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


if __name__ == '__main__':
    # Get params
    target_example = 2  # Snake
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_params(target_example,'AlexNet')
    attack_type = 'FGSM'

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


    plt.subplot(2,2,1)
    plt.imshow(guidedgrad)
    plt.title('Guided Grad Cam')
    plt.subplot(2,2,2)
    plt.imshow(grayguidedgrad[:,:,0])
    plt.title('Guided Grad Cam Grasycale')


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
    guidedgrad = save_gradient_images(cam_gb, 'Adversary_'+ file_name_to_export + '_GGrad_Cam')
    grayscale_cam_gb = convert_to_grayscale(cam_gb)
    grayguidedgrad = save_gradient_images(grayscale_cam_gb,'Adversary_'+ file_name_to_export + '_GGrad_Cam_gray')
    print('Guided grad cam completed')


    plt.subplot(2,2,3)
    plt.imshow(guidedgrad)
    plt.title('Adversary Guided Grad Cam')
    plt.subplot(2,2,4)
    plt.imshow(grayguidedgrad[:,:,0])
    plt.title('Adversary Guided Grad Cam Grasycale')

    plt.show()