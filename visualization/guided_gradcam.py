"""
Created on Thu Oct 23 11:27:15 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import numpy as np
from guided_backprop import GuidedBackprop
from matplotlib import pyplot as plt
import cv2
from attacks import attack
from misc_functions import (get_params,
                            convert_to_grayscale,
                            save_gradient_images,
                            prediction_reader)
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
                isTrained = True,
                training = "Normal",
                structure='',
                target_example = 3,
                attack_type = 'FGSM'):
    #if __name__ == '__main__':
    # Get params
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_params(target_example,choose_network,isTrained,training,structure)

    # Grad cam
    if choose_network == "ResNet50" or structure == 'ResNet50':
        gcv2 = GradCam(pretrained_model, target_layer=7,network = choose_network,structure=structure)
    elif choose_network == "AlexNet":
        gcv2 = GradCam(pretrained_model, target_layer=11,network = choose_network,structure=structure)
    elif choose_network == "VGG19" or structure == 'VGG19':
        gcv2 = GradCam(pretrained_model, target_layer=35,network = choose_network,structure=structure)


    # Generate cam mask

    cam = gcv2.generate_cam(prep_img, target_class)
    print('Grad cam completed')
    print('cam shape:',cam.shape)

    # Guided backprop
    GBP = GuidedBackprop(pretrained_model,choose_network,structure)
    # Get gradients
    guided_grads = GBP.generate_gradients(prep_img, target_class)
    print('Guided backpropagation completed')

    # Guided Grad cam
    cam_gb = guided_grad_cam(cam, guided_grads)
    guidedgrad = save_gradient_images(cam_gb, file_name_to_export + '_GGrad_Cam')
    grayscale_cam_gb = convert_to_grayscale(cam_gb)
    grayguidedgrad = save_gradient_images(grayscale_cam_gb, file_name_to_export + '_GGrad_Cam_gray')
    print('Guided grad cam completed')

    attack1 =  attack(choose_network,attack_type,pretrained_model,original_image,file_name_to_export,target_class)
    adversarialpic,adversarial,advers_class,orig_pred,adver_pred,diff = attack1.getstuff()

    orig_labs,orig_vals = prediction_reader(orig_pred,10,choose_network)
    adver_labs,adver_vals = prediction_reader(adver_pred,10,choose_network)
    indices = np.arange(len(orig_labs))

    cam = gcv2.generate_cam(adversarial, advers_class)
    print('Grad cam completed')

    # Guided backprop
    GBP = GuidedBackprop(pretrained_model,choose_network)
    # Get gradients
    guided_grads = GBP.generate_gradients(adversarial, advers_class)
    print('Guided backpropagation completed')

    # Guided Grad cam
    cam_gb = guided_grad_cam(cam, guided_grads)
    guidedgrad2 = save_gradient_images(cam_gb, 'Adversary_'+ file_name_to_export + '_GGrad_Cam')
    grayscale_cam_gb = convert_to_grayscale(cam_gb)
    grayguidedgrad2 = save_gradient_images(grayscale_cam_gb,'Adversary_'+ file_name_to_export + '_GGrad_Cam_gray')
    print('Adversary Guided grad cam completed')


    fig = plt.figure()
    fig.suptitle(file_name_to_export+' - '+attack_type+' - Guided GradCam')
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    ax0 = fig.add_subplot(2,4,1)
    ax0.imshow(original_image)
    ax0.set_title('Original Image')
    ax1 = fig.add_subplot(2,4,2)
    ax1.imshow(guidedgrad)
    ax1.set_title('Guided Grad Cam')
    ax2 = fig.add_subplot(2,4,3)
    ax2.imshow(grayguidedgrad[:,:,0])
    ax2.set_title('Guided Grad Cam Grasycale')

    ax9 = fig.add_subplot(2,4,4)
    ax9.bar(indices,orig_vals,align='center', alpha=0.5)
    ax9.set_title('Orignial Image Predictions')
    ax9.set_xticks(indices)
    ax9.set_xticklabels(orig_labs,rotation = 45,ha="right")

    adversarial = np.uint8(adversarialpic)
    adversarial = cv2.cvtColor(adversarial, cv2.COLOR_BGR2RGB)
    ax12 = fig.add_subplot(2,4,5)
    ax12.imshow(adversarial)
    ax12.set_title('Adversary Image(SSIM = '+str(diff)+')')

    ax3 = fig.add_subplot(2,4,6)
    ax3.imshow(guidedgrad2)
    ax3.set_title('Adversary Guided Grad Cam')
    ax4 = fig.add_subplot(2,4,7)
    ax4.imshow(grayguidedgrad2[:,:,0])
    ax4.set_title('Adversary Guided Grad Cam Grasycale')

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
    fig.savefig('Concise Results/'+file_name_to_export+'_'+attack_type+'_Guided GradCam('+
                train+choose_network+')',dpi = 100)

    #return np.cov(grayguidedgrad[:,:,0],grayguidedgrad2[:,:,0])
    return original_image, guidedgrad, grayguidedgrad[:,:,0], adversarial, guidedgrad2, grayguidedgrad2[:,:,0],\
           indices,orig_labs,orig_vals,adver_labs,adver_vals