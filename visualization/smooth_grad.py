import numpy as np
import cv2
import torch
from matplotlib import pyplot as plt
from torch.autograd import Variable

from attacks import attack
from misc_functions import (get_params,
                            convert_to_grayscale,
                            save_gradient_images,
                            prediction_reader)
from visualization.vanilla_backprop import VanillaBackprop

def generate_smooth_grad(Backprop, prep_img, target_class, param_n, param_sigma_multiplier):
    """
        Generates smooth gradients of given Backprop type. You can use this with both vanilla
        and guided backprop
    Args:
        Backprop (class): Backprop type
        prep_img (torch Variable): preprocessed image
        target_class (int): target class of imagenet
        param_n (int): Amount of images used to smooth gradient
        param_sigma_multiplier (int): Sigma multiplier when calculating std of noise
    """
    # Generate an empty image/matrix
    smooth_grad = np.zeros(prep_img.size()[1:])

    mean = 0
    sigma = param_sigma_multiplier / (torch.max(prep_img) - torch.min(prep_img)).data[0]
    for x in range(param_n):
        # Generate noise
        noise = Variable(prep_img.data.new(prep_img.size()).normal_(mean, sigma**2))
        # Add noise to the image
        noisy_img = prep_img + noise
        # Calculate gradients
        vanilla_grads = Backprop.generate_gradients(noisy_img, target_class)
        # Add gradients to smooth_grad
        smooth_grad = smooth_grad + vanilla_grads
    # Average it out
    smooth_grad = smooth_grad / param_n
    return smooth_grad

def runsmoothGrad(choose_network = 'AlexNet',
                 isTrained = True,
                 training = "Normal",
                 structure="ResNet50",
                 target_example = 3,
                 attack_type = 'FGSM'):

#if __name__ == '__main__':
    # Get params
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_params(target_example,choose_network,isTrained,training, structure)

    VBP = VanillaBackprop(pretrained_model,choose_network,structure)
    # GBP = GuidedBackprop(pretrained_model)  # if you want to use GBP dont forget to
    # change the parametre in generate_smooth_grad

    param_n = 50
    param_sigma_multiplier = 4
    smooth_grad = generate_smooth_grad(VBP,  # ^This parameter
                                       prep_img,
                                       target_class,
                                       param_n,
                                       param_sigma_multiplier)

    # Save colored gradients
    colorgrads = save_gradient_images(smooth_grad, file_name_to_export + '_SmoothGrad_color')
    # Convert to grayscale
    grayscale_smooth_grad = convert_to_grayscale(smooth_grad)
    # Save grayscale gradients
    graygrads = save_gradient_images(grayscale_smooth_grad, file_name_to_export + '_SmoothGrad_gray')
    print('Smooth grad completed')

    # Now the attack:
    attack1 = attack(attack_type,pretrained_model,
                                                           original_image,file_name_to_export,target_class)
    adversarialpic,adversarial,advers_class,orig_pred,adver_pred,diff = attack1.getstuff()

    orig_labs,orig_vals = prediction_reader(orig_pred,10,choose_network)
    adver_labs,adver_vals = prediction_reader(adver_pred,10,choose_network)
    indices = np.arange(len(orig_labs))

    smooth_grad = generate_smooth_grad(VBP,  # ^This parameter
                                       adversarial,
                                       advers_class,
                                       param_n,
                                       param_sigma_multiplier)

    # Save colored gradients
    colorgrads2 = save_gradient_images(smooth_grad, 'Adversary_'+file_name_to_export + '_SmoothGrad_color')
    # Convert to grayscale
    grayscale_smooth_grad = convert_to_grayscale(smooth_grad)
    # Save grayscale gradients
    graygrads2 = save_gradient_images(grayscale_smooth_grad, 'Adversary_'+file_name_to_export + '_SmoothGrad_gray')
    print('Adversary Smooth grad completed')


    fig = plt.figure()
    fig.suptitle(file_name_to_export+' - '+attack_type+' - Smooth BackProp')
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    ax0 = fig.add_subplot(2,4,1)
    ax0.imshow(original_image)
    ax0.set_title('Original Image')

    ax1 = fig.add_subplot(2,4,2)
    ax1.imshow(colorgrads)
    ax1.set_title('Smooth BP')
    ax2 = fig.add_subplot(2, 4, 3)
    ax2.imshow(graygrads[:,:,0])
    ax2.set_title('Smooth BP Gray')

    ax9 = fig.add_subplot(2,4,4)
    ax9.bar(indices,orig_vals,align='center', alpha=0.5)
    ax9.set_title('Orignial Image Predictions')
    ax9.set_xticks(indices)
    ax9.set_xticklabels(orig_labs,rotation = 45,ha="right")

    adversarial = cv2.cvtColor(np.uint8(adversarialpic), cv2.COLOR_BGR2RGB)
    ax12 = fig.add_subplot(2,4,5)
    ax12.imshow(adversarial)
    ax12.set_title('Adversary Image(SSIM = '+str(diff)+')')


    ax3 = fig.add_subplot(2, 4, 6)
    ax3.imshow(colorgrads2)
    ax3.set_title('Adversary Smooth BP')
    ax4 = fig.add_subplot(2, 4, 7)
    ax4.imshow(graygrads2[:,:,0])
    ax4.set_title('Adversary Smooth BP Gray')

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

    fig.savefig('Concise Results/'+file_name_to_export+'_'+attack_type+'_SmoothGrad('+train+choose_network+')',dpi = 100)

    #return np.cov(graygrads[:,:,0],graygrads2[:,:,0])
    return original_image,colorgrads,graygrads[:,:,0],adversarial,colorgrads2,graygrads2[:,:,0],\
           indices,orig_labs,orig_vals,adver_labs,adver_vals