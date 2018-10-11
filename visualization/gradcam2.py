"""
Created on Thu Oct 26 11:06:51 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage.measure import compare_ssim as ssim
from torch.nn import functional as F
from attacks import attack
from misc_functions import get_params, save_class_activation_on_image,prediction_reader

class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer,network,structure):
        self.network = network
        self.structure = structure
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None

        if self.network =="ResNet50" or self.structure =='ResNet50':
            i = 0
            for module in list(self.model.children())[:-1]:
                i += 1
                x = module(x)
                if i == self.target_layer:
                    x.register_hook(self.save_gradient)
                    conv_output = x
        else:
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            for module_pos, module in self.model.features._modules.items():
                # print("module_pos: ",module_pos," - Module:" , module)

                x = module(x)  # Forward
                if int(module_pos) == self.target_layer:
                    x.register_hook(self.save_gradient)
                    conv_output = x  # Save the convolution output on that layer

        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        conv_output, x = self.forward_pass_on_convolutions(x)
        # Forward pass on the classifier
        x = x.view(x.size(0), -1)  # Flatten
        if self.network == "ResNet50" or self.structure =="ResNet50":
            module  = list(self.model.children())[-1]
            x = x.view(x.size(0), -1)
            x = module(x)

        else:
            x = self.model.classifier(x)
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer,network,structure):
        self.network = network
        self.structure = structure
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer,self.network,self.structure)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        if torch.cuda.is_available():
            input_image = input_image.cuda()
        conv_output, model_output = self.extractor.forward_pass(input_image)

        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        one_hot_output = one_hot_output.to(device)
        one_hot_output[0][target_class] = 1

        # Zero grads

        if self.network == "ResNet50" or self.structure == 'ResNet50':
            for module in list(self.model.children())[:-1]:
                module.zero_grad()
            module = list(self.model.children())[-1]
            module.zero_grad()

        else:
            self.model.features.zero_grad()
            self.model.classifier.zero_grad()

        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)

        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.cpu().numpy()[0]

        # Get convolution outputs
        target = conv_output.data.cpu().numpy()[0]

        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient

        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)

        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = cv2.resize(cam, (224, 224))
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        return cam

#if __name__ == '__main__':
# Get params
def runGradCam2(choose_network = 'AlexNet',
               isTrained = True,
               training = "Normal",
               structure="",
               target_example = 3,
               attack_type = 'FGSM'):

    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_params(target_example, choose_network, isTrained, training, structure)

    # Grad cam
    if choose_network == "ResNet50" or structure == 'ResNet50':
        grad_cam = GradCam(pretrained_model, target_layer=7, network=choose_network, structure=structure)
    elif choose_network == "AlexNet":
        grad_cam = GradCam(pretrained_model, target_layer=11, network=choose_network, structure=structure)
    elif choose_network == "VGG19" or structure == 'VGG19':
        grad_cam = GradCam(pretrained_model, target_layer=35, network=choose_network, structure=structure)



    # Generate cam mask
    cam = grad_cam.generate_cam(prep_img,target_class)
    # Save mask
    gray,color, result = save_class_activation_on_image(choose_network, original_image, cam, file_name_to_export)

    print('Grad cam completed')

        # Adversary:

    attack1 = attack(choose_network,attack_type,pretrained_model,original_image,file_name_to_export,target_class)
    adversarialpic,adversarial,advers_class,orig_pred,adver_pred ,diff = attack1.getstuff()

    orig_labs,orig_vals = prediction_reader(orig_pred,10,choose_network)
    adver_labs,adver_vals = prediction_reader(adver_pred,10,choose_network)
    indices = np.arange(len(orig_labs))

    # Generate cam mask on adversarial
    cam = grad_cam.generate_cam(adversarial,advers_class)
    # Save mask
    gray2,color2, result2 = save_class_activation_on_image(choose_network, original_image, cam, 'Adversary_'+file_name_to_export)
    print('Adversary Grad cam completed')


    # Generate cam mask on image but adversarial class
    cam = grad_cam.generate_cam(prep_img,advers_class)
    # Save mask
    gray3,color3, result3 = save_class_activation_on_image(choose_network, original_image, cam, 'NotSoNormie_'+file_name_to_export)

    print('Normie but Advers Grad cam completed')

    # Generate cam mask on image but adversarial class
    cam = grad_cam.generate_cam(adversarial,target_class)
    # Save mask
    gray4,color4, result4 = save_class_activation_on_image(choose_network, original_image, cam, 'InvNotSoNormie_'+file_name_to_export)

    print('Normie but Advers Grad cam completed')
    fig = plt.figure()
    fig.suptitle(file_name_to_export+' - '+attack_type+' - GradCam')
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    ax0 = fig.add_subplot(4,5,1)
    ax0.imshow(original_image)
    ax0.set_title('Original Image')
    ax1 = fig.add_subplot(4,5,2)
    ax1.imshow(gray)
    ax1.set_title('Cam Grasycale')
    ax2 = fig.add_subplot(4,5,3)
    ax2.imshow(color)
    ax2.set_title('Cam HeatMap')
    ax3 = fig.add_subplot(4,5,4)
    ax3.imshow(result)
    ax3.set_title('Cam Result')

    ax9 = fig.add_subplot(4,5,5)
    ax9.bar(indices,orig_vals,align='center', alpha=0.5)
    ax9.set_title('Orignial Image Predictions')
    ax9.set_xticks(indices)
    ax9.set_xticklabels(orig_labs,rotation = 45,ha="right")

    adversarial = cv2.cvtColor(np.uint8(adversarialpic), cv2.COLOR_BGR2RGB)
    ax12 = fig.add_subplot(4,5,6)
    ax12.imshow(adversarial)
    ax12.set_title('Adversary Image(SSIM = '+str(diff)+')')

    diff2 = ssim(gray, gray2,multichannel=True)
    label = ', SSIM with above: {:.3f}'
    ax4 = fig.add_subplot(4,5,7)
    ax4.imshow(gray2)
    ax4.set_title('Adversary Cam Grasycale'+label.format(diff2))
    ax5 = fig.add_subplot(4,5,8)
    ax5.imshow(color2)
    ax5.set_title('Adversary Cam HeatMap')
    ax6 = fig.add_subplot(4,5,9)
    ax6.imshow(result2)
    ax6.set_title('Adversary Cam Result')


    ax10 = fig.add_subplot(4,5,10)
    ax10.bar(indices,adver_vals,align='center', alpha=0.5)
    ax10.set_title('Adversary Image Predictions')
    ax10.set_xticks(indices)
    ax10.set_xticklabels(adver_labs,rotation = 45,ha="right")

    diff3 = ssim(gray2, gray3,multichannel=True)
    ax4 = fig.add_subplot(4,5,12)
    ax4.imshow(gray3)
    ax4.set_title('NotSoNormie Cam Grasycale'+label.format(diff3))
    diff3 = ssim(color2, color3,multichannel=True)
    ax5 = fig.add_subplot(4,5,13)
    ax5.imshow(color3)
    ax5.set_title('NotSoNormie Cam HeatMap'+label.format(diff3))
    diff3 = ssim(result2, result3,multichannel=True)
    ax6 = fig.add_subplot(4,5,14)
    ax6.imshow(result3)
    ax6.set_title('NotSoNormie Cam Result'+label.format(diff3))

    diff3 = ssim(gray, gray4,multichannel=True)
    ax4 = fig.add_subplot(4,5,17)
    ax4.imshow(gray4)
    ax4.set_title('Inverse NotSoNormie Cam Grasycale'+label.format(diff3))
    diff3 = ssim(color4, color,multichannel=True)
    ax5 = fig.add_subplot(4,5,18)
    ax5.imshow(color4)
    ax5.set_title('Inverse NotSoNormie Cam HeatMap'+label.format(diff3))
    diff3 = ssim(result4, result,multichannel=True)
    ax6 = fig.add_subplot(4,5,19)
    ax6.imshow(result4)
    ax6.set_title('Inverse NotSoNormie Cam Result'+label.format(diff3))

    fig.set_size_inches(30, 36)
    fig.tight_layout()
    if isTrained:
        train = 'Trained'
    else:
        train = 'UnTrained'
    fig.savefig('Concise Results/'+file_name_to_export+'_'+attack_type+
                '_GradCam2('+train+choose_network+')',dpi = 100)
    #return np.cov(gray,gray2)
    return original_image,gray,color,result,adversarial,gray2,color2,result2, indices,orig_labs,orig_vals,adver_labs,adver_vals
