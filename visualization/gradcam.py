"""
Created on Thu Oct 26 11:06:51 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from attacks import attack
from misc_functions import get_params, save_class_activation_on_image,prediction_reader

class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer,network):
        self.network = network
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

        if self.network =="ResNet50":
            i = 0
            for module in list(self.model.children())[:-1]:
                i += 1
                x = module(x)
                if i == self.target_layer:
                    x.register_hook(self.save_gradient)
                    conv_output = x

        else:
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
        conv_output, x = self.forward_pass_on_convolutions(x)

        # Forward pass on the classifier
        x = x.view(x.size(0), -1)  # Flatten
        if self.network == "ResNet50":
            module  = list(self.model.children())[-1]
            x = module(x)

        else:
            x = self.model.classifier(x)
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer,network):
        self.network = network
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer,self.network)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1

        # Zero grads
        if self.network == "ResNet50":
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
        guided_gradients = self.extractor.gradients.data.numpy()[0]

        # Get convolution outputs
        target = conv_output.data.numpy()[0]

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
def runGradCam(choose_network = 'AlexNet',
               isTrained = True,
                 target_example = 3,
                 attack_type = 'FGSM'):

    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_params(target_example,choose_network,isTrained)

    # Grad cam
    if choose_network == "ResNet50":
        grad_cam = GradCam(pretrained_model, target_layer=7,network = choose_network)
    elif choose_network == "AlexNet":
        grad_cam = GradCam(pretrained_model, target_layer=11,network = choose_network)
    elif choose_network == "VGG19":
        grad_cam = GradCam(pretrained_model, target_layer=35,network = choose_network)


    # Generate cam mask
    cam = grad_cam.generate_cam(prep_img,target_class)
    # Save mask
    gray,color, result = save_class_activation_on_image(original_image, cam, file_name_to_export)

    print('Grad cam completed')

        # Adversary:

    attack1 = attack(attack_type,pretrained_model,original_image,file_name_to_export,target_class)
    adversarialpic,adversarial,advers_class,orig_pred,adver_pred ,diff = attack1.getstuff()

    orig_labs,orig_vals = prediction_reader(orig_pred,10)
    adver_labs,adver_vals = prediction_reader(adver_pred,10)
    indices = np.arange(len(orig_labs))

    # Generate cam mask
    cam = grad_cam.generate_cam(adversarial,advers_class)
    # Save mask
    gray2,color2, result2 = save_class_activation_on_image(original_image, cam, 'Adversary_'+file_name_to_export)
    print('Adversary Grad cam completed')



    fig = plt.figure()
    fig.suptitle(file_name_to_export+' - '+attack_type+' - GradCam')

    ax0 = fig.add_subplot(2,5,1)
    ax0.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    ax0.set_title('Original Image')
    ax1 = fig.add_subplot(2,5,2)
    ax1.imshow(gray)
    ax1.set_title('Cam Grasycale')
    ax2 = fig.add_subplot(2,5,3)
    ax2.imshow(color)
    ax2.set_title('Cam HeatMap')
    ax3 = fig.add_subplot(2,5,4)
    ax3.imshow(result)
    ax3.set_title('Cam Result')

    ax9 = fig.add_subplot(2,5,5)
    ax9.bar(indices,orig_vals,align='center', alpha=0.5)
    ax9.set_title('Orignial Image Predictions')
    ax9.set_xticks(indices)
    ax9.set_xticklabels(orig_labs,rotation = 45,ha="right")

    adversarial = np.uint8(adversarialpic)
    ax12 = fig.add_subplot(2,5,6)
    ax12.imshow(cv2.cvtColor(adversarial, cv2.COLOR_BGR2RGB))
    ax12.set_title('Adversary Image(SSIM = '+str(diff)+')')

    ax4 = fig.add_subplot(2,5,7)
    ax4.imshow(gray2)
    ax4.set_title('Adversary Cam Grasycale')
    ax5 = fig.add_subplot(2,5,8)
    ax5.imshow(color2)
    ax5.set_title('Adversary Cam HeatMap')
    ax6 = fig.add_subplot(2,5,9)
    ax6.imshow(result2)
    ax6.set_title('Adversary Cam Result')

    ax10 = fig.add_subplot(2,5,10)
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
                '_GradCam('+train+choose_network+')',dpi = 100)

    return np.cov(gray,gray2)