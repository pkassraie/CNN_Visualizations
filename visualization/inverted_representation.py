"""
NOT UPDATED FOR RESNET50
"""
import cv2
import torch
import numpy as np
from torch.autograd import Variable
from torch.optim import SGD
from matplotlib import pyplot as plt
from attacks import attack
from misc_functions import get_params, recreate_image,prediction_reader



class InvertedRepresentation():
    def __init__(self, model,network):
        self.network = network
        self.model = model
        self.model.eval()

    def alpha_norm(self, input_matrix, alpha):
        """
            Converts matrix to vector then calculates the alpha norm
        """
        alpha_norm = ((input_matrix.view(-1))**alpha).sum()
        return alpha_norm

    def total_variation_norm(self, input_matrix, beta):
        """
            Total variation norm is the second norm in the paper
            represented as R_V(x)
        """
        to_check = input_matrix[:, :-1, :-1]  # Trimmed: right - bottom
        one_bottom = input_matrix[:, 1:, :-1]  # Trimmed: top - right
        one_right = input_matrix[:, :-1, 1:]  # Trimmed: top - right
        total_variation = (((to_check - one_bottom)**2 +
                            (to_check - one_right)**2)**(beta/2)).sum()
        return total_variation

    def euclidian_loss(self, org_matrix, target_matrix):
        """
            Euclidian loss is the main loss function in the paper
            ||fi(x) - fi(x_0)||_2^2& / ||fi(x_0)||_2^2
        """
        distance_matrix = target_matrix - org_matrix
        euclidian_distance = self.alpha_norm(distance_matrix, 2)
        normalized_euclidian_distance = euclidian_distance / self.alpha_norm(org_matrix, 2)
        return normalized_euclidian_distance

    def get_output_from_specific_layer(self, x, layer_id):
        """
            Saves the output after a forward pass until nth layer
            This operation could be done with a forward hook too
            but this one is simpler (I think)
        """
        layer_output = None
        if self.network == "ResNet50":
            i = 0
            for module in list(self.model.children())[:-1]:
                x = module(x)
                if i == int(layer_id) :
                    layer_output = x[0]
                    break
                i += 1
        else:
            for index, layer in enumerate(self.model.features):
                x = layer(x)
                if str(index) == str(layer_id):
                    layer_output = x[0]
                    break
        return layer_output

    def generate_inverted_image_specific_layer(self, input_image, img_size, advers, target_layer=3):

        if advers ==True:
            name = ''
        else:
            name = '_Adversarial'
        # Generate a random image which we will optimize
        opt_img = Variable(1e-1 * torch.randn(1, 3, img_size, img_size), requires_grad=True)
        # Define optimizer for previously created image
        optimizer = SGD([opt_img], lr=1e4, momentum=0.9)
        # Get the output from the model after a forward pass until target_layer
        # with the input image (real image, NOT the randomly generated one)
        input_image_layer_output = \
            self.get_output_from_specific_layer(input_image, target_layer)

        # Alpha regularization parametrs
        # Parameter alpha, which is actually sixth norm
        alpha_reg_alpha = 6
        # The multiplier, lambda alpha
        alpha_reg_lambda = 1e-7

        # Total variation regularization parameters
        # Parameter beta, which is actually second norm
        tv_reg_beta = 2
        # The multiplier, lambda beta
        tv_reg_lambda = 1e-8

        for i in range(251): #Increase later
            optimizer.zero_grad()
            # Get the output from the model after a forward pass until target_layer
            # with the generated image (randomly generated one, NOT the real image)
            output = self.get_output_from_specific_layer(opt_img, target_layer)
            # Calculate euclidian loss
            euc_loss = 1e-1 * self.euclidian_loss(input_image_layer_output.detach(), output)
            # Calculate alpha regularization
            reg_alpha = alpha_reg_lambda * self.alpha_norm(opt_img, alpha_reg_alpha)
            # Calculate total variation regularization
            reg_total_variation = tv_reg_lambda * self.total_variation_norm(opt_img,
                                                                            tv_reg_beta)
            # Sum all to optimize
            loss = euc_loss + reg_alpha + reg_total_variation
            # Step
            loss.backward()
            optimizer.step()
            # Generate image every 50 iterations
            if i % 50 == 0:
                print('Iteration:', str(i), 'Loss:', loss.data.numpy())
                x = recreate_image(opt_img)
                cv2.imwrite('results/Inv_Image_Layer_' + str(target_layer) +
                            '_Iteration_' + str(i)+ name + '.jpg', x)
            # Reduce learning rate every 40 iterations
            if i % 40 == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 1/10
        return x;

def runInvRep(choose_network = 'AlexNet',
              isTrained = True,
              training = "Normal",
              structure="ResNet50",
              target_example = 3,
              target_layer = 0,
              attack_type = 'FGSM'):

    #if __name__ == '__main__':
        # Get params
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_params(target_example,choose_network,isTrained,training,structure)

    inverted_representation = InvertedRepresentation(pretrained_model,choose_network)
    image_size = original_image.shape[0]  # width & height

    if target_layer == 0:
        if choose_network == "AlexNet":
            target_layer = 11
        elif choose_network == "VGG19":
            target_layer = 36
        elif choose_network == "ResNet50":
            target_layer = 8

    cleanres = inverted_representation.generate_inverted_image_specific_layer(prep_img,
                                                                   image_size,
                                                                   False,
                                                                   target_layer)

    attack1 = attack(choose_network,attack_type,pretrained_model,original_image,
                                                           file_name_to_export,target_class)
    adversarialpic,adversarial,advers_class,orig_pred,adver_pred,diff = attack1.getstuff()

    orig_labs,orig_vals = prediction_reader(orig_pred,10,choose_network=choose_network)
    adver_labs,adver_vals = prediction_reader(adver_pred,10,choose_network=choose_network)
    indices = np.arange(len(orig_labs))

    adversres = inverted_representation.generate_inverted_image_specific_layer(adversarial,
                                                                   image_size,
                                                                   True,
                                                                   target_layer)
    fig = plt.figure()
    fig.suptitle(file_name_to_export+' - '+attack_type+' - Inverted Representation')

    ax0 = fig.add_subplot(2,3,1)
    ax0.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    ax0.set_title('Original Image')

    ax1 = fig.add_subplot(2,3,2)
    ax1.imshow(cleanres)
    ax1.set_title('Normal Inverted Repres')

    ax9 = fig.add_subplot(2,3,3)
    ax9.bar(indices,orig_vals,align='center', alpha=0.5)
    ax9.set_title('Orignial Image Predictions')
    ax9.set_xticks(indices)
    ax9.set_xticklabels(orig_labs,rotation = 45,ha="right")

    adversarial = np.uint8(adversarialpic)
    ax12 = fig.add_subplot(2,3,4)
    ax12.imshow(cv2.cvtColor(adversarial, cv2.COLOR_BGR2RGB))
    ax12.set_title('Adversary Image(SSIM = '+str(diff)+')')

    ax2 = fig.add_subplot(2,3,5)
    ax2.imshow(adversres)
    ax2.set_title('Adversary Inverted Repres')

    ax10 = fig.add_subplot(2,3,6)
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
    fig.savefig('Concise Results/'+file_name_to_export+'_'+attack_type+'_InvertedRep('+train+choose_network+')',dpi = 100)