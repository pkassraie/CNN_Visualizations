import torch
from torch.autograd import Variable
from torchvision import models
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import compare_ssim as ssim
from misc_functions import get_params, prediction_reader
from attacks import attack
from customization.loadModel import loadModel


use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor


def tv_norm(input, tv_beta):
    img = input[0, 0, :]
    row_grad = torch.mean(torch.abs((img[:-1, :] - img[1:, :])).pow(tv_beta))
    col_grad = torch.mean(torch.abs((img[:, :-1] - img[:, 1:])).pow(tv_beta))
    return row_grad + col_grad


def preprocess_image(img,network = '',training = ''):
    if network == 'Custom':
        if training == 'Normal':
            means = [0.4914, 0.4822, 0.4465]
            stds = [0.2023, 0.1994, 0.2010]
            preprocessed_img = img.copy()[:, :, ::-1]
            for i in range(3):
                preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
                preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
            preprocessed_img = \
                np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
        else:
            preprocessed_img = img.copy()[:, :, ::-1]
            preprocessed_img = np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    else:
        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]

        preprocessed_img = img.copy()[:, :, ::-1]
        for i in range(3):
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
        preprocessed_img = \
            np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))

    if use_cuda:
        preprocessed_img_tensor = torch.from_numpy(preprocessed_img).cuda()
    else:
        preprocessed_img_tensor = torch.from_numpy(preprocessed_img)

    preprocessed_img_tensor.unsqueeze_(0)
    return Variable(preprocessed_img_tensor, requires_grad=False)


def save(mask, img, blurred, name):
    mask = mask.cpu().data.numpy()[0]
    mask = np.transpose(mask, (1, 2, 0))

    mask = (mask - np.min(mask)) / np.max(mask)
    mask = 1 - mask
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)

    heatmap = np.float32(heatmap) / 255
    cam = 1.0 * heatmap + np.float32(img) / 255
    cam = cam / np.max(cam)

    img = np.float32(img) / 255
    perturbated = np.multiply(1 - mask, img) + np.multiply(mask, blurred)


    return np.uint8(255 * heatmap), np.uint8(255 * mask), np.uint8(255 * cam)


def numpy_to_torch(img, requires_grad=True):
    if len(img.shape) < 3:
        output = np.float32([img])
    else:
        output = np.transpose(img, (2, 0, 1))

    output = torch.from_numpy(output)
    if use_cuda:
        output = output.cuda()

    output.unsqueeze_(0)
    v = Variable(output, requires_grad=requires_grad)
    return v


def load_model(choose_network, trained,training, structure):
    if choose_network == 'VGG19':
        model = models.vgg19(pretrained=trained)
        model.eval()
        if use_cuda:
            model.cuda()

        for p in model.features.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = False

    elif choose_network == 'AlexNet':
        model = models.alexnet(pretrained=trained)
        model.eval()
        if use_cuda:
            model.cuda()

        for p in model.features.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = False

    elif choose_network == 'ResNet50':
        model = models.resnet50(pretrained=trained)
        model.eval()
        if use_cuda:
            model.cuda()

        for param in model.parameters():
            param.requires_grad = False

    elif choose_network == 'Custom':
        model = loadModel(training,structure)
        model.eval()
        if use_cuda:
            model.cuda()

        for param in model.parameters():
            param.requires_grad = False

    return model


def prep_img(original_img,network,training):
    original_img = cv2.resize(original_img, (224, 224))
    img = np.float32(original_img) / 255
    blurred_img1 = cv2.GaussianBlur(img, (11, 11), 5)
    blurred_img2 = np.float32(cv2.medianBlur(np.uint8(original_img), 11)) / 255
    blurred_img_numpy = (blurred_img1 + blurred_img2) / 2
    mask_init = np.ones((28, 28), dtype=np.float32)

    # Convert to torch variables
    img = preprocess_image(img,network,training)
    blurred_img = preprocess_image(blurred_img2,network,training)
    mask = numpy_to_torch(mask_init)
    return img, blurred_img, mask, blurred_img_numpy


def optimizeMask(model, iters, mask, img, blurred_img):
    # Hyper parameters.
    tv_beta = 3
    learning_rate = 0.1
    max_iterations = iters
    l1_coeff = 0.01
    tv_coeff = 0.2
    if use_cuda:
        upsample = torch.nn.Upsample(size=(224, 224)).cuda()
    else:
        upsample = torch.nn.Upsample(size=(224, 224))

    optimizer = torch.optim.Adam([mask], lr=learning_rate)
    target = torch.nn.Softmax(dim=1)(model(img))
    category = np.argmax(target.cpu().data.numpy())

    for i in range(max_iterations):
        upsampled_mask = upsample(mask)
        # The single channel mask is used with an RGB image,
        # so the mask is duplicated to have 3 channel,
        upsampled_mask = \
            upsampled_mask.expand(1, 3, upsampled_mask.size(2), upsampled_mask.size(3))

        # Use the mask to perturbated the input image.
        perturbated_input = img.mul(upsampled_mask) + \
                            blurred_img.mul(1 - upsampled_mask)
        noise = np.zeros((224, 224, 3), dtype=np.float32)
        cv2.randn(noise, 0, 0.2)
        noise = numpy_to_torch(noise)
        perturbated_input = perturbated_input + noise

        outputs = torch.nn.Softmax(dim=1)(model(perturbated_input))
        loss = l1_coeff * torch.mean(torch.abs(1 - mask)) + \
               tv_coeff * tv_norm(mask, tv_beta) + outputs[0, category]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("step:", i, " Loss:", loss)

        # Optional: clamping seems to give better results
        mask.data.clamp_(0, 1)

    return upsample(mask)

def optimizeMaskadvers( model, iters, mask, img, blurred_img,advers_class):
    # Hyper parameters.
    tv_beta = 3
    learning_rate = 0.1
    max_iterations = iters
    l1_coeff = 0.01
    tv_coeff = 0.2

    if use_cuda:
        upsample = torch.nn.Upsample(size=(224, 224)).cuda()
    else:
        upsample = torch.nn.Upsample(size=(224, 224))
    optimizer = torch.optim.Adam([mask], lr=learning_rate)

    category = advers_class

    for i in range(max_iterations):
        upsampled_mask = upsample(mask)
        # The single channel mask is used with an RGB image,
        # so the mask is duplicated to have 3 channel,
        upsampled_mask = \
            upsampled_mask.expand(1, 3, upsampled_mask.size(2), upsampled_mask.size(3))

        # Use the mask to perturbated the input image.
        perturbated_input = img.mul(upsampled_mask) + \
                            blurred_img.mul(1 - upsampled_mask)

        noise = np.zeros((224, 224, 3), dtype=np.float32)

        cv2.randn(noise, 0, 0.2)
        noise = numpy_to_torch(noise)
        perturbated_input = perturbated_input + noise

        outputs = torch.nn.Softmax(dim=1)(model(perturbated_input))
        loss = l1_coeff * torch.mean(torch.abs(1 - mask)) + \
               tv_coeff * tv_norm(mask, tv_beta) + outputs[0, category]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("step:", i, " Loss:", loss)

        # Optional: clamping seems to give better results
        mask.data.clamp_(0, 1)

    return upsample(mask)

def runExplain2(choose_network='AlexNet',
               isTrained=True,
               training = "Normal",
               structure="ResNet50",
               target_example=0,
               iters=5,
               attack_type='FGSM'):

    model = load_model(choose_network,isTrained,training,structure)

    (original_img, _, target_class, file_name_to_export, pretrained_model) = get_params(target_example,
                                                                                        choose_network, isTrained,
                                                                                        training,structure)


    # Natural Image:
    img, blurred_img, mask, blurred_img_numpy = prep_img(original_img,choose_network,training)
    upsampled_mask = optimizeMask(model, iters, mask, img, blurred_img)

    heat1,mask1, cam1 = save(upsampled_mask, original_img, blurred_img_numpy, file_name_to_export)
    print("Interpretable Explanations Completed")


    attack1 = attack(choose_network,attack_type, pretrained_model, original_img, file_name_to_export, target_class)
    adversarialpic, adversarial,advers_class, orig_pred, adver_pred, diff = attack1.getstuff()

    orig_labs, orig_vals = prediction_reader(orig_pred, 10,choose_network)
    adver_labs, adver_vals = prediction_reader(adver_pred, 10,choose_network)
    indices = np.arange(len(orig_labs))

    # Adversary:
    img, blurred_img, mask, blurred_img_numpy = prep_img(adversarialpic,choose_network,training)
    upsampled_mask = optimizeMaskadvers(model, iters, mask, img, blurred_img,advers_class)
    heat2,mask2,cam2 = save(upsampled_mask, original_img, blurred_img_numpy, 'Adversarial_' + file_name_to_export)
    print("Adversary Interpretable Explanations Completed")

    #NotSoNormie
    img, blurred_img, mask, blurred_img_numpy = prep_img(original_img,choose_network,training)
    upsampled_mask = optimizeMaskadvers(model, iters, mask, img, blurred_img,advers_class)
    heat3,mask3,cam3 = save(upsampled_mask, original_img, blurred_img_numpy, 'NotSoNormie_' + file_name_to_export)
    print("NotSoNormie Interpretable Explanations Completed")

    #Inv NotSoNormie
    img, blurred_img, mask, blurred_img_numpy = prep_img(adversarialpic,choose_network,training)
    upsampled_mask = optimizeMask(model, iters, mask, img, blurred_img)
    heat4,mask4,cam4 = save(upsampled_mask, original_img, blurred_img_numpy, 'InvNotSoNormie_' + file_name_to_export)
    print("Inverse NotSoNormie Interpretable Explanations Completed")

    # Ploting:
    fig = plt.figure()
    fig.suptitle(file_name_to_export + ' - ' + attack_type + ' - Interpretable Explanations')

    ax11 = fig.add_subplot(4,5, 1)
    ax11.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    ax11.set_title('Original Image')

    ax1 = fig.add_subplot(4,5, 2)
    ax1.imshow(cv2.cvtColor(heat1, cv2.COLOR_BGR2RGB))
    ax1.set_title('Learned Mask Color')

    ax2 = fig.add_subplot(4,5, 3)
    ax2.imshow(mask1[:,:,0])
    ax2.set_title('Learned Mask Gray')

    ax3 = fig.add_subplot(4,5, 4)
    ax3.imshow(cv2.cvtColor(cam1, cv2.COLOR_BGR2RGB))
    ax3.set_title('Cam Result')

    ax9 = fig.add_subplot(4,5, 5)
    ax9.bar(indices, orig_vals, align='center', alpha=0.5)
    ax9.set_title('Orignial Image Predictions')
    ax9.set_xticks(indices)
    ax9.set_xticklabels(orig_labs, rotation=45, ha="right")
    ax12 = fig.add_subplot(4,5, 6)
    ax12.imshow(cv2.cvtColor(np.uint8(adversarialpic), cv2.COLOR_BGR2RGB))
    ax12.set_title('Adversary Image(SSIM = ' + str(diff) + ')')

    label = ', SSIM with above:{:.3f}'

    diff = ssim(heat1, heat2,multichannel=True)
    ax5 = fig.add_subplot(4,5, 7)
    ax5.imshow(cv2.cvtColor(heat2, cv2.COLOR_BGR2RGB))
    ax5.set_title('Adversary Mask Color'+label.format(diff))

    diff = ssim(mask1[:,:,0], mask2[:,:,0],multichannel=True)
    ax6 = fig.add_subplot(4,5, 8)
    ax6.imshow(mask2[:,:,0])
    ax6.set_title('Adversary Mask Gray'+label.format(diff))

    diff = ssim(cam2,cam1,multichannel=True)
    cam2 = cv2.cvtColor(cam2, cv2.COLOR_BGR2RGB)
    ax7 = fig.add_subplot(4,5, 9)
    ax7.imshow(cam2)
    ax7.set_title('Adversary Cam Result'+label.format(diff))

    ax10 = fig.add_subplot(4,5, 10)
    ax10.bar(indices, adver_vals, align='center', alpha=0.5)
    ax10.set_title('Adversary Image Predictions')
    ax10.set_xticks(indices)
    ax10.set_xticklabels(adver_labs, rotation=45, ha="right")

    diff = ssim(heat3, heat2,multichannel=True)
    ax5 = fig.add_subplot(4,5, 12)
    ax5.imshow(cv2.cvtColor(heat3, cv2.COLOR_BGR2RGB))
    ax5.set_title('NotSoNormie Mask Color'+label.format(diff))

    diff = ssim(mask3[:,:,0], mask2[:,:,0],multichannel=True)
    ax6 = fig.add_subplot(4,5, 13)
    ax6.imshow(mask3[:,:,0])
    ax6.set_title('NotSoNormie Mask Gray'+label.format(diff))

    cam3 = cv2.cvtColor(cam3, cv2.COLOR_BGR2RGB)
    diff = ssim(cam3, cam2,multichannel=True)
    ax7 = fig.add_subplot(4,5, 14)
    ax7.imshow(cam3)
    ax7.set_title('NotSoNormie Cam Result'+label.format(diff))

    diff = ssim(heat4, heat1,multichannel=True)
    ax5 = fig.add_subplot(4,5, 17)
    ax5.imshow(cv2.cvtColor(heat4, cv2.COLOR_BGR2RGB))
    ax5.set_title('Inv NotSoNormie Mask Color'+label.format(diff))

    diff = ssim(mask4[:,:,0], mask1[:,:,0],multichannel=True)
    ax6 = fig.add_subplot(4,5, 18)
    ax6.imshow(mask4[:,:,0])
    ax6.set_title('Inv NotSoNormie Mask Gray'+label.format(diff))

    cam3 = cv2.cvtColor(cam4, cv2.COLOR_BGR2RGB)
    diff = ssim(cam1, cam4,multichannel=True)
    ax7 = fig.add_subplot(4,5, 19)
    ax7.imshow(cam4)
    ax7.set_title('Inv NotSoNormie Cam Result'+label.format(diff))

    fig.set_size_inches(32, 36)
    fig.tight_layout()
    if isTrained:
        train = 'Trained'
    else:
        train = 'UnTrained'
    fig.savefig('Concise Results/' + file_name_to_export + '_' + attack_type +
                '_InterpExp2'+
                '_'+str(iters)+'iters(' + train + choose_network + ')', dpi=100)

    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    heat = cv2.cvtColor(heat1, cv2.COLOR_BGR2RGB)
    cam = cv2.cvtColor(cam1, cv2.COLOR_BGR2RGB)
    adversarialpic = cv2.cvtColor(np.uint8(adversarialpic), cv2.COLOR_BGR2RGB)
    return original_img,heat, mask,cam,\
           adversarialpic,heat2,mask2, cam2,\
           indices,orig_labs,orig_vals,adver_labs,adver_vals