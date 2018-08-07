import torch
from torch.autograd import Variable
from torchvision import models
import cv2
import numpy as np
from matplotlib import pyplot as plt
from misc_functions import get_params, prediction_reader
from attacks import attack

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor


def tv_norm(input, tv_beta):
    img = input[0, 0, :]
    row_grad = torch.mean(torch.abs((img[:-1, :] - img[1:, :])).pow(tv_beta))
    col_grad = torch.mean(torch.abs((img[:, :-1] - img[:, 1:])).pow(tv_beta))
    return row_grad + col_grad


def preprocess_image(img):
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

    cv2.imwrite('results/' + name + "_Explain_Perturb.jpg", np.uint8(255 * perturbated))
    cv2.imwrite('results/' + name + "_Explain_Heatmap.jpg", np.uint8(255 * heatmap))
    cv2.imwrite('results/' + name + "_Explain_Mask.jpg", np.uint8(255 * mask))
    cv2.imwrite('results/' + name + "_Explain_Cam.jpg", np.uint8(255 * cam))


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


def load_model(choose_network='VGG19', trained=True):
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

    return model


def prep_img(original_img):
    original_img = cv2.resize(original_img, (224, 224))
    img = np.float32(original_img) / 255
    blurred_img1 = cv2.GaussianBlur(img, (11, 11), 5)
    blurred_img2 = np.float32(cv2.medianBlur(np.uint8(original_img), 11)) / 255
    blurred_img_numpy = (blurred_img1 + blurred_img2) / 2
    mask_init = np.ones((28, 28), dtype=np.float32)

    # Convert to torch variables
    img = preprocess_image(img)
    blurred_img = preprocess_image(blurred_img2)
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


def runExplain(choose_network='AlexNet',
               isTrained=True,
               target_example=0,
               iters=5,
               attack_type='FGSM'):
    model = load_model(choose_network, isTrained)

    (original_img, _, target_class, file_name_to_export, pretrained_model) = get_params(target_example,
                                                                                        choose_network, isTrained)

    attack1 = attack(attack_type, pretrained_model, original_img, file_name_to_export, target_class)
    adversarialpic, adversarial,advers_class, orig_pred, adver_pred, diff = attack1.getstuff()

    orig_labs, orig_vals = prediction_reader(orig_pred, 10)
    adver_labs, adver_vals = prediction_reader(adver_pred, 10)
    indices = np.arange(len(orig_labs))

    # Natural Image:
    img, blurred_img, mask, blurred_img_numpy = prep_img(original_img)
    upsampled_mask = optimizeMask(model, iters, mask, img, blurred_img)
    save(upsampled_mask, original_img, blurred_img_numpy, file_name_to_export)
    print("Interpretable Explanations Completed")

    # Adversary:
    img, blurred_img, mask, blurred_img_numpy = prep_img(adversarialpic)
    upsampled_mask = optimizeMask(model, iters, mask, img, blurred_img)
    save(upsampled_mask, original_img, blurred_img_numpy, 'Adversarial_' + file_name_to_export)
    print("Adversary Interpretable Explanations Completed")

    # Ploting:
    fig = plt.figure()
    fig.suptitle(file_name_to_export + ' - ' + attack_type + ' - Interpretable Explanations')

    ax11 = fig.add_subplot(2, 5, 1)
    ax11.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    ax11.set_title('Original Image')

    heat = cv2.imread('results/' + file_name_to_export + "_Explain_Heatmap.jpg", 1)
    ax1 = fig.add_subplot(2, 5, 2)
    ax1.imshow(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB))
    ax1.set_title('Learned Mask Color')

    mask = cv2.imread('results/' + file_name_to_export + "_Explain_Mask.jpg", 0)
    ax2 = fig.add_subplot(2, 5, 3)
    ax2.imshow(mask)
    ax2.set_title('Learned Mask Gray')

    cam = cv2.imread('results/' + file_name_to_export + "_Explain_Cam.jpg", 1)
    ax3 = fig.add_subplot(2, 5, 4)
    ax3.imshow(cv2.cvtColor(cam, cv2.COLOR_BGR2RGB))
    ax3.set_title('Cam Result')

    ax9 = fig.add_subplot(2, 5, 5)
    ax9.bar(indices, orig_vals, align='center', alpha=0.5)
    ax9.set_title('Orignial Image Predictions')
    ax9.set_xticks(indices)
    ax9.set_xticklabels(orig_labs, rotation=45, ha="right")

    ax12 = fig.add_subplot(2, 5, 6)
    ax12.imshow(cv2.cvtColor(np.uint8(adversarialpic), cv2.COLOR_BGR2RGB))
    ax12.set_title('Adversary Image(SSIM = ' + str(diff) + ')')

    heat2 = cv2.imread('results/Adversarial_' + file_name_to_export + "_Explain_Heatmap.jpg", 1)
    ax5 = fig.add_subplot(2, 5, 7)
    ax5.imshow(cv2.cvtColor(heat2, cv2.COLOR_BGR2RGB))
    ax5.set_title('Adversary Mask Color')

    mask2 = cv2.imread('results/Adversarial_' + file_name_to_export + "_Explain_Mask.jpg", 0)
    ax6 = fig.add_subplot(2, 5, 8)
    ax6.imshow(mask2)
    ax6.set_title('Adversary Mask Gray')

    cam2 = cv2.imread('results/Adversarial_' + file_name_to_export + "_Explain_Cam.jpg", 1)
    ax7 = fig.add_subplot(2, 5, 9)
    ax7.imshow(cv2.cvtColor(cam2, cv2.COLOR_BGR2RGB))
    ax7.set_title('Adversary Cam Result')

    ax10 = fig.add_subplot(2, 5, 10)
    ax10.bar(indices, adver_vals, align='center', alpha=0.5)
    ax10.set_title('Adversary Image Predictions')
    ax10.set_xticks(indices)
    ax10.set_xticklabels(adver_labs, rotation=45, ha="right")

    fig.set_size_inches(32, 18)
    fig.tight_layout()
    if isTrained:
        train = 'Trained'
    else:
        train = 'UnTrained'
    fig.savefig('Concise Results/' + file_name_to_export + '_' + attack_type +
                '_InterpExp(' + train + choose_network + ')', dpi=100)
