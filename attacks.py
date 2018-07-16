import numpy as np
import foolbox
import cv2
from misc_functions import preprocess_image


# So basically, the attack functions receive a trained model, an image (and sometimes) the correct label.
# They return a processed adversarial image, ready to be fed to the visualizer.

def attack(type, model, image, file_name,label):

    # instantiate the model
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    fmodel = foolbox.models.PyTorchModel(
        model, bounds=(0, 1), num_classes=1000, preprocessing=(mean, std))
    temp = image.copy()
    # get source image and label
    image = np.swapaxes(image,0,2)
    image = np.swapaxes(image,1,2)
    image = np.float32(image / 255.)  # because our model expects values in [0, 1]
    #label = target_class  #WHAT SHOULD I DO HERE?!?
    label = np.argmax(fmodel.predictions(image))

    # select the attack:
    if type == 'FGSM':
        attack = foolbox.attacks.FGSM(fmodel)
        attack_name = '_FGSM'
        adversarial = attack(image,np.int64(label))
    elif type =='PGD':
        attack = foolbox.attacks.RandomStartProjectedGradientDescentAttack(fmodel)
        attack_name = '_PGD'
        adversarial = attack(image,np.int64(label))
    elif type == 'DeepFool':
        attack = foolbox.attacks.DeepFoolLinfinityAttack(fmodel)
        attack_name = '_DeepFool'
        adversarial = attack(image,np.int64(label))
    elif type =='SinglePixel':
        attack = foolbox.attacks.SinglePixelAttack(fmodel)
        attack_name = '_SinglePixel'
        adversarial = attack(image,np.int64(label))
    elif type =='Boundary':
        attack = foolbox.attacks.BoundaryAttack(fmodel) # Default iterations ir on 5000!!
        attack_name = '_Boundary'
        adversarial = attack(image,np.int64(label))

    adversarial = attack(image,np.int64(label)) # shouldn't this be trained on the actual output instead of the label?
    # apply attack on source image
    advers_class = np.argmax(fmodel.predictions(adversarial))

    print('label', label)
    print('predicted class', np.argmax(fmodel.predictions(image)))
    print('adversarial class', advers_class)

    adversarial = np.swapaxes(adversarial,0,2)
    adversarial = np.swapaxes(adversarial,0,1)
    adversarial = adversarial * 255.

    cv2.imwrite('results/'+file_name+ attack_name +'_Attack.jpg',adversarial)
    cv2.imwrite('results/'+ file_name+'.jpg',temp)
    adversarial = preprocess_image(adversarial)

    return adversarial,advers_class