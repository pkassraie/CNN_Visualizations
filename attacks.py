import numpy as np
import foolbox
import cv2
import torch
from misc_functions import preprocess_image
from matplotlib import pyplot as plt
from skimage.measure import compare_ssim as ssim

# So basically, the attack functions receive a trained model, an image (and sometimes) the correct label.
# They return a processed adversarial image, ready to be fed to the visualizer.

class attack():


    def __init__(self,model_name,type, model, image, file_name,label):
        self.type = type
        self.model = model
        self.model_name = model_name
        self.image = image
        self.file_name = file_name
        self.label = label

    def runAttack(self):
        # instantiate the model

        if self.model_name == 'Custom':
            ## MAYBE YOU SHOULD NOT ADD PREPROCESSING FOR ADVERSARIAL TRAININGS!!!!
            mean = np.array([0.4914, 0.4822, 0.4465]).reshape((3, 1, 1))
            std = np.array([0.2023, 0.1994, 0.2010]).reshape((3, 1, 1))
            fmodel = foolbox.models.PyTorchModel(self.model, bounds=(0, 1), num_classes=10, preprocessing=(mean, std))

        else:
            mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
            std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
            fmodel = foolbox.models.PyTorchModel(self.model, bounds=(0, 1), num_classes=1000, preprocessing=(mean, std))
        temp = self.image.copy()
        # get source image and label

        image = np.swapaxes(self.image,0,2)
        image = np.swapaxes(image,1,2)
        image = np.float32(image / 255.)  # because our model expects values in [0, 1]

        #label = target_class  #WHAT SHOULD I DO HERE?!?
        label = np.argmax(fmodel.predictions(image))
        orig_preds = np.array(fmodel.predictions(image))

        # select the attack:
        if self.type == 'FGSM':
            attack = foolbox.attacks.FGSM(fmodel)
            attack_name = '_FGSM'
            adversarial = attack(image,np.int64(label))

        elif self.type =='PGD':
            iterations = 40
            epsilon = 0.3
            if self.model_name == 'Custom':
                iterations = 100
                epsilon = 0.7
            attack = foolbox.attacks.ProjectedGradientDescentAttack(fmodel)
            attack_name = '_PGD'
            adversarial = attack(image,np.int64(label),iterations=iterations, epsilon=epsilon)

        elif self.type == 'DeepFool':
            attack = foolbox.attacks.DeepFoolLinfinityAttack(fmodel)
            attack_name = '_DeepFool'
            adversarial = attack(image,np.int64(label))

        elif self.type =='SinglePixel':
            attack = foolbox.attacks.SinglePixelAttack(fmodel)
            attack_name = '_SinglePixel'
            adversarial = attack(image,np.int64(label))

        elif self.type =='Boundary':
            attack = foolbox.attacks.BoundaryAttack(fmodel) # Default iterations ir on 5000!!
            attack_name = '_Boundary'
            adversarial = attack(image,np.int64(label))

        elif self.type == 'RPGD':
            attack = foolbox.attacks.RandomStartProjectedGradientDescentAttack(fmodel)
            attack_name = '_RPGD'
            adversarial = attack(image,np.int64(label)) # shouldn't this be trained on the actual output instead of the label?

        elif self.type == 'LBFGS':
            attack = foolbox.attacks.LBFGSAttack(fmodel)
            attack_name = '_LBFGS'
            adversarial = attack(image,np.int64(label))

        elif self.type == 'SalMap':
            attack = foolbox.attacks.SaliencyMapAttack(fmodel)
            attack_name = '_SalMap'
            adversarial = attack(image,np.int64(label))

        # apply attack on source image
        advers_class = np.argmax(fmodel.predictions(adversarial))
        adver_preds = np.array(fmodel.predictions(adversarial))


        print('predicted class', np.argmax(fmodel.predictions(image)))
        print('adversarial class', advers_class)

        adversarial = np.swapaxes(adversarial,0,2)
        adversarial = np.swapaxes(adversarial,0,1)
        adversarial = adversarial * 255.

        cv2.imwrite('results/'+self.file_name+ attack_name +'_Attack.jpg',adversarial)
        #cv2.imwrite('results/'+self.file_name+ attack_name +'_Attack(Difference).jpg',adversarial-temp)


        diff = ssim(np.float32(temp), adversarial,multichannel=True)
        adversarial2 = preprocess_image(adversarial)

        return adversarial,adversarial2,advers_class,orig_preds,adver_preds,diff

    def getstuff(self):
        adversarial,adversarial2,advers_class,orig_pred,adver_pred,diff = self.runAttack()
        return adversarial,adversarial2,advers_class,orig_pred,adver_pred,diff

