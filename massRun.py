#!/usr/bin/python

from visualization import *
from drawPlot import *

attacks = ['FGSM','PGD','DeepFool','Boundary','SinglePixel','SalMap','LBFGS','RPGD']
networks = ['VGG19','AlexNet','ResNet50','Custom']
visualmethods = ['VanillaBP','SmoothGrad','Explain']#'GradCam','GGradCam','GBP',
trained = True
target_example = range(0, 6)


#runVanillaBP('Custom')


#runExplain2('ResNet50',target_example=4,iters=20)
#runGradCam2('ResNet50',target_example=5,attack_type='DeepFool')
#runInvRep('VGG19',target_example=5)
#compareAttacks('Explain', 'ResNet50', 4, training='', structure='')
#compareVisualizations('FGSM','AlexNet',6,training='', structure='')
#compareNetworks('FGSM','GradCam',4)

# Compatible with all networks:
runExplain(target_example=4,attack_type="LBFGS",isTrained=trained,iters=100,choose_network='AlexNet')
#runGradCam(target_example=5, attack_type=a, isTrained=trained,choose_network="AlexNet")
#runGGradCam(target_example=i, attack_type=a, isTrained=trained,choose_network="ResNet50")
#runGBackProp(target_example=4, attack_type='LBFGS', isTrained=trained,choose_network="ResNet50")
#runsmoothGrad(target_example=3,attack_type='Boundary',choose_network='VGG19')
#runVanillaBP(target_example=5, attack_type='SalMap', isTrained=trained,choose_network="ResNet50")


# Only for VGG19 & AlexNet
runInvRep(target_example=4, attack_type='FGSM', isTrained=trained,choose_network="VGG19")
#runDeepDream(target_example=3,iters=10)