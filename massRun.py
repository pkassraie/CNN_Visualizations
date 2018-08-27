from visualization import *
from drawPlot import *

attacks = ['FGSM','PGD','DeepFool','Boundary','SinglePixel','SalMap','LBFGS','RPGD']
networks = ['VGG19','AlexNet','ResNet50','Custom']
trained = True
target_example = range(0, 6)
a = 'FGSM'
i = 4

compareAttacks('GradCam', 'VGG19', 6, training='', structure='')
#compareVisualizations(False,'FGSM','AlexNet',6,training='', structure='')
# Compatible with all networks:
#runExplain(target_example=i,attack_type="LBFGS",isTrained=trained,iters=200,choose_network='ResNet50')
#runGradCam(target_example=5, attack_type=a, isTrained=trained,choose_network="AlexNet")
#runGGradCam(target_example=i, attack_type=a, isTrained=trained,choose_network="ResNet50")
#runGBackProp(target_example=i, attack_type=a, isTrained=trained,choose_network="ResNet50")
#runsmoothGrad(target_example=3,attack_type='Boundary',choose_network='VGG19')
#runVanillaBP(target_example=i, attack_type=a, isTrained=trained,choose_network="ResNet50")


# Only for VGG19 & AlexNet
#runInvRep(target_example=i, attack_type=a, isTrained=trained,choose_network="VGG19")
#runDeepDream(target_example=3,iters=10)