
from visualization import *

attacks = ['FGSM','PGD','Boundary','SinglePixel','DeepFool']
networks = ['VGG19','AlexNet']
target_example = range(0, 3) #its 4 really

#for i in target_example:
    #runDeepDream(target_example=i)
#runGradCam(target_example=3,choose_network='VGG19')
runGGradCam(target_example=3,choose_network='VGG19')
    #runGBackProp(target_example=i)
    #runInvRep(target_example=i)
    #runsmoothGrad(target_example=i)
    #runVanillaBP(target_example=i)