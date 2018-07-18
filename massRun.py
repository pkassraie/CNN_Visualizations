
from visualization import *
from matplotlib import pyplot as plt
attacks = ['FGSM','PGD','DeepFool','Boundary','SinglePixel','SalMap','LBFGS','RPGD']
networks = ['VGG19','AlexNet']
trained = True
target_example = range(0, 4) #its (0,4) really

#fig = plt.figure()
#fig.suptitle('Covariance Matrices of'+str(target_example)+' (on AlexNet)')
#j = 1
a = 'FGSM'

for i in target_example:
    #runDeepDream(target_example=i)
    runGradCam(target_example=i, attack_type=a, isTrained=trained)
    runGGradCam(target_example=i, attack_type=a, isTrained=trained)
    runGBackProp(target_example=i, attack_type=a, isTrained=trained)
    runInvRep(target_example=i, attack_type=a, isTrained=trained)
    runsmoothGrad(target_example=i, attack_type=a, isTrained=trained)
    runVanillaBP(target_example=i, attack_type=a, isTrained=trained)

    #ax1 = fig.add_subplot(3,3,j)
    #ax1.imshow(A)
    #ax1.set_title('GradCam ' + a)
    #ax2 = fig.add_subplot(3,3,j+1)
    #ax2.imshow(B)
    #ax2.set_title('Guided GradCam ' + a)
    #ax3 = fig.add_subplot(3,3,j+2)
    #ax3.imshow(C)
    #ax3.set_title('Smooth Grad ' + a)
    #j=j+3

#fig.set_size_inches(18.5, 10.5)
#fig.savefig('Concise Results/Covariance'+str(target_example)+'_AlexNet',dpi = 100)