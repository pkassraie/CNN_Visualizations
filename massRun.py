
from visualization import *
from matplotlib import pyplot as plt
attacks = ['FGSM','PGD','DeepFool'] # also 'Boundary','SinglePixel'
networks = ['VGG19','AlexNet']
target_example = range(0, 4) #its 4 really

fig = plt.figure()
fig.suptitle('Covariance Matrices of spider (on AlexNet)')
j = 1
#for i in target_example:
i=2
for a in attacks:
    #runDeepDream(target_example=i)
    A = runGradCam(target_example=i,attack_type=a)
    B = runGGradCam(target_example=i, attack_type=a)
    #runGBackProp(target_example=i)
    #runInvRep(target_example=i)
    C = runsmoothGrad(target_example=i,attack_type=a)
    #runVanillaBP(target_example=i)
    ax1 = fig.add_subplot(3,3,j)
    ax1.imshow(A)
    ax1.set_title('GradCam ' + a)
    ax2 = fig.add_subplot(3,3,j+1)
    ax2.imshow(B)
    ax2.set_title('Guided GradCam ' + a)
    ax3 = fig.add_subplot(3,3,j+2)
    ax3.imshow(C)
    ax3.set_title('Smooth Grad ' + a)
    j=j+3

fig.set_size_inches(18.5, 10.5)
fig.savefig('Concise Results/Spider_AlexNet',dpi = 100)