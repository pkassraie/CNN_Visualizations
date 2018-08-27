from visualization import runExplain,runGGradCam,runGradCam,runVanillaBP,runsmoothGrad,runGBackProp
from misc_functions import get_params
from matplotlib import pyplot as plt
# for grad cam these are the outputs:
#original_image,gray,color,result,adversarial,gray2,color2,result2
# for explain:
#original_img,heat, mask,cam,adversarialpic,heat2, cam2
# for GBP:
#original_image, colorgrads,graygrads,possal, negsal, adversarial,colorgrads2,graygrads2,possal2,negsal2
# for GGCam:
# original_image, guidedgrad, grayguidedgrad, adversarial, guidedgrad2, grayguidedgrad2
# for smooth_grad:
#original_image,colorgrads,graygrads,adversarial,colorgrads2,graygrads2
# vanilla BP:
# original_image,vanilbp,grayvanilbp,adversarial,vanilbp2,grayvanilbp2

# photo index, network, visualization
def compareAttacks(vizmethod, choose_network, image_index, training='', structure=''):
    isTrained = True
    _,_,_,img_name,_ = get_params(image_index,choose_network,isTrained,training, structure)
    attacks = ['FGSM','DeepFool','PGD','SalMap','LBFGS','RPGD' , 'Boundary']#,'SinglePixel']
    rows = 1+len(attacks)
    fig = plt.figure()
    fig.suptitle('Comparing Attacks:'+img_name+' - '+ vizmethod)


    if vizmethod == 'Explain':
        iters = 50
        j = 1
        for i in attacks:

            original_img,heat, mask,cam,\
            adversarialpic,heat2, mask2, cam2,\
                indices,orig_labs,orig_vals,adver_labs,adver_vals= runExplain(choose_network,isTrained,
                                                                              training,structure,
                                                                              image_index,iters,
                                                                              attack_type=i)

            if j == 1:
                ax11 = fig.add_subplot(rows, 5, 1)
                ax11.imshow(original_img)
                ax11.set_title('Original Image')
                ax1 = fig.add_subplot(rows, 5, 2)
                ax1.imshow(heat)
                ax1.set_title('Learned Mask Color')
                ax2 = fig.add_subplot(rows, 5, 3)
                ax2.imshow(mask)
                ax2.set_title('Learned Mask Gray')
                ax3 = fig.add_subplot(rows, 5, 4)
                ax3.imshow(cam)
                ax3.set_title('Cam Result')
                ax9 = fig.add_subplot(rows, 5, 5)
                ax9.bar(indices, orig_vals, align='center', alpha=0.5)
                ax9.set_title('Orignial Image Predictions')
                ax9.set_xticks(indices)
                ax9.set_xticklabels(orig_labs, rotation=45, ha="right")


            ax12 = fig.add_subplot(rows, 5, 5*j+1)
            ax12.imshow(adversarialpic)
            ax12.set_title(i + ' Attack')
            ax5 = fig.add_subplot(rows, 5, 5*j+2)
            ax5.imshow(heat2)
            ax6 = fig.add_subplot(rows, 5, 5*j+3)
            ax6.imshow(mask2)
            ax6.set_title('Adversary Mask Gray')
            ax7 = fig.add_subplot(rows, 5, 5*j+4)
            ax7.imshow(cam2)
            ax7.set_title('Adversary Cam Result')
            ax10 = fig.add_subplot(rows, 5, 5*(j+1))
            ax10.bar(indices, adver_vals, align='center', alpha=0.5)
            ax10.set_title('Adversary Image Predictions')
            ax10.set_xticks(indices)
            ax10.set_xticklabels(adver_labs, rotation=45, ha="right")
            j += 1

        fig.set_size_inches(32, 9*rows)
        fig.tight_layout()
        fig.savefig('Comparing/AttackComp' +'_' +
                    img_name +'_' +
                    vizmethod + ' (' +
                    choose_network +'_' +
                    training + '_' +
                    structure + ' )', dpi=100)



    elif vizmethod == 'GradCam':
        j=1
        for i in attacks:
            print(i)
            original_image,gray,color,result,adversarial,gray2,color2,result2,\
                indices,orig_labs,orig_vals,adver_labs,adver_vals = runGradCam(choose_network,isTrained,
                                                                               training,structure,
                                                                               image_index,attack_type=i)
            if j == 1:
                ax0 = fig.add_subplot(rows,5,1)
                ax0.imshow(original_image)
                ax0.set_title('Original Image')
                ax1 = fig.add_subplot(rows,5,2)
                ax1.imshow(gray)
                ax1.set_title('Cam Grasycale')
                ax2 = fig.add_subplot(rows,5,3)
                ax2.imshow(color)
                ax2.set_title('Cam HeatMap')
                ax3 = fig.add_subplot(rows,5,4)
                ax3.imshow(result)
                ax3.set_title('Cam Result')
                ax9 = fig.add_subplot(rows,5,5)
                ax9.bar(indices,orig_vals,align='center', alpha=0.5)
                ax9.set_title('Orignial Image Predictions')
                ax9.set_xticks(indices)
                ax9.set_xticklabels(orig_labs,rotation = 45,ha="right")

            ax12 = fig.add_subplot(rows,5,5*j+1)
            ax12.imshow(adversarial)
            ax12.set_title(i + ' Attack')
            ax4 = fig.add_subplot(rows,5,5*j+2)
            ax4.imshow(gray2)
            ax4.set_title('Adversary Cam Grasycale')
            ax5 = fig.add_subplot(rows,5,5*j+3)
            ax5.imshow(color2)
            ax5.set_title('Adversary Cam HeatMap')
            ax6 = fig.add_subplot(rows,5,5*j+4)
            ax6.imshow(result2)
            ax6.set_title('Adversary Cam Result')

            ax10 = fig.add_subplot(rows,5,5*j+5)
            ax10.bar(indices,adver_vals,align='center', alpha=0.5)
            ax10.set_title('Adversary Image Predictions')
            ax10.set_xticks(indices)
            ax10.set_xticklabels(adver_labs,rotation = 45,ha="right")
            j +=1

        fig.set_size_inches(32, 9*rows)
        fig.tight_layout()
        fig.savefig('Comparing/AttackComp' +'_' +
                    img_name +'_' +
                    vizmethod + ' (' +
                    choose_network +'_' +
                    training + '_' +
                    structure + ' )', dpi=100)

    elif vizmethod == 'GBP':
        j =1
        for i in attacks:
            original_image, colorgrads,graygrads,possal, negsal, \
            adversarial,colorgrads2,graygrads2,possal2,negsal2,\
                indices,orig_labs,orig_vals,adver_labs,adver_vals = runGBackProp(choose_network,isTrained,
                                                                                 training,structure,
                                                                                 image_index,attack_type=i)
            if j ==1:
                ax11 = fig.add_subplot(2,6,1)
                ax11.imshow(original_image)
                ax11.set_title('Original Image')

                ax1 = fig.add_subplot(2,6,2)
                ax1.imshow(colorgrads)
                ax1.set_title('Guided BP Color')

                ax2 = fig.add_subplot(2, 6, 3)
                ax2.imshow(graygrads)
                ax2.set_title( 'Guided BP Gray')
                ax3 = fig.add_subplot(2, 6, 4)
                ax3.imshow(possal)
                ax3.set_title('Positive Saliency')
                ax4 = fig.add_subplot(2, 6, 5)
                ax4.imshow(negsal)
                ax4.set_title('Negative Saliency')


                ax9 = fig.add_subplot(2,6,6)
                ax9.bar(indices,orig_vals,align='center', alpha=0.5)
                ax9.set_title('Orignial Image Predictions')
                ax9.set_xticks(indices)
                ax9.set_xticklabels(orig_labs,rotation = 45,ha="right")

            ax12 = fig.add_subplot(2,6,6*j+1)
            ax12.imshow(adversarial)
            ax12.set_title(i + ' Attack')
            ax5 = fig.add_subplot(2, 6, 6*j+2)
            ax5.imshow(colorgrads2)
            ax5.set_title('Adversarial Guided BP Color')
            ax6 = fig.add_subplot(2, 6, 6*j+3)
            ax6.imshow(graygrads2)
            ax6.set_title('Adversarial'+ 'Guided BP Gray')
            ax7 = fig.add_subplot(2, 6, 6*j+4)
            ax7.imshow(possal2)
            ax7.set_title('Adversarial ''Positive Saliency')
            ax8 = fig.add_subplot(2, 6, 6*j+5)
            ax8.imshow(negsal2)
            ax8.set_title('Adversarial'+'Negative Saliency')
            ax10 = fig.add_subplot(2,6,6*j+6)
            ax10.bar(indices,adver_vals,align='center', alpha=0.5)
            ax10.set_title('Adversary Image Predictions')
            ax10.set_xticks(indices)
            ax10.set_xticklabels(adver_labs,rotation = 45,ha="right")
            j += 1

        fig.set_size_inches(32, 9*rows)
        fig.tight_layout()
        fig.savefig('Comparing/AttackComp' +'_' +
                    img_name +'_' +
                    vizmethod + ' (' +
                    choose_network +'_' +
                    training + '_' +
                    structure + ' )', dpi=100)
    elif vizmethod == 'GGradCam':
        j = 1
        for i in attacks:
            original_image, guidedgrad, grayguidedgrad,\
            adversarial, guidedgrad2, grayguidedgrad2,\
                indices,orig_labs,orig_vals,adver_labs,adver_vals = runGGradCam(choose_network,isTrained,
                                                                                training,structure,
                                                                                image_index,attack_type=i)
            if j ==1:
                ax0 = fig.add_subplot(rows,4,1)
                ax0.imshow(original_image)
                ax0.set_title('Original Image')
                ax1 = fig.add_subplot(rows,4,2)
                ax1.imshow(guidedgrad)
                ax1.set_title('Guided Grad Cam')
                ax2 = fig.add_subplot(rows,4,3)
                ax2.imshow(grayguidedgrad)
                ax2.set_title('Guided Grad Cam Grasycale')

                ax9 = fig.add_subplot(rows,4,4)
                ax9.bar(indices,orig_vals,align='center', alpha=0.5)
                ax9.set_title('Orignial Image Predictions')
                ax9.set_xticks(indices)
                ax9.set_xticklabels(orig_labs,rotation = 45,ha="right")

            ax12 = fig.add_subplot(rows,4,4*j+1)
            ax12.imshow(adversarial)
            ax12.set_title(i + ' Attack')

            ax3 = fig.add_subplot(rows,4,4*j+2)
            ax3.imshow(guidedgrad2)
            ax3.set_title('Adversary Guided Grad Cam')
            ax4 = fig.add_subplot(rows,4,4*j+3)
            ax4.imshow(grayguidedgrad2)
            ax4.set_title('Adversary Guided Grad Cam Grasycale')

            ax10 = fig.add_subplot(rows,4,4*j+4)
            ax10.bar(indices,adver_vals,align='center', alpha=0.5)
            ax10.set_title('Adversary Image Predictions')
            ax10.set_xticks(indices)
            ax10.set_xticklabels(adver_labs,rotation = 45,ha="right")

        fig.set_size_inches(32, 9*rows)
        fig.tight_layout()
        fig.savefig('Comparing/AttackComp' +'_' +
                    img_name +'_' +
                    vizmethod + ' (' +
                    choose_network +'_' +
                    training + '_' +
                    structure + ' )', dpi=100)
    elif vizmethod == 'SmoothGrad':
        j=1
        for i in attacks:
            original_image,colorgrads,graygrads,\
            adversarial,colorgrads2,graygrads2,\
                indices,orig_labs,orig_vals,adver_labs,adver_vals = runsmoothGrad(choose_network,isTrained,
                                                                                  training,structure,
                                                                                  image_index,attack_type=i)
            if j ==1:
                ax0 = fig.add_subplot(rows,4,1)
                ax0.imshow(original_image)
                ax0.set_title('Original Image')

                ax1 = fig.add_subplot(rows,4,2)
                ax1.imshow(colorgrads)
                ax1.set_title('Smooth BP')
                ax2 = fig.add_subplot(rows,4, 3)
                ax2.imshow(graygrads)
                ax2.set_title('Smooth BP Gray')

                ax9 = fig.add_subplot(rows,4,4)
                ax9.bar(indices,orig_vals,align='center', alpha=0.5)
                ax9.set_title('Orignial Image Predictions')
                ax9.set_xticks(indices)
                ax9.set_xticklabels(orig_labs,rotation = 45,ha="right")

            ax12 = fig.add_subplot(rows,4,4*j+1)
            ax12.imshow(adversarial)
            ax12.set_title(i + ' Attack')
            ax3 = fig.add_subplot(rows,4,4*j+2)
            ax3.imshow(colorgrads2)
            ax3.set_title('Adversary Smooth BP')
            ax4 = fig.add_subplot(rows,4, 4*j+3)
            ax4.imshow(graygrads2)
            ax4.set_title('Adversary Smooth BP Gray')
            ax10 = fig.add_subplot(rows,4,4*j+4)
            ax10.bar(indices,adver_vals,align='center', alpha=0.5)
            ax10.set_title('Adversary Image Predictions')
            ax10.set_xticks(indices)
            ax10.set_xticklabels(adver_labs,rotation = 45,ha="right")
            j += 1

        fig.set_size_inches(32, 9*rows)
        fig.tight_layout()
        fig.savefig('Comparing/AttackComp' +'_' +
                    img_name +'_' +
                    vizmethod + ' (' +
                    choose_network +'_' +
                    training + '_' +
                    structure + ' )', dpi=100)

    elif vizmethod == 'VanillaBP':
        j = 1
        for i in attacks:
            original_image,vanilbp,grayvanilbp,\
            adversarial,vanilbp2,grayvanilbp2,\
                indices,orig_labs,orig_vals,adver_labs,adver_vals = runVanillaBP(choose_network,
                                                                                 isTrained,training,structure,
                                                                                 image_index,attack_type=i)
            if j==1:
                ax0 = fig.add_subplot(rows,4,1)
                ax0.imshow(original_image)
                ax0.set_title('Original Image')
                ax1 = fig.add_subplot(rows,4,2)
                ax1.imshow(vanilbp)
                ax1.set_title('Vanilla BackProp')
                ax2 = fig.add_subplot(rows,4,3)
                ax2.imshow(grayvanilbp)
                ax2.set_title('Vanilla BackProp GrayScale')
                ax9 = fig.add_subplot(rows,4,4)
                ax9.bar(indices,orig_vals,align='center', alpha=0.5)
                ax9.set_title('Orignial Image Predictions')
                ax9.set_xticks(indices)
                ax9.set_xticklabels(orig_labs,rotation = 45,ha="right")

            ax12 = fig.add_subplot(rows,4,j*4+1)
            ax12.imshow(adversarial)
            ax12.set_title(i + ' Attack')
            ax3 = fig.add_subplot(rows,4,j*4+2)
            ax3.imshow(vanilbp2)
            ax3.set_title('Adversary Vanilla BackProp')
            ax4 = fig.add_subplot(rows,4,j*4+3)
            ax4.imshow(grayvanilbp2)
            ax4.set_title('Adversary Vanilla BackProp GrayScale')
            ax10 = fig.add_subplot(rows,4,j*4+4)
            ax10.bar(indices,adver_vals,align='center', alpha=0.5)
            ax10.set_title('Adversary Image Predictions')
            ax10.set_xticks(indices)
            ax10.set_xticklabels(adver_labs,rotation = 45,ha="right")
            j += 1

        fig.set_size_inches(32, 9*rows)
        fig.tight_layout()
        fig.savefig('Comparing/AttackComp' +'_' +
                    img_name +'_' +
                    vizmethod + ' (' +
                    choose_network +'_' +
                    training + '_' +
                    structure + ' )', dpi=100)

# *********************************************************************************************************
# *********************************************************************************************************

def compareNetworks(attackmethod,vizmethod, image_index, training='Normal'):
    isTrained = True
    i = attackmethod
    _,_,_,img_name,_ = get_params(image_index,'AlexNet',isTrained) #dont mind this alexnet here.

    networks = ('AlexNet','VGG19','ResNet50','Custom')
    structures = ('ResNet50','VGG19')
    numberofmodels = len(networks)+len(structures)-1
    rows = 2*numberofmodels

    fig = plt.figure()
    fig.suptitle('Comparing Networks:'+img_name+' - '+ attackmethod)
    j = 0
    n = 0
    s = 0
    while n<len(networks) and s<len(structures):
        choose_network = networks[n]
        structure = structures[s]

        if vizmethod == 'Explain':
            iters = 100
            original_img,heat, mask,cam,adversarialpic,heat2, mask2, cam2,\
                    indices,orig_labs,orig_vals,adver_labs,adver_vals= runExplain(choose_network,isTrained,
                                                                                  training,structure,
                                                                                  image_index,iters,
                                                                                  attack_type=i)
            ax11 = fig.add_subplot(rows,5, 5*j+1)
            ax11.imshow(original_img)
            ax11.set_title('Original Image. Method: '+ vizmethod)
            ax1 = fig.add_subplot(rows,5, 5*j+2)
            ax1.imshow(heat)
            ax1.set_title('Learned Mask Color')
            ax2 = fig.add_subplot(rows,5, 5*j+3)
            ax2.imshow(mask)
            ax2.set_title('Learned Mask Gray')
            ax3 = fig.add_subplot(rows,5, 5*j+4)
            ax3.imshow(cam)
            ax3.set_title('Cam Result')
            ax9 = fig.add_subplot(rows,5, 5*j+5)
            ax9.bar(indices, orig_vals, align='center', alpha=0.5)
            ax9.set_title('Orignial Image Predictions')
            ax9.set_xticks(indices)
            ax9.set_xticklabels(orig_labs, rotation=45, ha="right")
            ax12 = fig.add_subplot(rows,5, 5*j+6)
            ax12.imshow(adversarialpic)
            ax12.set_title('Adversarial Image')
            ax5 = fig.add_subplot(rows,5, 5*j+7)
            ax5.imshow(heat2)
            ax6 = fig.add_subplot(rows,5, 5*j+8)
            ax6.imshow(mask2)
            ax6.set_title('Adversary Mask Gray')
            ax7 = fig.add_subplot(rows,5, 5*j+9)
            ax7.imshow(cam2)
            ax7.set_title('Adversary Cam Result')
            ax10 = fig.add_subplot(rows,5, 5*j+10)
            ax10.bar(indices, adver_vals, align='center', alpha=0.5)
            ax10.set_title('Adversary Image Predictions')
            ax10.set_xticks(indices)
            ax10.set_xticklabels(adver_labs, rotation=45, ha="right")
            j += 2

        elif vizmethod == 'GradCam':
            original_image,gray,color,result,adversarial,gray2,color2,result2,\
                indices,orig_labs,orig_vals,adver_labs,adver_vals = runGradCam(choose_network,isTrained,
                                                                               training,structure,
                                                                               image_index,attack_type=i)
            ax0 = fig.add_subplot(rows,5,5*j+1)
            ax0.imshow(original_image)
            ax0.set_title('Original Image. Method:' +vizmethod)
            ax1 = fig.add_subplot(rows,5,5*j+2)
            ax1.imshow(gray)
            ax1.set_title('Cam Grasycale')
            ax2 = fig.add_subplot(rows,5,5*j+3)
            ax2.imshow(color)
            ax2.set_title('Cam HeatMap')
            ax3 = fig.add_subplot(rows,5,5*j+4)
            ax3.imshow(result)
            ax3.set_title('Cam Result')
            ax9 = fig.add_subplot(rows,5,5*j+5)
            ax9.bar(indices,orig_vals,align='center', alpha=0.5)
            ax9.set_title('Orignial Image Predictions')
            ax9.set_xticks(indices)
            ax9.set_xticklabels(orig_labs,rotation = 45,ha="right")
            ax12 = fig.add_subplot(rows,5,5*j+6)
            ax12.imshow(adversarial)
            ax12.set_title('Adversarial Image')
            ax4 = fig.add_subplot(rows,5,5*j+7)
            ax4.imshow(gray2)
            ax4.set_title('Adversary Cam Grasycale')
            ax5 = fig.add_subplot(rows,5,5*j+8)
            ax5.imshow(color2)
            ax5.set_title('Adversary Cam HeatMap')
            ax6 = fig.add_subplot(rows,5,5*j+9)
            ax6.imshow(result2)
            ax6.set_title('Adversary Cam Result')
            ax10 = fig.add_subplot(rows,5,5*j+10)
            ax10.bar(indices,adver_vals,align='center', alpha=0.5)
            ax10.set_title('Adversary Image Predictions')
            ax10.set_xticks(indices)
            ax10.set_xticklabels(adver_labs,rotation = 45,ha="right")
            j +=2

        elif vizmethod == 'GBP':
            original_image, colorgrads,graygrads,possal, negsal, \
            adversarial,colorgrads2,graygrads2,possal2,negsal2,\
                indices,orig_labs,orig_vals,adver_labs,adver_vals = runGBackProp(choose_network,isTrained,
                                                                                 training,structure,
                                                                                 image_index,attack_type=i)

            ax11 = fig.add_subplot(rows,6,6*j+1)
            ax11.imshow(original_image)
            ax11.set_title('Original Image. Method:' +vizmethod)
            ax1 = fig.add_subplot(rows,6,6*j+2)
            ax1.imshow(colorgrads)
            ax1.set_title('Guided BP Color')
            ax2 = fig.add_subplot(rows,6, 6*j+3)
            ax2.imshow(graygrads)
            ax2.set_title( 'Guided BP Gray')
            ax3 = fig.add_subplot(rows,6, 6*j+4)
            ax3.imshow(possal)
            ax3.set_title('Positive Saliency')
            ax4 = fig.add_subplot(rows,6, 6*j+5)
            ax4.imshow(negsal)
            ax4.set_title('Negative Saliency')
            ax9 = fig.add_subplot(rows,6,6*j+6)
            ax9.bar(indices,orig_vals,align='center', alpha=0.5)
            ax9.set_title('Orignial Image Predictions')
            ax9.set_xticks(indices)
            ax9.set_xticklabels(orig_labs,rotation = 45,ha="right")
            ax12 = fig.add_subplot(rows,6,6*j+7)
            ax12.imshow(adversarial)
            ax12.set_title('Adversarial Image')
            ax5 = fig.add_subplot(rows,6, 6*j+8)
            ax5.imshow(colorgrads2)
            ax5.set_title('Adversarial Guided BP Color')
            ax6 = fig.add_subplot(rows,6, 6*j+9)
            ax6.imshow(graygrads2)
            ax6.set_title('Adversarial'+ 'Guided BP Gray')
            ax7 = fig.add_subplot(rows,6, 6*j+10)
            ax7.imshow(possal2)
            ax7.set_title('Adversarial ''Positive Saliency')
            ax8 = fig.add_subplot(rows,6, 6*j+11)
            ax8.imshow(negsal2)
            ax8.set_title('Adversarial'+'Negative Saliency')
            ax10 = fig.add_subplot(rows,6,6*j+12)
            ax10.bar(indices,adver_vals,align='center', alpha=0.5)
            ax10.set_title('Adversary Image Predictions')
            ax10.set_xticks(indices)
            ax10.set_xticklabels(adver_labs,rotation = 45,ha="right")
            j += 2

        elif vizmethod == 'GGradCam':
            original_image, guidedgrad, grayguidedgrad,\
            adversarial, guidedgrad2, grayguidedgrad2,\
                indices,orig_labs,orig_vals,adver_labs,adver_vals = runGGradCam(choose_network,isTrained,
                                                                                training,structure,
                                                                                image_index,attack_type=i)
            ax0 = fig.add_subplot(rows,4,4*j+1)
            ax0.imshow(original_image)
            ax0.set_title('Original Image. Method:' +vizmethod)
            ax1 = fig.add_subplot(rows,4,4*j+2)
            ax1.imshow(guidedgrad)
            ax1.set_title('Guided Grad Cam')
            ax2 = fig.add_subplot(rows,4,4*j+3)
            ax2.imshow(grayguidedgrad)
            ax2.set_title('Guided Grad Cam Grasycale')
            ax9 = fig.add_subplot(rows,4,4*j+4)
            ax9.bar(indices,orig_vals,align='center', alpha=0.5)
            ax9.set_title('Orignial Image Predictions')
            ax9.set_xticks(indices)
            ax9.set_xticklabels(orig_labs,rotation = 45,ha="right")
            ax12 = fig.add_subplot(rows,4,4*j+5)
            ax12.imshow(adversarial)
            ax12.set_title('Adversarial Image')
            ax3 = fig.add_subplot(rows,4,64*j+6)
            ax3.imshow(guidedgrad2)
            ax3.set_title('Adversary Guided Grad Cam')
            ax4 = fig.add_subplot(rows,4,4*j+7)
            ax4.imshow(grayguidedgrad2)
            ax4.set_title('Adversary Guided Grad Cam Grasycale')
            ax10 = fig.add_subplot(rows,4,4*j+8)
            ax10.bar(indices,adver_vals,align='center', alpha=0.5)
            ax10.set_title('Adversary Image Predictions')
            ax10.set_xticks(indices)
            ax10.set_xticklabels(adver_labs,rotation = 45,ha="right")
            j += 2

        elif vizmethod == 'SmoothGrad':
            original_image,colorgrads,graygrads,\
            adversarial,colorgrads2,graygrads2,\
                indices,orig_labs,orig_vals,adver_labs,adver_vals = runsmoothGrad(choose_network,isTrained,
                                                                                  training,structure,
                                                                                  image_index,attack_type=i)

            ax0 = fig.add_subplot(rows,4,4*j+1)
            ax0.imshow(original_image)
            ax0.set_title('Original Image. Method:' +vizmethod)
            ax1 = fig.add_subplot(rows,4,4*j+2)
            ax1.imshow(colorgrads)
            ax1.set_title('Smooth BP')
            ax2 = fig.add_subplot(rows,4,4*j+3)
            ax2.imshow(graygrads)
            ax2.set_title('Smooth BP Gray')
            ax9 = fig.add_subplot(rows,4,4*j+4)
            ax9.bar(indices,orig_vals,align='center', alpha=0.5)
            ax9.set_title('Orignial Image Predictions')
            ax9.set_xticks(indices)
            ax9.set_xticklabels(orig_labs,rotation = 45,ha="right")
            ax12 = fig.add_subplot(rows,4,4*j+5)
            ax12.imshow(adversarial)
            ax12.set_title('Adversarial Image')
            ax3 = fig.add_subplot(rows,4,4*j+6)
            ax3.imshow(colorgrads2)
            ax3.set_title('Adversary Smooth BP')
            ax4 = fig.add_subplot(rows,4, 4*j+7)
            ax4.imshow(graygrads2)
            ax4.set_title('Adversary Smooth BP Gray')
            ax10 = fig.add_subplot(rows,4,4*j+8)
            ax10.bar(indices,adver_vals,align='center', alpha=0.5)
            ax10.set_title('Adversary Image Predictions')
            ax10.set_xticks(indices)
            ax10.set_xticklabels(adver_labs,rotation = 45,ha="right")
            j += 2


        elif vizmethod == 'VanillaBP':
            original_image,vanilbp,grayvanilbp,\
            adversarial,vanilbp2,grayvanilbp2,\
                indices,orig_labs,orig_vals,adver_labs,adver_vals = runVanillaBP(choose_network,
                                                                                 isTrained,training,structure,
                                                                                 image_index,attack_type=i)

            ax0 = fig.add_subplot(rows,4,4*j+1)
            ax0.imshow(original_image)
            ax0.set_title('Original Image. Method:' +vizmethod)
            ax1 = fig.add_subplot(rows,4,4*j+2)
            ax1.imshow(vanilbp)
            ax1.set_title('Vanilla BackProp')
            ax2 = fig.add_subplot(rows,4,4*j+3)
            ax2.imshow(grayvanilbp)
            ax2.set_title('Vanilla BackProp GrayScale')
            ax9 = fig.add_subplot(rows,4,4*j+4)
            ax9.bar(indices,orig_vals,align='center', alpha=0.5)
            ax9.set_title('Orignial Image Predictions')
            ax9.set_xticks(indices)
            ax9.set_xticklabels(orig_labs,rotation = 45,ha="right")
            ax12 = fig.add_subplot(rows,4,4*j+5)
            ax12.imshow(adversarial)
            ax12.set_title('Adversarial Image')
            ax3 = fig.add_subplot(rows,4,4*j+6)
            ax3.imshow(vanilbp2)
            ax3.set_title('Adversary Vanilla BackProp')
            ax4 = fig.add_subplot(rows,4,4*j+7)
            ax4.imshow(grayvanilbp2)
            ax4.set_title('Adversary Vanilla BackProp GrayScale')
            ax10 = fig.add_subplot(rows,4,4*j+8)
            ax10.bar(indices,adver_vals,align='center', alpha=0.5)
            ax10.set_title('Adversary Image Predictions')
            ax10.set_xticks(indices)
            ax10.set_xticklabels(adver_labs,rotation = 45,ha="right")
            j += 2

        if choose_network=='Custom':
            s += 1
        else:
            n += 1

    fig.set_size_inches(32, 9*rows)
    fig.tight_layout()
    fig.savefig('Comparing/NetworkComp' +'_' +
                img_name +'_' +
                attackmethod + ' (' +
                choose_network +'_' +
                training + '_' +
                structure + ' )', dpi=100)

# *********************************************************************************************************
# *********************************************************************************************************

def compareVisualizations(attackmethod, choose_network, image_index, training='', structure=''):
    isTrained = True
    i = attackmethod
    _,_,_,img_name,_ = get_params(image_index,choose_network,isTrained,training, structure)
    vizmethods = ['GradCam','GBP','GGradCam']#,'SmoothGrad','VanillaBP',Explain']
    rows = 2*len(vizmethods)
    fig = plt.figure()
    fig.suptitle('Comparing Visualizations:'+img_name+' - '+ attackmethod)
    j = 0

    for vizmethod in vizmethods:
        if vizmethod == 'Explain':
            iters = 100
            original_img,heat, mask,cam,adversarialpic,heat2, mask2, cam2,\
                    indices,orig_labs,orig_vals,adver_labs,adver_vals= runExplain(choose_network,isTrained,
                                                                                  training,structure,
                                                                                  image_index,iters,
                                                                                  attack_type=i)
            ax11 = fig.add_subplot(rows,6, 6*j+1)
            ax11.imshow(original_img)
            ax11.set_title('Original Image. Method: '+ vizmethod)
            ax1 = fig.add_subplot(rows,6, 6*j+2)
            ax1.imshow(heat)
            ax1.set_title('Learned Mask Color')
            ax2 = fig.add_subplot(rows,6, 6*j+3)
            ax2.imshow(mask)
            ax2.set_title('Learned Mask Gray')
            ax3 = fig.add_subplot(rows,6, 6*j+4)
            ax3.imshow(cam)
            ax3.set_title('Cam Result')
            ax9 = fig.add_subplot(rows,6, 6*j+6)
            ax9.bar(indices, orig_vals, align='center', alpha=0.5)
            ax9.set_title('Orignial Image Predictions')
            ax9.set_xticks(indices)
            ax9.set_xticklabels(orig_labs, rotation=45, ha="right")
            ax12 = fig.add_subplot(rows,6, 6*j+7)
            ax12.imshow(adversarialpic)
            ax12.set_title('Adversarial Image')
            ax5 = fig.add_subplot(rows,6, 6*j+8)
            ax5.imshow(heat2)
            ax6 = fig.add_subplot(rows,6, 6*j+9)
            ax6.imshow(mask2)
            ax6.set_title('Adversary Mask Gray')
            ax7 = fig.add_subplot(rows,6, 6*j+10)
            ax7.imshow(cam2)
            ax7.set_title('Adversary Cam Result')
            ax10 = fig.add_subplot(rows,6, 6*j+12)
            ax10.bar(indices, adver_vals, align='center', alpha=0.5)
            ax10.set_title('Adversary Image Predictions')
            ax10.set_xticks(indices)
            ax10.set_xticklabels(adver_labs, rotation=45, ha="right")
            j += 2

        elif vizmethod == 'GradCam':
            original_image,gray,color,result,adversarial,gray2,color2,result2,\
                indices,orig_labs,orig_vals,adver_labs,adver_vals = runGradCam(choose_network,isTrained,
                                                                               training,structure,
                                                                               image_index,attack_type=i)
            ax0 = fig.add_subplot(rows,6,6*j+1)
            ax0.imshow(original_image)
            ax0.set_title('Original Image. Method:' +vizmethod)
            ax1 = fig.add_subplot(rows,6,6*j+2)
            ax1.imshow(gray)
            ax1.set_title('Cam Grasycale')
            ax2 = fig.add_subplot(rows,6,6*j+3)
            ax2.imshow(color)
            ax2.set_title('Cam HeatMap')
            ax3 = fig.add_subplot(rows,6,6*j+4)
            ax3.imshow(result)
            ax3.set_title('Cam Result')
            ax9 = fig.add_subplot(rows,6,6*j+6)
            ax9.bar(indices,orig_vals,align='center', alpha=0.5)
            ax9.set_title('Orignial Image Predictions')
            ax9.set_xticks(indices)
            ax9.set_xticklabels(orig_labs,rotation = 45,ha="right")
            ax12 = fig.add_subplot(rows,6,6*j+7)
            ax12.imshow(adversarial)
            ax12.set_title('Adversarial Image')
            ax4 = fig.add_subplot(rows,6,6*j+8)
            ax4.imshow(gray2)
            ax4.set_title('Adversary Cam Grasycale')
            ax5 = fig.add_subplot(rows,6,6*j+9)
            ax5.imshow(color2)
            ax5.set_title('Adversary Cam HeatMap')
            ax6 = fig.add_subplot(rows,6,6*j+10)
            ax6.imshow(result2)
            ax6.set_title('Adversary Cam Result')
            ax10 = fig.add_subplot(rows,6,6*j+12)
            ax10.bar(indices,adver_vals,align='center', alpha=0.5)
            ax10.set_title('Adversary Image Predictions')
            ax10.set_xticks(indices)
            ax10.set_xticklabels(adver_labs,rotation = 45,ha="right")
            j +=2

        elif vizmethod == 'GBP':
            original_image, colorgrads,graygrads,possal, negsal, \
            adversarial,colorgrads2,graygrads2,possal2,negsal2,\
                indices,orig_labs,orig_vals,adver_labs,adver_vals = runGBackProp(choose_network,isTrained,
                                                                                 training,structure,
                                                                                 image_index,attack_type=i)

            ax11 = fig.add_subplot(rows,6,6*j+1)
            ax11.imshow(original_image)
            ax11.set_title('Original Image. Method:' +vizmethod)
            ax1 = fig.add_subplot(rows,6,6*j+2)
            ax1.imshow(colorgrads)
            ax1.set_title('Guided BP Color')
            ax2 = fig.add_subplot(rows,6, 6*j+3)
            ax2.imshow(graygrads)
            ax2.set_title( 'Guided BP Gray')
            ax3 = fig.add_subplot(rows,6, 6*j+4)
            ax3.imshow(possal)
            ax3.set_title('Positive Saliency')
            ax4 = fig.add_subplot(rows,6, 6*j+5)
            ax4.imshow(negsal)
            ax4.set_title('Negative Saliency')
            ax9 = fig.add_subplot(rows,6,6*j+6)
            ax9.bar(indices,orig_vals,align='center', alpha=0.5)
            ax9.set_title('Orignial Image Predictions')
            ax9.set_xticks(indices)
            ax9.set_xticklabels(orig_labs,rotation = 45,ha="right")
            ax12 = fig.add_subplot(rows,6,6*j+7)
            ax12.imshow(adversarial)
            ax12.set_title('Adversarial Image')
            ax5 = fig.add_subplot(rows,6, 6*j+8)
            ax5.imshow(colorgrads2)
            ax5.set_title('Adversarial Guided BP Color')
            ax6 = fig.add_subplot(rows,6, 6*j+9)
            ax6.imshow(graygrads2)
            ax6.set_title('Adversarial'+ 'Guided BP Gray')
            ax7 = fig.add_subplot(rows,6, 6*j+10)
            ax7.imshow(possal2)
            ax7.set_title('Adversarial ''Positive Saliency')
            ax8 = fig.add_subplot(rows,6, 6*j+11)
            ax8.imshow(negsal2)
            ax8.set_title('Adversarial'+'Negative Saliency')
            ax10 = fig.add_subplot(rows,6,6*j+12)
            ax10.bar(indices,adver_vals,align='center', alpha=0.5)
            ax10.set_title('Adversary Image Predictions')
            ax10.set_xticks(indices)
            ax10.set_xticklabels(adver_labs,rotation = 45,ha="right")
            j += 2

        elif vizmethod == 'GGradCam':
            original_image, guidedgrad, grayguidedgrad,\
            adversarial, guidedgrad2, grayguidedgrad2,\
                indices,orig_labs,orig_vals,adver_labs,adver_vals = runGGradCam(choose_network,isTrained,
                                                                                training,structure,
                                                                                image_index,attack_type=i)
            ax0 = fig.add_subplot(rows,6,6*j+1)
            ax0.imshow(original_image)
            ax0.set_title('Original Image. Method:' +vizmethod)
            ax1 = fig.add_subplot(rows,6,6*j+2)
            ax1.imshow(guidedgrad)
            ax1.set_title('Guided Grad Cam')
            ax2 = fig.add_subplot(rows,6,6*j+3)
            ax2.imshow(grayguidedgrad)
            ax2.set_title('Guided Grad Cam Grasycale')
            ax9 = fig.add_subplot(rows,6,6*j+6)
            ax9.bar(indices,orig_vals,align='center', alpha=0.5)
            ax9.set_title('Orignial Image Predictions')
            ax9.set_xticks(indices)
            ax9.set_xticklabels(orig_labs,rotation = 45,ha="right")
            ax12 = fig.add_subplot(rows,6,6*j+7)
            ax12.imshow(adversarial)
            ax12.set_title('Adversarial Image')
            ax3 = fig.add_subplot(rows,6,6*j+8)
            ax3.imshow(guidedgrad2)
            ax3.set_title('Adversary Guided Grad Cam')
            ax4 = fig.add_subplot(rows,6,6*j+9)
            ax4.imshow(grayguidedgrad2)
            ax4.set_title('Adversary Guided Grad Cam Grasycale')
            ax10 = fig.add_subplot(rows,6,6*j+12)
            ax10.bar(indices,adver_vals,align='center', alpha=0.5)
            ax10.set_title('Adversary Image Predictions')
            ax10.set_xticks(indices)
            ax10.set_xticklabels(adver_labs,rotation = 45,ha="right")
            j += 2

        elif vizmethod == 'SmoothGrad':
            original_image,colorgrads,graygrads,\
            adversarial,colorgrads2,graygrads2,\
                indices,orig_labs,orig_vals,adver_labs,adver_vals = runsmoothGrad(choose_network,isTrained,
                                                                                  training,structure,
                                                                                  image_index,attack_type=i)

            ax0 = fig.add_subplot(rows,6,6*j+1)
            ax0.imshow(original_image)
            ax0.set_title('Original Image. Method:' +vizmethod)
            ax1 = fig.add_subplot(rows,6,6*j+2)
            ax1.imshow(colorgrads)
            ax1.set_title('Smooth BP')
            ax2 = fig.add_subplot(rows,6,6*j+3)
            ax2.imshow(graygrads)
            ax2.set_title('Smooth BP Gray')
            ax9 = fig.add_subplot(rows,6,6*j+6)
            ax9.bar(indices,orig_vals,align='center', alpha=0.5)
            ax9.set_title('Orignial Image Predictions')
            ax9.set_xticks(indices)
            ax9.set_xticklabels(orig_labs,rotation = 45,ha="right")
            ax12 = fig.add_subplot(rows,6,6*j+7)
            ax12.imshow(adversarial)
            ax12.set_title('Adversarial Image')
            ax3 = fig.add_subplot(rows,6,6*j+8)
            ax3.imshow(colorgrads2)
            ax3.set_title('Adversary Smooth BP')
            ax4 = fig.add_subplot(rows,6, 6*j+9)
            ax4.imshow(graygrads2)
            ax4.set_title('Adversary Smooth BP Gray')
            ax10 = fig.add_subplot(rows,6,6*j+12)
            ax10.bar(indices,adver_vals,align='center', alpha=0.5)
            ax10.set_title('Adversary Image Predictions')
            ax10.set_xticks(indices)
            ax10.set_xticklabels(adver_labs,rotation = 45,ha="right")
            j += 2

        elif vizmethod == 'VanillaBP':
            original_image,vanilbp,grayvanilbp,\
            adversarial,vanilbp2,grayvanilbp2,\
                indices,orig_labs,orig_vals,adver_labs,adver_vals = runVanillaBP(choose_network,
                                                                                 isTrained,training,structure,
                                                                                 image_index,attack_type=i)

            ax0 = fig.add_subplot(rows,6,6*j+1)
            ax0.imshow(original_image)
            ax0.set_title('Original Image. Method:' +vizmethod)
            ax1 = fig.add_subplot(rows,6,6*j+2)
            ax1.imshow(vanilbp)
            ax1.set_title('Vanilla BackProp')
            ax2 = fig.add_subplot(rows,6,6*j+3)
            ax2.imshow(grayvanilbp)
            ax2.set_title('Vanilla BackProp GrayScale')
            ax9 = fig.add_subplot(rows,6,6*j+6)
            ax9.bar(indices,orig_vals,align='center', alpha=0.5)
            ax9.set_title('Orignial Image Predictions')
            ax9.set_xticks(indices)
            ax9.set_xticklabels(orig_labs,rotation = 45,ha="right")
            ax12 = fig.add_subplot(rows,6,6*j+7)
            ax12.imshow(adversarial)
            ax12.set_title('Adversarial Image')
            ax3 = fig.add_subplot(rows,6,6*j+8)
            ax3.imshow(vanilbp2)
            ax3.set_title('Adversary Vanilla BackProp')
            ax4 = fig.add_subplot(rows,6,6*j+9)
            ax4.imshow(grayvanilbp2)
            ax4.set_title('Adversary Vanilla BackProp GrayScale')
            ax10 = fig.add_subplot(rows,6,6*j+12)
            ax10.bar(indices,adver_vals,align='center', alpha=0.5)
            ax10.set_title('Adversary Image Predictions')
            ax10.set_xticks(indices)
            ax10.set_xticklabels(adver_labs,rotation = 45,ha="right")
            j += 2

    fig.set_size_inches(32, 9*rows)
    fig.tight_layout()
    fig.savefig('Comparing/VisualizationComp' +'_' +
                img_name +'_' +
                attackmethod + ' (' +
                choose_network +'_' +
                training + '_' +
                structure + ' )', dpi=100)


# *********************************************************************************************************
# *********************************************************************************************************

def compareTraining(attackmethod,vizmethod,structure,image_index):
    isTrained = True
    choose_network = 'Custom'
    i = attackmethod
    _,_,_,img_name,_ = get_params(image_index,'AlexNet',True) #dont mind this alexnet here.

    numberofmodels = 4
    rows = 2*numberofmodels

    fig = plt.figure()
    fig.suptitle('Comparing Networks:'+img_name+' - '+ attackmethod)

    n = 0
    j = 0
    ifNoise = 'No'

    while n<numberofmodels:

        if n==0:
            training = 'Normal'

        elif n==1:
            training = 'Adversarial'
        elif n ==2 :
            training = 'Normal'
            ifNoise = 'Yes'
            print('add noise to the image and then give it to the normally trained network')
        elif n ==3 :
            training = 'Adversarial'
            ifNoise = 'Yes'
            print('add noise to the image and then give it to the normally trained network')

        if vizmethod == 'Explain':
            iters = 100
            original_img,heat, mask,cam,adversarialpic,heat2, mask2, cam2,\
                    indices,orig_labs,orig_vals,adver_labs,adver_vals= runExplain(choose_network,isTrained,
                                                                                  training,structure,
                                                                                  image_index,iters,
                                                                                  attack_type=i)
            ax11 = fig.add_subplot(rows,5, 5*j+1)
            ax11.imshow(original_img)
            ax11.set_title('Original Image. Training: ' + training + 'Noise:' +ifNoise+ vizmethod)
            ax1 = fig.add_subplot(rows,5, 5*j+2)
            ax1.imshow(heat)
            ax1.set_title('Learned Mask Color')
            ax2 = fig.add_subplot(rows,5, 5*j+3)
            ax2.imshow(mask)
            ax2.set_title('Learned Mask Gray')
            ax3 = fig.add_subplot(rows,5, 5*j+4)
            ax3.imshow(cam)
            ax3.set_title('Cam Result')
            ax9 = fig.add_subplot(rows,5, 5*j+5)
            ax9.bar(indices, orig_vals, align='center', alpha=0.5)
            ax9.set_title('Orignial Image Predictions')
            ax9.set_xticks(indices)
            ax9.set_xticklabels(orig_labs, rotation=45, ha="right")
            ax12 = fig.add_subplot(rows,5, 5*j+6)
            ax12.imshow(adversarialpic)
            ax12.set_title('Adversarial Image. Training: ' + training + 'Noise:' +ifNoise)
            ax5 = fig.add_subplot(rows,5, 5*j+7)
            ax5.imshow(heat2)
            ax6 = fig.add_subplot(rows,5, 5*j+8)
            ax6.imshow(mask2)
            ax6.set_title('Adversary Mask Gray')
            ax7 = fig.add_subplot(rows,5, 5*j+9)
            ax7.imshow(cam2)
            ax7.set_title('Adversary Cam Result')
            ax10 = fig.add_subplot(rows,5, 5*j+10)
            ax10.bar(indices, adver_vals, align='center', alpha=0.5)
            ax10.set_title('Adversary Image Predictions')
            ax10.set_xticks(indices)
            ax10.set_xticklabels(adver_labs, rotation=45, ha="right")
            j += 2

        elif vizmethod == 'GradCam':
            original_image,gray,color,result,adversarial,gray2,color2,result2,\
                indices,orig_labs,orig_vals,adver_labs,adver_vals = runGradCam(choose_network,isTrained,
                                                                               training,structure,
                                                                               image_index,attack_type=i)
            ax0 = fig.add_subplot(rows,5,5*j+1)
            ax0.imshow(original_image)
            ax0.set_title('Original Image. Training: ' + training + 'Noise:' +ifNoise +vizmethod)
            ax1 = fig.add_subplot(rows,5,5*j+2)
            ax1.imshow(gray)
            ax1.set_title('Cam Grasycale')
            ax2 = fig.add_subplot(rows,5,5*j+3)
            ax2.imshow(color)
            ax2.set_title('Cam HeatMap')
            ax3 = fig.add_subplot(rows,5,5*j+4)
            ax3.imshow(result)
            ax3.set_title('Cam Result')
            ax9 = fig.add_subplot(rows,5,5*j+5)
            ax9.bar(indices,orig_vals,align='center', alpha=0.5)
            ax9.set_title('Orignial Image Predictions')
            ax9.set_xticks(indices)
            ax9.set_xticklabels(orig_labs,rotation = 45,ha="right")
            ax12 = fig.add_subplot(rows,5,5*j+6)
            ax12.imshow(adversarial)
            ax12.set_title('Adversarial Image. Training: ' + training + 'Noise:' +ifNoise)
            ax4 = fig.add_subplot(rows,5,5*j+7)
            ax4.imshow(gray2)
            ax4.set_title('Adversary Cam Grasycale')
            ax5 = fig.add_subplot(rows,5,5*j+8)
            ax5.imshow(color2)
            ax5.set_title('Adversary Cam HeatMap')
            ax6 = fig.add_subplot(rows,5,5*j+9)
            ax6.imshow(result2)
            ax6.set_title('Adversary Cam Result')
            ax10 = fig.add_subplot(rows,5,5*j+10)
            ax10.bar(indices,adver_vals,align='center', alpha=0.5)
            ax10.set_title('Adversary Image Predictions')
            ax10.set_xticks(indices)
            ax10.set_xticklabels(adver_labs,rotation = 45,ha="right")
            j +=2

        elif vizmethod == 'GBP':
            original_image, colorgrads,graygrads,possal, negsal, \
            adversarial,colorgrads2,graygrads2,possal2,negsal2,\
                indices,orig_labs,orig_vals,adver_labs,adver_vals = runGBackProp(choose_network,isTrained,
                                                                                 training,structure,
                                                                                 image_index,attack_type=i)
            ax11 = fig.add_subplot(rows,6,6*j+1)
            ax11.imshow(original_image)
            ax11.set_title('Original Image. Training: ' + training + 'Noise:' +ifNoise +vizmethod)
            ax1 = fig.add_subplot(rows,6,6*j+2)
            ax1.imshow(colorgrads)
            ax1.set_title('Guided BP Color')
            ax2 = fig.add_subplot(rows,6, 6*j+3)
            ax2.imshow(graygrads)
            ax2.set_title( 'Guided BP Gray')
            ax3 = fig.add_subplot(rows,6, 6*j+4)
            ax3.imshow(possal)
            ax3.set_title('Positive Saliency')
            ax4 = fig.add_subplot(rows,6, 6*j+5)
            ax4.imshow(negsal)
            ax4.set_title('Negative Saliency')
            ax9 = fig.add_subplot(rows,6,6*j+6)
            ax9.bar(indices,orig_vals,align='center', alpha=0.5)
            ax9.set_title('Orignial Image Predictions')
            ax9.set_xticks(indices)
            ax9.set_xticklabels(orig_labs,rotation = 45,ha="right")
            ax12 = fig.add_subplot(rows,6,6*j+7)
            ax12.imshow(adversarial)
            ax12.set_title('Adversarial Image. Training: ' + training + 'Noise:' +ifNoise)
            ax5 = fig.add_subplot(rows,6, 6*j+8)
            ax5.imshow(colorgrads2)
            ax5.set_title('Adversarial Guided BP Color')
            ax6 = fig.add_subplot(rows,6, 6*j+9)
            ax6.imshow(graygrads2)
            ax6.set_title('Adversarial'+ 'Guided BP Gray')
            ax7 = fig.add_subplot(rows,6, 6*j+10)
            ax7.imshow(possal2)
            ax7.set_title('Adversarial ''Positive Saliency')
            ax8 = fig.add_subplot(rows,6, 6*j+11)
            ax8.imshow(negsal2)
            ax8.set_title('Adversarial'+'Negative Saliency')
            ax10 = fig.add_subplot(rows,6,6*j+12)
            ax10.bar(indices,adver_vals,align='center', alpha=0.5)
            ax10.set_title('Adversary Image Predictions')
            ax10.set_xticks(indices)
            ax10.set_xticklabels(adver_labs,rotation = 45,ha="right")
            j += 2

        elif vizmethod == 'GGradCam':
            original_image, guidedgrad, grayguidedgrad,\
            adversarial, guidedgrad2, grayguidedgrad2,\
                indices,orig_labs,orig_vals,adver_labs,adver_vals = runGGradCam(choose_network,isTrained,
                                                                                training,structure,
                                                                                image_index,attack_type=i)
            ax0 = fig.add_subplot(rows,4,4*j+1)
            ax0.imshow(original_image)
            ax0.set_title('Original Image. Training: ' + training + 'Noise:' +ifNoise +vizmethod)
            ax1 = fig.add_subplot(rows,4,4*j+2)
            ax1.imshow(guidedgrad)
            ax1.set_title('Guided Grad Cam')
            ax2 = fig.add_subplot(rows,4,4*j+3)
            ax2.imshow(grayguidedgrad)
            ax2.set_title('Guided Grad Cam Grasycale')
            ax9 = fig.add_subplot(rows,4,4*j+4)
            ax9.bar(indices,orig_vals,align='center', alpha=0.5)
            ax9.set_title('Orignial Image Predictions')
            ax9.set_xticks(indices)
            ax9.set_xticklabels(orig_labs,rotation = 45,ha="right")
            ax12 = fig.add_subplot(rows,4,4*j+5)
            ax12.imshow(adversarial)
            ax12.set_title('Adversarial Image. Training: ' + training + 'Noise:' +ifNoise)
            ax3 = fig.add_subplot(rows,4,64*j+6)
            ax3.imshow(guidedgrad2)
            ax3.set_title('Adversary Guided Grad Cam')
            ax4 = fig.add_subplot(rows,4,4*j+7)
            ax4.imshow(grayguidedgrad2)
            ax4.set_title('Adversary Guided Grad Cam Grasycale')
            ax10 = fig.add_subplot(rows,4,4*j+8)
            ax10.bar(indices,adver_vals,align='center', alpha=0.5)
            ax10.set_title('Adversary Image Predictions')
            ax10.set_xticks(indices)
            ax10.set_xticklabels(adver_labs,rotation = 45,ha="right")
            j += 2

        elif vizmethod == 'SmoothGrad':
            original_image,colorgrads,graygrads,\
            adversarial,colorgrads2,graygrads2,\
                indices,orig_labs,orig_vals,adver_labs,adver_vals = runsmoothGrad(choose_network,isTrained,
                                                                                  training,structure,
                                                                                  image_index,attack_type=i)
            ax0 = fig.add_subplot(rows,4,4*j+1)
            ax0.imshow(original_image)
            ax0.set_title('Original Image. Training: ' + training + 'Noise:' +ifNoise +vizmethod)
            ax1 = fig.add_subplot(rows,4,4*j+2)
            ax1.imshow(colorgrads)
            ax1.set_title('Smooth BP')
            ax2 = fig.add_subplot(rows,4,4*j+3)
            ax2.imshow(graygrads)
            ax2.set_title('Smooth BP Gray')
            ax9 = fig.add_subplot(rows,4,4*j+4)
            ax9.bar(indices,orig_vals,align='center', alpha=0.5)
            ax9.set_title('Orignial Image Predictions')
            ax9.set_xticks(indices)
            ax9.set_xticklabels(orig_labs,rotation = 45,ha="right")
            ax12 = fig.add_subplot(rows,4,4*j+5)
            ax12.imshow(adversarial)
            ax12.set_title('Adversarial Image. Training: ' + training + 'Noise:' +ifNoise)
            ax3 = fig.add_subplot(rows,4,4*j+6)
            ax3.imshow(colorgrads2)
            ax3.set_title('Adversary Smooth BP')
            ax4 = fig.add_subplot(rows,4, 4*j+7)
            ax4.imshow(graygrads2)
            ax4.set_title('Adversary Smooth BP Gray')
            ax10 = fig.add_subplot(rows,4,4*j+8)
            ax10.bar(indices,adver_vals,align='center', alpha=0.5)
            ax10.set_title('Adversary Image Predictions')
            ax10.set_xticks(indices)
            ax10.set_xticklabels(adver_labs,rotation = 45,ha="right")
            j += 2

        elif vizmethod == 'VanillaBP':
            original_image,vanilbp,grayvanilbp,\
            adversarial,vanilbp2,grayvanilbp2,\
                indices,orig_labs,orig_vals,adver_labs,adver_vals = runVanillaBP(choose_network,
                                                                                 isTrained,training,structure,
                                                                                 image_index,attack_type=i)
            ax0 = fig.add_subplot(rows,4,4*j+1)
            ax0.imshow(original_image)
            ax0.set_title('Original Image. Training: ' + training + 'Noise:' +ifNoise +vizmethod)
            ax1 = fig.add_subplot(rows,4,4*j+2)
            ax1.imshow(vanilbp)
            ax1.set_title('Vanilla BackProp')
            ax2 = fig.add_subplot(rows,4,4*j+3)
            ax2.imshow(grayvanilbp)
            ax2.set_title('Vanilla BackProp GrayScale')
            ax9 = fig.add_subplot(rows,4,4*j+4)
            ax9.bar(indices,orig_vals,align='center', alpha=0.5)
            ax9.set_title('Orignial Image Predictions')
            ax9.set_xticks(indices)
            ax9.set_xticklabels(orig_labs,rotation = 45,ha="right")
            ax12 = fig.add_subplot(rows,4,4*j+5)
            ax12.imshow(adversarial)
            ax12.set_title('Adversarial Image. Training: ' + training + 'Noise:' +ifNoise)
            ax3 = fig.add_subplot(rows,4,4*j+6)
            ax3.imshow(vanilbp2)
            ax3.set_title('Adversary Vanilla BackProp')
            ax4 = fig.add_subplot(rows,4,4*j+7)
            ax4.imshow(grayvanilbp2)
            ax4.set_title('Adversary Vanilla BackProp GrayScale')
            ax10 = fig.add_subplot(rows,4,4*j+8)
            ax10.bar(indices,adver_vals,align='center', alpha=0.5)
            ax10.set_title('Adversary Image Predictions')
            ax10.set_xticks(indices)
            ax10.set_xticklabels(adver_labs,rotation = 45,ha="right")
            j += 2

        n += 1




    fig.set_size_inches(32, 9*rows)
    fig.tight_layout()
    fig.savefig('Comparing/TrainingComp' +'_' +
                img_name +'_' +
                attackmethod +'_' +
                vizmethod +'('+
                structure + ' )', dpi=100)