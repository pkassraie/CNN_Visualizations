import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image
def imshow(img):
    original_image = prep_img.numpy().transpose(0, 2, 3, 1)[0, :, :, :]
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    original_image = std * original_image + mean
    original_image = np.clip(original_image, 0, 1)
    plt.imshow(original_image)

print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
if __name__ == '__main__':
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # get some random training images
    prep_img, target_class= next(iter(testloader))

    # show images
    imshow(torchvision.utils.make_grid(prep_img))
    plt.show()
    # print labels
    print(classes[target_class])