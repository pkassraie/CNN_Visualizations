import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from tqdm import tqdm


# Model
class AttackPGD(nn.Module):
    def __init__(self, basic_net, config):
        super(AttackPGD, self).__init__()
        self.basic_net = basic_net
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        assert config['loss_func'] == 'xent', 'Only xent supported for now.'

    def forward(self, inputs, targets):
        if not args.attack:
            return self.basic_net(inputs), inputs

        x = inputs.detach()
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(self.num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.basic_net(x)
                loss = F.cross_entropy(logits, targets, size_average=False)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.step_size*torch.sign(grad.detach())
            x = torch.min(torch.max(x, inputs - self.epsilon), inputs + self.epsilon)
            x = torch.clamp(x, 0, 1)

        return self.basic_net(x), x

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    iterator = tqdm(trainloader, ncols=0, leave=False)
    for batch_idx, (inputs, targets) in enumerate(iterator):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, pert_inputs = net(inputs, targets)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        iterator.set_description(str(predicted.eq(targets).sum().item()/targets.size(0)))

    acc = 100.*correct/total
    print('Train acc:', acc)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        iterator = tqdm(testloader, ncols=0, leave=False)
        for batch_idx, (inputs, targets) in enumerate(iterator):
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.no_grad():
                outputs, pert_inputs = net(inputs, targets)
                loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            iterator.set_description(str(predicted.eq(targets).sum().item()/targets.size(0)))

            if args.visualize and batch_idx == 0:
                if not os.path.isdir('viz'): os.mkdir('viz')
                torchvision.utils.save_image(inputs, 'viz/{}_clean.jpg'.format(epoch))
                torchvision.utils.save_image(pert_inputs, 'viz/{}_pert.jpg'.format(epoch))

    # Save checkpoint.
    acc = 100.*correct/total
    print('Val acc:', acc)
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': basic_net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_advers_'+modelName+'.t7')
        if not os.path.isdir('trainedmodels'):
            os.mkdir('trainedmodels')
        torch.save(net.state_dict(), './trainedmodels/'+'Adversarial_'+modelName)
        best_acc = acc



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Adversarial Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--attack', '-a', action='store_true', help='attack')
parser.add_argument('--visualize', '-v', action='store_true', help='visualize some perturbed images')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # Normalization messes with l-inf bounds.
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print('==> Building model..')
basic_net = torchvision.models.resnet50(pretrained=True)
modelName = 'ResNet50'

## Change the last layer since we don't have 1000 classes but we want to used a pretrained model
if modelName == 'ResNet50':
    num_ftrs = basic_net.fc.in_features
    basic_net.fc = nn.Linear(num_ftrs, len(classes))
elif modelName =='VGG19':
    num_ftrs = basic_net.classifier[6].in_features
    basic_net.classsifier[6] = nn.Linear(num_ftrs,len(classes))

basic_net = basic_net.to(device)
# From https://github.com/MadryLab/cifar10_challenge/blob/master/config.json
config = {
    'epsilon': 8.0 / 255,
    'num_steps': 10,
    'step_size': 2.0 / 255,
    'random_start': True,
    'loss_func': 'xent',
}

net = AttackPGD(basic_net, config)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt_advers_'+modelName+'.t7')
    basic_net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
if modelName == 'ResNet50':
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50, 100], gamma=0.1)
elif modelName == 'VGG19':
    optimizer = optim.SGD(net.parameters(), lr=args.lr / 10, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 100], gamma=0.1)


for epoch in range(start_epoch, start_epoch+150):
    scheduler.step()
    train(epoch)
    test(epoch)