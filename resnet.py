import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim


"""
resnet规律如下：
1:basic block:
    (0)width和channel——num的变化规律刚好相反（width缩小为2分之1时，channel——num扩大为2倍）
    (1)针对每一个layer的第一次：每一层两次conv，且第二层stride恒为1，除了layer1的第一层stride为1，剩下的第一层stride均为2
    注意：stride的计算方法为output channel size/input channel size
    （2）针对每一个layer的后面几次：stride恒为1
    （3）对于stride等于2时，shortcut需要做一次维度变换，stride等于1则直接一个空的sequential即可
"""
#定义网络
#首先定义几个基本模块
#定义basic block
class Basic_block(nn.Module):
    def __init__(self,input_channel_num,output_channel_size,stride=1):
        super().__init__()
        self.stride=output_channel_size/input_channel_num
        self.conv1=nn.Conv2d(input_channel_num,output_channel_size,kernel_size=3,stride=self.stride)
        self.bn=nn.BatchNorm2d(output_channel_size)
        self.conv2=nn.Conv2d(input_channel_num,output_channel_size,kernel_size=3,stride=1)

        if self.stride==1:
            self.shortcut=nn.Sequential()
        else :
            self.shortcut=nn.Sequential(
                nn.Conv2d(input_channel_num,output_channel_size,kernel_size=1,stride=2)
            )
    def forward(self,x):
        self.identity=x
        x=F.relu(self.bn(self.conv1(x)))
        x=self.bn(self.conv1(x))
        x+=self.shortcut(self.identity)
        x=F.relu(x)
        return x

class Resnet(nn.Module):
    def __init__(self,block_kind,block_num):
        super().__init__()
        input_picture_width=224
        self.fixedconvlayer=nn.Conv2d(3,64,kernel_size=7,stride=2)
        self.layer1=self.make_layer(block_kind,64,block_num[0],stride=1)
        self.layer2=self.make_layer(block_kind,64,block_num[1],stride=2)
        self.layer2 = self.make_layer(block_kind, 128, block_num[1], stride=2)
        self.layer2 = self.make_layer(block_kind, 256, block_num[1], stride=2)
        self.pool = nn.AvgPool2d(2, 2, padding=1)
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 10)
        )
    def make_layer(self,block_kind,input_channel_size,block_num,stride):
        strides=[stride]+[1]*(block_num-1)
        layers=[]

        for stride in strides:
            output_channel_size=input_channel_size*2
            layers.append(block_kind(input_channel_size,output_channel_size))
            input_channel_size=output_channel_size
        return nn.Sequential(*layers)
    def forward(self,x):
        x=F.relu(self.bn(self.fixedconvlayer(x)))
        x=F.max_pool2d(x,(2,2))
        self.layer1=self.make_layer(x)
        self.layer2=self.make_layer(x)
        self.layer3=self.make_layer(x)
        self.layer4=self.make_layer(x)
        x= x.view(x.size(0), -1)
        self.fc(x)
        return x
resnet18=Resnet(Basic_block,[2,2,2,2])

#load data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((224,224)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet18.parameters(), lr=0.001, momentum=0.9)
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = resnet18(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')





