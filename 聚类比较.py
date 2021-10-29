import torch
import torch.nn as nn
from sklearn import mixture
from sklearn import cluster
import numpy as np
import scipy
import os
import matplotlib.pyplot as plt
from sklearn import metrics
import torchvision.transforms as transforms
import PIL.Image as Image
from math import *
import os
from PIL import Image


# 聚类的函数，输入(B,HW,C)大小向量；输出为gmm mask，就是一堆0 0 0 1 1 2之类的标签，总长为HW
# 再往后就是用index对语义进行操作了
# 画图部分自己写下吧，看换个函数
class Args():
    def __init__(self):
        self.Nc=3
        self.Ns=3
        self.content="/Users/apple/Downloads/style transfer/result/wave.jpg"
        self.style="/Users/apple/Downloads/style transfer/result/wave.jpg"

        self.batch_size=1
def gmm_cluster(F, k, image):
    F = F.numpy()
    B, HW, C = F.shape


    gmm = np.zeros((B, HW))

    for i in range(B):
        model = mixture.GaussianMixture(n_components=k, covariance_type='full', random_state=1).fit(F[i])

        score = model.score(F[i])
        print("gmm_score",score)
        score=str(score)
        f = open("/Users/apple/Downloads/style transfer/结果.txt", "a")

        f.write("gmm_score:")
        f.write(score)
        f.write("\n")
        f.close()
        gmm[i]=model.predict(F[i])
    F = torch.tensor(F)
    gmm_ = gmm.reshape(int(sqrt(HW)), int(sqrt(HW)))

    gmm_distribution = gmm.reshape(HW, 1)
    # print("gmm_",gmm_)

    x = np.arange(int(sqrt(HW)))
    y = np.arange(int(sqrt(HW)))
    fig, ax = plt.subplots()
    # 显示定义colormap，可以用于保持色彩一致
    ax.pcolormesh(x, y, gmm_[:][::-1], cmap='viridis', vmin=0, vmax=k - 1)
    plt.savefig("/Users/apple/Downloads/style transfer/result/"+os.path.basename(image)+ '_gmm_cluster.jpg')
    plt.cla()
    plt.close(fig)
    return gmm


def kmeans(F, k, image):
    F = F.numpy()
    B, HW, C = F.shape

    gmm = np.zeros((B, HW))

    for i in range(B):
        model=cluster.KMeans(n_clusters=k, random_state=1) .fit(F[i])
        score = model.score(F[i])
        print("kmeans_score",score)
        score = str(score)
        f=open("/Users/apple/Downloads/style transfer/结果.txt","a")


        f.write("kmeans_score:")
        f.write(score)
        f.write("\n")

        f.close()

        gmm[i]=model.predict(F[i])
    F = torch.tensor(F)
    gmm_ = gmm.reshape(int(sqrt(HW)), int(sqrt(HW)))

    gmm_distribution = gmm.reshape(HW, 1)
    # print("gmm_",gmm_)

    x = np.arange(int(sqrt(HW)))
    y = np.arange(int(sqrt(HW)))
    fig, ax = plt.subplots()
    # 显示定义colormap，可以用于保持色彩一致
    ax.pcolormesh(x, y, gmm_[:][::-1], cmap='viridis', vmin=0, vmax=k - 1)
    plt.savefig("/Users/apple/Downloads/style transfer/result/" + os.path.basename(image)+'_kmeans_cluster.jpg')
    plt.cla()
    plt.close(fig)
    return gmm


# 聚类相似度计算函数，现在是每个content聚类和多个style聚类加权对应
# 注意这个discrepancy的差异得足够大（类之间discrepancy差别），毕竟MMD原来是做假设检验的，要避免存伪错误
# 后面的similarity是discrepancy加softmax
# Fc Fs要给同层次的
def cluster_matching(Fc, Fs, args):
    Fc = Fc.view(Fc.shape[0], Fc.shape[1], -1).permute(0, 2, 1)
    Fs = Fs.view(Fs.shape[0], Fs.shape[1], -1).permute(0, 2, 1)
    kmeans_c = kmeans(Fc, args.Nc, args.content)
    kmeans_s = kmeans(Fs, args.Ns, args.style)

    gmm_c = gmm_cluster(Fc, args.Nc, args.content)
    gmm_s = gmm_cluster(Fs, args.Ns, args.style)
    return 1




vgg19 = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1,
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


# 注意一下序号顺序
class Encoder(nn.Module):
    def __init__(self, pretrained_path=None, requires_grad=False):
        super(Encoder, self).__init__()
        self.vgg = vgg19

        if pretrained_path is not None:
            self.vgg.load_state_dict(torch.load(pretrained_path))

        if requires_grad == False:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.vgg[:4](x)
        relu2_1 = self.vgg[4:11](relu1_1)
        relu3_1 = self.vgg[11:18](relu2_1)
        relu4_1 = self.vgg[18:31](relu3_1)
        relu5_1 = self.vgg[31:44](relu4_1)
        return [relu1_1, relu2_1, relu3_1, relu4_1, relu5_1]


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


def test_transform():
    transform_list = [
        transforms.ToTensor()
    ]

    transform = transforms.Compose(transform_list)
    return transform




#x=cluster_matching(content_features[4].cpu(),style_features[4].cpu(), args)#relu5_1
# 源目录

input ="/Users/apple/Downloads/style transfer/test-image"

# 输出目录

output = "/Users/apple/Downloads/style transfer/result/"

def modify():
    # 切换目录
    os.chdir(input)

    # 遍历目录下所有的文件
    for image_name in os.listdir(os.getcwd()):
        if image_name==".DS_Store":
            pass
        else:

            im = Image.open(os.path.join(input, image_name))
            width, height = im.size
            width = max(width, height)
            newsize = (width, width)
            im = im.resize(newsize)
            im=im.convert('RGB')

            im.save(output+image_name)
    for image_name in os.listdir(os.getcwd()):
        if image_name=='.DS_Store':
            pass
        else:
            args = Args()
            args.content=os.path.join(input, image_name)
            args.style=os.path.join(input, image_name)
            f=open("/Users/apple/Downloads/style transfer/结果.txt",'a')
            f.write("image_name")
            print("already write")
            f.write(image_name)
            f.close()
            tf = test_transform()

            Ic = tf(Image.open(args.content))

            Is = tf(Image.open(args.style))

            Ic = Ic.unsqueeze(0)

            Is = Is.unsqueeze(0)
            encoder = Encoder(pretrained_path='/Users/apple/Downloads/style transfer/vgg19.pth', requires_grad=False)
            content_features = encoder(Ic)
            print(content_features)
            style_features = encoder(Is)

            #x = cluster_matching(content_features[3].cpu(), style_features[3].cpu(), args)  # relu4_1

            x=cluster_matching(content_features[4].cpu(),style_features[4].cpu(), args)#relu5_1
            # 源目录

modify()

"""
例子用法
def forward(self, Ic, Is, args):
    content_features = self.encoder(Ic)
    style_features = self.encoder(Is)

    ind_c_relu4_1, ind_s_relu4_1, similarity_relu4_1 = cluster_matching(content_features[3].cpu(), style_features[3].cpu(), args)
    ind_c_relu5_1, ind_s_relu5_1, similarity_relu5_1 = cluster_matching(content_features[4].cpu(), style_features[4].cpu(), args)

    # ind_c_relu4_1[b][i]是对应content聚类中第b个batch的第i类的index，可以输出看一下
    # similarity[b][i][j]代表第b个batch第i个content cluster和第j个style cluster的相似度
"""
