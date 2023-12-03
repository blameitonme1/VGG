import torch
from torch import nn
from d2l import torch as d2l

def vgg_block(num_convs, in_channels, out_channels):
    # VGG是使用块的网络，现在定义VGG块
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU()) # 别忘了加入激活函数
        in_channels = out_channels
    # 现在layer就是一系列的卷积层
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2)) # 最大池化层，分辨率减半
    return nn.Sequential(*layers)
# 一个超参数，决定VGG块有多少卷积层和输出通道数
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    # 构建卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels ,out_channels))
        in_channels = out_channels
    # 全连接层部分
    return nn.Sequential(
        *conv_blks, nn.Flatten(), # 因为要使用全连接层，需要把2d数据拉平
        # 注意还加入了dropout控制模型复杂度
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(p=0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
        nn.Linear(4096, 10)
    )

net = vgg(conv_arch)
X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)
# 构建通道数较小的conv_arch，足够训练fashionMNIST了
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)
lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

