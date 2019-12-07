import pandas as pd
from glob import glob
import os
from shutil import copyfile
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from numpy.random import permutation
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18,resnet34
from torchvision.models.inception import inception_v3
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pickle
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda')
is_cuda = torch.cuda.is_available()
is_cuda
# nvidia-smi查看内存



def imshow(inp, cmap=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, cmap)
    plt.savefig('cat.jpg')


class FeaturesDataset(Dataset):

    def __init__(self, featlst, labellst):
        self.featlst = featlst
        self.labellst = labellst

    def __getitem__(self, index):
        return (self.featlst[index], self.labellst[index])

    def __len__(self):
        return len(self.labellst)


class FullyConnectedModel(nn.Module):

    def __init__(self, in_size, out_size):
        super().__init__()
        self.fc = nn.Linear(in_size, out_size)

    def forward(self, inp):
        out = self.fc(inp)     # inp.shape = torch.Size([64, 512])
        return out             # out.shape = torch.Size([64, 2])


classes = 2
fc_in_size = 512      # 不是书上的8192
fc = FullyConnectedModel(fc_in_size,classes)
if is_cuda:
    fc = fc.cuda()
optimizer = optim.Adam(fc.parameters(),lr=0.0001)
# optimizer = optim.SGD(fc.parameters(),lr=0.0001)
def fit(epoch, model, data_loader, phase='training', volatile=False):
    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()
        volatile = True
    running_loss = 0.0
    running_correct = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        if is_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile), Variable(target)      # data.shape = torch.Size([64, 512])
        if phase == 'training':
            optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)      # 用于反向传播

        loss_ = F.cross_entropy(output,target,size_average=False)     # 参数size_average在默认情况下，批处理中的每个损失元素的平均损失。如果字段size_average设置为False，则对每个批的损失进行求和。
        loss__ = 0.0
        loss__ += F.cross_entropy(output,target,size_average=False)
        running_loss += F.cross_entropy(output, target, size_average=False).item()     # 计算损失和
        preds = output.data.max(dim=1, keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()     # running_correct为target为1的个数，并且是累加和
        if phase == 'training':
            loss.backward()
            optimizer.step()
    t = batch_idx     # 训练集train里的t = 359，验证集validation里的t = 31
    loss = running_loss / len(data_loader.dataset)
    accuracy = 100. * running_correct / len(data_loader.dataset)

    print(
        f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy:{10}.{4}}')
    return loss, accuracy

def train():
    data_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # For DogsandCats dataset
    train_dset = ImageFolder('/home/ZhangXueLiang/LiMiao/dataset/dogsandcats/train', transform=data_transform)
    val_dset = ImageFolder('/home/ZhangXueLiang/LiMiao/dataset/dogsandcats/valid', transform=data_transform)
    # imshow(train_dset[150][0])
    train_loader = DataLoader(train_dset, batch_size=32, shuffle=False, num_workers=3)
    val_loader = DataLoader(val_dset, batch_size=32, shuffle=False, num_workers=3)
    my_resnet = resnet34(pretrained=True)

    if is_cuda:
        my_resnet = my_resnet.cuda()

    m = nn.Sequential(*list(my_resnet.children())[:-1])     # 去掉了my_resnet的最后一层线性层，并且给每一层编号

    # For training data

    # Stores the labels of the train data
    trn_labels = []

    # Stores the pre convoluted features of the train data
    trn_features = []

    # Iterate through the train data and store the calculated features and the labels
    for d, la in train_loader:
        o = m(Variable(d.cuda()))
        o = o.view(o.size(0), -1)
        trn_labels.extend(la)
        trn_features.extend(o.cpu().data)

    # For validation data

    # Iterate through the validation data and store the calculated features and the labels
    val_labels = []
    val_features = []
    for d, la in val_loader:
        o = m(Variable(d.cuda()))
        o = o.view(o.size(0), -1)
        val_labels.extend(la)
        val_features.extend(o.cpu().data)

    # Creating dataset for train and validation
    trn_feat_dset = FeaturesDataset(trn_features, trn_labels)
    val_feat_dset = FeaturesDataset(val_features, val_labels)

    # Creating data loader for train and validation
    trn_feat_loader = DataLoader(trn_feat_dset, batch_size=64, shuffle=True)
    val_feat_loader = DataLoader(val_feat_dset, batch_size=64)

    train_losses, train_accuracy = [], []
    val_losses, val_accuracy = [], []
    for epoch in range(1, 10):
        epoch_loss, epoch_accuracy = fit(epoch, fc, trn_feat_loader, phase='training')
        val_epoch_loss, val_epoch_accuracy = fit(epoch, fc, val_feat_loader, phase='validation')
        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)
        val_losses.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)



def main():
    # resnet = resnet34(pretrained=True)
    # print(f'resnet34{resnet}')
    train()

    pass


if __name__ == '__main__':
    main()

