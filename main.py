import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision import transforms
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle

import models as n

LABEL_CLASSES = [0, 1, 2, 7, 8, 9]
UNLABEL_CLASSES = [0, 1, 2, 3, 4, 5, 6]


# 自定义数据集类，用于加载筛选后的数据
class SelfCIFAR10(Dataset):
    def __init__(self, dataset, classes=None):
        self.data = []
        if classes is not None:
            for item in dataset:
                if item[1] in classes:
                    self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def appendata(self, new_data, ):
        self.data.append(new_data)


# 按要求分割CIFAR10
def genDataset():
    # 加载CIFAR-10数据集
    transform = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    cifar_dataset = datasets.CIFAR10(
        root='../datasets', train=True, transform=transform, download=True)

    # 拆分数据集为已知类和未知类
    label_dataset = SelfCIFAR10(cifar_dataset, LABEL_CLASSES)
    unlabel_dataset = SelfCIFAR10(cifar_dataset, UNLABEL_CLASSES)
    with open('splited_data/label_dataset.pkl', 'wb') as file:
        pickle.dump(label_dataset, file)
    with open('splited_data/unlabel_dataset.pkl', 'wb') as file:
        pickle.dump(unlabel_dataset, file)
    pass


# 先进行监督训练，保存参数
def trainSupvNet():
    with open('splited_data/label_dataset.pkl', 'rb') as file:
        label_dataset = pickle.load(file)
    label_loader = DataLoader(label_dataset, batch_size=128, shuffle=True)

    net = n.ResNet18()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    net.train()
    for epoch in range(10):  # loop over the dataset multiple times
        train_loss = 0
        correct = 0
        total = 0
        for i, (inputs, targets) in enumerate(label_loader):
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print(
                f'Loss: {train_loss / (i + 1)} | Acc: {100. * correct / total} {correct} {total}')

        scheduler.step()

    print('Finished Training')

    PATH = '../pretrained/cifar_net.pth'
    torch.save(net.state_dict(), PATH)
    pass


# 测试监督训练的准确度
def test(epoch):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    testset = datasets.CIFAR10(
        root='../datasets', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    model = n.ResNet18()
    model.load_state_dict(torch.load('../pretrained/cifar_net.pth'))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            outputs = model(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    print(f'acc: {acc}')


# 深度嵌入聚类模型
class DEC(nn.Module):
    def __init__(self, n_clusters, n_features):
        super(DEC, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, 10)  # 10是嵌入维度
        )
        self.cluster_layer = nn.Sequential(
            nn.Linear(10, n_clusters)
        )

    def forward(self, x):
        x = self.encoder(x)
        features = x
        x = self.cluster_layer(x)
        return x, features


def trainDEC(train_loader):
    n_clusters = len(UNLABEL_CLASSES)  # 选择聚类簇的数量
    n_features = 32 * 32 * 3  # CIFAR-10 图像尺寸

    model = DEC(n_clusters, n_features).cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.5)
    criterion = nn.KLDivLoss()

    model.train()
    for epoch in range(10):
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.view(data.size(0), -1).cuda()
            optimizer.zero_grad()
            outputs, _ = model(data)
            loss = criterion(torch.log(outputs), outputs).mean()
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch}  Loss: {loss}')

    # 获取嵌入表示
    model.eval()
    embeddings = []
    with torch.no_grad():
        for data, _ in train_loader:
            data = data.view(data.size(0), -1).cuda()
            _, features = model(data)
            embeddings.append(features)
    embeddings = torch.cat(embeddings).cpu().numpy()

    # 使用 K-Means 进行聚类
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    kmeans.fit(embeddings)

    # 打印每个簇的样本数量
    print("Cluster sizes:", np.bincount(kmeans.labels_))

    # 聚类结果
    print("Cluster labels:", kmeans.labels_)
    pass


# 再进行聚类
def cluster():
    with open('splited_data/unlabel_dataset.pkl', 'rb') as file:
        unlabel_dataset = pickle.load(file)
    unlabel_loader = DataLoader(unlabel_dataset, batch_size=64, shuffle=True)

    n_clusters = len(UNLABEL_CLASSES)  # 选择聚类簇的数量
    n_features = 32 * 32 * 3  # CIFAR-10 图像尺寸

    dec = DEC(n_clusters, n_features).cuda()
    optimizer = optim.Adam(dec.parameters(), lr=0.01, weight_decay=0.5)
    criterion = nn.KLDivLoss()

    dec.train()
    for epoch in range(10):
        for batch_idx, (data, _) in enumerate(unlabel_loader):
            data = data.view(data.size(0), -1).cuda()
            optimizer.zero_grad()
            outputs, _ = dec(data)
            loss = criterion(torch.log(outputs), outputs).mean()
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch}  Loss: {loss}')

    # 获取嵌入表示
    dec.eval()
    embeddings = []
    with torch.no_grad():
        for data, _ in unlabel_loader:
            data = data.view(data.size(0), -1).cuda()
            _, features = dec(data)
            embeddings.append(features)
    embeddings = torch.cat(embeddings).cpu().numpy()

    # 使用 K-Means 进行聚类
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    kmeans.fit(embeddings)

    # 打印每个簇的样本数量和聚类结果
    print(
        f"Cluster sizes: {np.bincount(kmeans.labels_)}, Cluster labels: {kmeans.labels_}")
    cluster_labels = kmeans.labels_

    # 按类拆分聚类后的数据
    cluster_data = []
    for i in range(n_clusters):
        cluster_data.append(SelfCIFAR10([]))
    for i, label in enumerate(cluster_labels):
        data = unlabel_dataset[i]
        cluster_data[label].appendata(data)

    model = n.ResNet18()
    model.load_state_dict(torch.load('../pretrained/cifar_net.pth'))
    model.eval()

    # 依次将每簇的数据丢进监督网络，给softmax结果趋向one-hot的数据打上伪标签
    for i, dataset in enumerate(cluster_data):
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        outputs_list = []
        with torch.no_grad():
            for data in loader:
                inputs, labels = data
                outputs = model(inputs)
                outputs_list.append(outputs)
        all_outputs = torch.cat(outputs_list, dim=0)
        softmax_outputs = F.softmax(all_outputs, dim=1)
        result = F.softmax(torch.mean(softmax_outputs, dim=0))
        print(f'cluster {i}: {result}')

        # if torch.max(result, dim=0) >= 0.5:
        #     for

    pass


genDataset()
trainSupvNet()
cluster()
