"""
pytorch 完成简单的图像识别分类任务
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets


class ImageClassify:

    def __init__(self, size=(50, 50), model=None):
        """
        图像分类器
        :param size: 图像压缩后的尺寸
        :param model: 模型，如果为空，默认使用 ResNet18 模型
        """
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        if model is None:
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.freeze_model(self.model)

    @staticmethod
    def freeze_model(model):
        """
        冻结模型参数。
        :param model: 模型
        :return:
        """
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, 2)

    def load_data(self, path, batch_size=64, shuffle=True):
        """
        加载数据集
        :param batch_size: 数据分批大小
        :param path: 数据集路径
        :param shuffle: 是否打乱数据集
        :return: 数据迭代器
        """
        data = datasets.ImageFolder(path, transform=self.transform)
        loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
        return loader

    def train(self, loader: DataLoader, n_iter=100, lr=0.001):
        """
        训练模型。
        :param loader: 训练集路径
        :param n_iter: 迭代次数
        :param lr: 学习率
        :return:
        """
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.fc.parameters(), lr=lr)
        for epoch in range(n_iter):
            losses = 0
            for images, labels in loader:
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                losses += loss.item()
            print(f'[{epoch}/{n_iter}] 损失率:{losses}')

    def predict(self, loader: DataLoader):
        """
        预测并评估模型。
        :param loader: 测试集路径
        :return:
        """
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                print('预测', predicted)
                print('实际', labels)
                correct += np.array(predicted == labels).sum().item()
        print('模型准确率: {:.2f}%'.format(100 * correct / total))

    def save_model(self, filepath):
        """
        保存模型
        :param filepath: 模型保存路径
        """
        torch.save(self.model.state_dict(), filepath)
        print("模型保存至：", filepath)

    def load_model(self, filepath):
        """
        加载模型
        :param filepath: 模型文件路径
        """
        self.model.load_state_dict(torch.load(filepath))
        print("模型已加载：", filepath)
