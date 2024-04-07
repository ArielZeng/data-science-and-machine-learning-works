import os
from datetime import time

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch import nn, optim
from torch.utils.data import DataLoader
from PIL import Image

#Way1
# Set the device to GPU if CUDA is available, otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load and Transform the Dataset
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Paths to the train and test directories
train_dir = 'D:/course/CS767/6/hw/tiny-imagenet-200/tiny-imagenet-200/train'
test_dir = 'D:/course/CS767/6/hw/tiny-imagenet-200/tiny-imagenet-200/val'

# Load datasets
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

# DataLoaders
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load Pre-Trained ResNet50V2 Model
res50v2 = models.resnet50(pretrained=True).to(device)

# Evaluation
res50v2.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        outputs = res50v2(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the network on the test images: {accuracy}%')
#Accuracy of the network on the test images: 0.05%

#Way2
# Set the device to GPU if CUDA is available, otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load and Transform the Dataset
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# # Paths to the train and test directories
train_dir = 'D:/course/CS767/6/hw/tiny-imagenet-200/tiny-imagenet-200/train'
test_dir = 'D:/course/CS767/6/hw/tiny-imagenet-200/tiny-imagenet-200/val'
#
# # Load datasets
# train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
# test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
#
# # DataLoaders
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


###our small dataset

# 获取前 2 个子目录
subdirs = os.listdir(train_dir)
subdirs = subdirs[:2]  # 保留前 2 个子目录
subdirs = [os.path.join(train_dir, subdir) for subdir in subdirs]

# 创建自定义数据集
# 注意：这里使用了一个简单的方法，直接将子目录的图像放入一个列表中
train_data = []
for subdir in subdirs:
    print(subdir)
    for img_name in os.listdir(os.path.join(subdir, "images")):
        img_path = os.path.join(subdir, "images", img_name)
        img = Image.open(img_path).convert('RGB')
        img = transform(img)
        label = subdirs.index(subdir)
        train_data.append((img, label))

# DataLoader
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)



# 测试集路径和 ground truth 文件路径
test_dir = test_dir # 替换为实际路径
gt_file = os.path.join(test_dir, 'val_annotations.txt')  # 替换为实际路径

# 要测试的两个类别的标识符
trained_classes = ['n01443537', 'n01629819']

# 读取 ground truth 文件并筛选图像
selected_images = []
with open(gt_file, 'r') as file:
    for line in file:
        parts = line.strip().split('\t')
        if parts[1] in trained_classes:
            selected_images.append((parts[0], trained_classes.index(parts[1])))

# 创建测试集
test_data = []
for img_name, label in selected_images:
    img_path = os.path.join(test_dir, "images", img_name)
    img = Image.open(img_path).convert('RGB')
    img = transform(img)  # 使用前面定义的 transform
    test_data.append((img, label))

# DataLoader
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Load and Modify Pre-Trained Models
num_classes = 2  # 200
res50 = models.resnet50(pretrained=True)
res50.fc = nn.Linear(res50.fc.in_features, num_classes)
res50 = res50.to(device)


# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer_res50 = optim.Adam(res50.parameters())

start = time.time()
# Training Loop
num_epochs = 5  # Set the number of epochs
for epoch in range(num_epochs):
    for data in train_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Training VGG19
        optimizer_res50.zero_grad()
        outputs_res50 = res50(inputs)
        loss_res50 = criterion(outputs_res50, labels)
        loss_res50.backward()
        optimizer_res50.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: res50 - {loss_res50.item()}")

print(f"res50, time=", time.time()-start)

# Evaluation Loop
with torch.no_grad():
    res50.eval()

    correct_res50, total_res50 = 0, 0

    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        # Evaluate VGG19
        outputs_res50 = res50(images)
        _, predicted_res50 = torch.max(outputs_res50.data, 1)
        total_res50 += labels.size(0)
        correct_res50 += (predicted_res50 == labels).sum().item()


        # Calculate and print accuracy for each model
    accuracy_res50 = 100 * correct_res50 / total_res50

    print(f'Accuracy of res50 on the test images: {accuracy_res50}%')

# Epoch 1/5, Loss: res50 - 0.18396206200122833
# Epoch 2/5, Loss: res50 - 0.04171460494399071
# Epoch 3/5, Loss: res50 - 0.011209889315068722
# Epoch 4/5, Loss: res50 - 0.052293770015239716
# Epoch 5/5, Loss: res50 - 0.04790140688419342
# res50, time= 254.13953351974487
# Accuracy of res50 on the test images: 98.0%

