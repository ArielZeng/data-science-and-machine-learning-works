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

# Paths to the test directories
test_dir = 'D:/course/CS767/6/hw/tiny-imagenet-200/tiny-imagenet-200/val'

# Load the test dataset
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the pre-trained VGG13 model
vgg13 = models.vgg13(pretrained=True)
vgg13 = vgg13.to(device)

# Evaluation Loop
with torch.no_grad():
    vgg13.eval()

    correct_vgg13, total_vgg13 = 0, 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        # Evaluate VGG13
        outputs_vgg13 = vgg13(images)
        _, predicted_vgg13 = torch.max(outputs_vgg13.data, 1)
        total_vgg13 += labels.size(0)
        correct_vgg13 += (predicted_vgg13 == labels).sum().item()

    # Calculate and print accuracy
    accuracy_vgg13 = 100 * correct_vgg13 / total_vgg13

    print(f'Accuracy of VGG13 on the test images: {accuracy_vgg13}%')
#Accuracy of VGG13 on the test images: 0.01%

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
vgg13 = models.vgg13()
vgg13.classifier[6] = nn.Linear(vgg13.classifier[6].in_features, num_classes)
vgg13 = vgg13.to(device)


# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer_vgg13 = optim.Adam(vgg13.parameters())

start = time.time()
# Training Loop
num_epochs = 5  # Set the number of epochs
for epoch in range(num_epochs):
    for data in train_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Training VGG19
        optimizer_vgg13.zero_grad()
        outputs_vgg13 = vgg13(inputs)
        loss_vgg13 = criterion(outputs_vgg13, labels)
        loss_vgg13.backward()
        optimizer_vgg13.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: VGG13 - {loss_vgg13.item()}")

print(f"vgg13, time=", time.time()-start)

# Evaluation Loop
with torch.no_grad():
    vgg13.eval()

    correct_vgg13, total_vgg13 = 0, 0

    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        # Evaluate VGG19
        outputs_vgg13 = vgg13(images)
        _, predicted_vgg13 = torch.max(outputs_vgg13.data, 1)
        total_vgg13 += labels.size(0)
        correct_vgg13 += (predicted_vgg13 == labels).sum().item()


        # Calculate and print accuracy for each model
    accuracy_vgg13 = 100 * correct_vgg13 / total_vgg13

    print(f'Accuracy of VGG13 on the test images: {accuracy_vgg13}%')

"""
Epoch 1/5, Loss: VGG13 - 0.3104260563850403
Epoch 2/5, Loss: VGG13 - 0.08233198523521423
Epoch 3/5, Loss: VGG13 - 0.05809102579951286
Epoch 4/5, Loss: VGG13 - 0.4389575123786926
Epoch 5/5, Loss: VGG13 - 1.3858008287570556e-06
vgg13, time= 669.9690737724304
Accuracy of VGG19 on the test images: 96.0%
"""