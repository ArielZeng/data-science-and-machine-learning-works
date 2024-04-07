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

# Transformations applied to the dataset
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Paths to the test directory
test_dir = 'D:/course/CS767/6/hw/tiny-imagenet-200/tiny-imagenet-200/val'  # Replace with your test directory path

# Load the entire test dataset
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the pre-trained VGG19 model
vgg19 = models.vgg19(pretrained=True)
vgg19 = vgg19.to(device)

# Evaluation Loop
with torch.no_grad():
    vgg19.eval()  # Set the model to evaluation mode

    correct, total = 0, 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = vgg19(images)
        _, predicted = torch.max(outputs.data, 1)

        # Update the total and correct counts
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Calculate and print accuracy
    accuracy = 100 * correct / total
    print(f'Accuracy of the pre-trained VGG19 on the test images: {accuracy}%')
#Accuracy of the pre-trained VGG19 on the test images: 0.08%

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
vgg19 = models.vgg19()
vgg19.classifier[6] = nn.Linear(vgg19.classifier[6].in_features, num_classes)
vgg19 = vgg19.to(device)


# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer_vgg19 = optim.Adam(vgg19.parameters())

start = time.time()
# Training Loop
num_epochs = 25  # Set the number of epochs
for epoch in range(num_epochs):
    for data in train_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Training VGG19
        optimizer_vgg19.zero_grad()
        outputs_vgg19 = vgg19(inputs)
        loss_vgg19 = criterion(outputs_vgg19, labels)
        loss_vgg19.backward()
        optimizer_vgg19.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: VGG19 - {loss_vgg19.item()}")

print(f"vgg19, time=", time.time()-start)

# Evaluation Loop
with torch.no_grad():
    vgg19.eval()

    correct_vgg19, total_vgg19 = 0, 0

    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        # Evaluate VGG19
        outputs_vgg19 = vgg19(images)
        _, predicted_vgg19 = torch.max(outputs_vgg19.data, 1)
        total_vgg19 += labels.size(0)
        correct_vgg19 += (predicted_vgg19 == labels).sum().item()


        # Calculate and print accuracy for each model
    accuracy_vgg19 = 100 * correct_vgg19 / total_vgg19

    print(f'Accuracy of VGG19 on the test images: {accuracy_vgg19}%')

"""
Epoch 1/5, Loss: VGG19 - 0.6941019296646118
Epoch 2/5, Loss: VGG19 - 0.6889247298240662
Epoch 3/5, Loss: VGG19 - 0.6904973983764648
Epoch 4/5, Loss: VGG19 - 0.6986040472984314
Epoch 5/5, Loss: VGG19 - 0.6948384046554565
vgg19, time= 696.8499584197998
Accuracy of VGG19 on the test images: 50.0%
"""
