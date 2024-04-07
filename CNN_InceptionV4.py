import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from InceptionV1 import SimplifiedInceptionV1
import requests
import os
from PIL import Image
import time
# Assuming 'inceptionv4' from 'inceptionv4.py' is already imported
from inceptionv4 import inceptionv4, InceptionV4

#way1
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

# Load datasets
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

# DataLoaders
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
model_weights_path = 'D:/course/CS767/6/hw/inceptionv4-8e4777a0.pth'
# Load Pre-Trained InceptionV4 Model
inceptionV4 = InceptionV4()
inceptionV4.load_state_dict(torch.load(model_weights_path, map_location=device))
inceptionV4 = inceptionV4.to(device)


# Manually load the weights
#

# Evaluation
inceptionV4.eval()
correct = 0
total = 0
start = time.time()
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        outputs = inceptionV4(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of Inception V4 on the test images: {accuracy}%')
print(f'Evaluation time: {time.time() - start} seconds')
#Accuracy of Inception V4 on the validation images: 0.00%

#way2
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


# res18 = models.resnet18(pretrained=True)
# res18.fc = nn.Linear(res18.fc.in_features, num_classes)
model_weights_path = 'D:/course/CS767/6/hw/inceptionv4-8e4777a0.pth'

# Load the model without pretrained weights
inceptionV4 = InceptionV4(num_classes=num_classes)

# Manually load the weights
# inceptionV4.load_state_dict(torch.load(model_weights_path, map_location=device))

# Then replace the last layer
inceptionV4.last_linear = nn.Linear(in_features=inceptionV4.last_linear.in_features,
                                    out_features=num_classes)

inceptionV4 = inceptionV4.to(device)


# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer_InceptionV4 = optim.Adam(inceptionV4.parameters())

start = time.time()
# Training Loop
num_epochs = 5  # Set the number of epochs
for epoch in range(num_epochs):
    for data in train_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Training VGG19
        optimizer_InceptionV4.zero_grad()
        outputs_InceptionV4 = inceptionV4(inputs)
        loss_InceptionV4 = criterion(outputs_InceptionV4, labels)
        loss_InceptionV4.backward()
        optimizer_InceptionV4.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: Inception V4 - {loss_InceptionV4.item()}")

print(f"InceptionV4, time=", time.time()-start)

# Evaluation Loop
with torch.no_grad():
    inceptionV4.eval()

    correct_InceptionV4, total_InceptionV4 = 0, 0

    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        # Evaluate VGG19
        outputs_InceptionV4 = inceptionV4(images)
        _, predicted_InceptionV4 = torch.max(outputs_InceptionV4.data, 1)
        total_InceptionV4 += labels.size(0)
        correct_InceptionV4 += (predicted_InceptionV4 == labels).sum().item()


        # Calculate and print accuracy for each model
    accuracy_InceptionV4 = 100 * correct_InceptionV4 / total_InceptionV4

    print(f'Accuracy of Inception V4 on the test images: {accuracy_InceptionV4}%')

# Epoch 1/5, Loss: Inception V4 - 0.9245947599411011
# Epoch 2/5, Loss: Inception V4 - 0.01889055222272873
# Epoch 3/5, Loss: Inception V4 - 0.36755186319351196
# Epoch 4/5, Loss: Inception V4 - 0.7356168031692505
# Epoch 5/5, Loss: Inception V4 - 0.005022209603339434
# InceptionV4, time= 382.72450852394104
# Accuracy of Inception V4 on the test images: 97.0%