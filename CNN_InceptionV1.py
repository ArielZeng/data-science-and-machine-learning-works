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
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from PIL import Image
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

# Paths to the validation directory
val_dir = 'D:/course/CS767/6/hw/tiny-imagenet-200/tiny-imagenet-200/val'

# Load validation dataset
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

# DataLoader for validation dataset
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load Pre-Trained Inception V1 Model
# Note: Replace this with the correct way to load your pre-trained Inception V1 model
# For example, if you have a custom function to load it, use that
inceptionV1 = SimplifiedInceptionV1()  # Replace with your model loading function
inceptionV1 = inceptionV1.to(device)

# Evaluation Loop
inceptionV1.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in val_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        outputs = inceptionV1(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Calculate and print accuracy
accuracy = 100 * correct / total
print(f'Accuracy of Inception V1 on the validation images: {accuracy}%')
#Accuracy of Inception V1 on the validation images: 0.02%

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
inceptionV1 = SimplifiedInceptionV1(num_classes=num_classes)

# Manually load the weights
# inceptionV4.load_state_dict(torch.load(model_weights_path, map_location=device))

# Then replace the last layer
inceptionV1.last_linear = nn.Linear(in_features=inceptionV1.last_linear.in_features,
                                    out_features=num_classes)

inceptionV1 = inceptionV1.to(device)


# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer_inceptionV1 = optim.Adam(inceptionV1.parameters())

start = time.time()
# Training Loop
num_epochs = 5  # Set the number of epochs
for epoch in range(num_epochs):
    for data in train_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Training VGG19
        optimizer_inceptionV1.zero_grad()
        outputs_inceptionV1 = inceptionV1(inputs)
        loss_inceptionV1 = criterion(outputs_inceptionV1, labels)
        loss_inceptionV1.backward()
        optimizer_inceptionV1.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: Inception V1 - {loss_inceptionV1.item()}")

print(f"inceptionV1, time=", time.time()-start)

# Evaluation Loop
with torch.no_grad():
    inceptionV1.eval()

    correct_inceptionV1, total_inceptionV1 = 0, 0

    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        # Evaluate VGG19
        outputs_inceptionV1 = inceptionV1(images)
        _, predicted_inceptionV1 = torch.max(outputs_inceptionV1.data, 1)
        total_inceptionV1 += labels.size(0)
        correct_inceptionV1 += (predicted_inceptionV1 == labels).sum().item()


        # Calculate and print accuracy for each model
    accuracy_vgg19 = 100 * correct_inceptionV1 / total_inceptionV1

    print(f'Accuracy of Inception V1 on the test images: {accuracy_vgg19}%')


"""
Epoch 1/5, Loss: Inception V4 - 0.1060948297381401
Epoch 2/5, Loss: Inception V4 - 0.18519024550914764
Epoch 3/5, Loss: Inception V4 - 0.36149805784225464
Epoch 4/5, Loss: Inception V4 - 0.02416372485458851
Epoch 5/5, Loss: Inception V4 - 0.16895146667957306
vgg13, time= 84.13413619995117
Accuracy of Inception V4 on the test images: 98.0%
"""