import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import torch.nn.init as init

# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        #self.bn1 = nn.BatchNorm1d(512)  # BatchNorm for the first layer
        self.fc2 = nn.Linear(512, 256)
        #self.bn2 = nn.BatchNorm1d(256)  # BatchNorm for the second layer
        self.fc3 = nn.Linear(256, 10) # 10 classes for MNIST digits
        #self.dropout = nn.Dropout(0.2)  # 20% probability of an element to be dropped

        # # Weight initialization
        # init.normal_(self.fc1.weight, mean=0.0, std=0.01)  # Normal distribution
        # init.normal_(self.fc2.weight, mean=0.0, std=0.01)  # Normal distribution
        # init.xavier_uniform_(self.fc3.weight)  # Xavier/Glorot uniform distribution
        #
        # # Bias initialization to zero
        # init.zeros_(self.fc1.bias)
        # init.zeros_(self.fc2.bias)
        # init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = x.view(-1, 28*28) # Flatten the image
        #x = self.bn1(self.fc1(x))
        x = torch.relu(self.fc1(x)) # Using ReLU as default activation function
        #x = self.bn2(self.fc2(x))
        #x = self.dropout1(x)  # Apply dropout after ReLU activation
        x = torch.relu(self.fc2(x))
        #x = self.dropout2(x)  # Apply dropout after ReLU activation
        #x = self.fc1(x)  # Using linear as default activation function
        #x = self.fc2(x)
        #x = torch.sigmoid(self.fc1(x))  # Using ReLU as default activation function
        #x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x) # No activation here, raw scores
        return F.log_softmax(x, dim=1)

# Prepare the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Instantiate the model
model = Net()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# optimizer = torch.optim.Adadelta(model.parameters(), lr=0.01, rho=0.9, eps=1e-06)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# optimizer = torch.optim.NAdam(model.parameters(), lr=0.01)

# Define the training loop
def train(epoch, model, train_loader, optimizer, criterion):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # Zero the gradients
        output = model(data)   # Get the output for the batch
        loss = criterion(output, target)  # Calculate the loss
        loss.backward()        # Backpropagation
        optimizer.step()       # Update the weights
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))





# Define the testing loop
def test(model, test_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad():  # Turn off gradients for validation, saves memory and computations
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    return test_loss, accuracy



# Training
total_start = time.time()
num_epochs = 5
for epoch in range(1, num_epochs + 1):
    start_time = time.time()
    train(epoch, model, train_loader, optimizer, criterion)
    end_time = time.time()
    print(f"Epoch {epoch} took {end_time - start_time:.2f}s")
total_end = time.time()

print("total time", total_end - total_start)

# After training, call the test function with your data:
test_loss, test_accuracy = test(model, test_loader, criterion)
print("acc=", test_accuracy)


