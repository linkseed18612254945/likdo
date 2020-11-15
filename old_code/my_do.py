import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import ToTensor
import tqdm
import numpy as np

# Set environment
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 32
learning_rate = 0.001
epoch_size = 2
input_size = 28
num_classes = 10
input_channel = 1

# Build dataset
dataset_path = '/home/ubuntu/likun/image_data'
train_data = torchvision.datasets.MNIST(root=dataset_path, train=True, transform=ToTensor())
test_data = torchvision.datasets.MNIST(root=dataset_path, train=False, transform=ToTensor())
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

# Build model
class CNN(nn.Module):
    def __init__(self, input_channel, input_size, num_classes):
        super(CNN, self).__init__()
        self.c_layer1 = nn.Sequential(
            nn.Conv2d(input_channel, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.c_layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc = nn.Linear(32 * int(input_size / 4) ** 2, num_classes)

    def forward(self, x):
        output = self.c_layer1(x)
        output = self.c_layer2(output)
        output = output.reshape(x.shape[0], -1)
        output = self.fc(output)
        return output


model = CNN(input_channel, input_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

# Train
for epoch in range(epoch_size):
    model.train()
    losses = []
    for batch in tqdm.tqdm(list(train_dataloader), desc=f"Epoch {epoch + 1}, Training"):
        model.zero_grad()
        data = batch[0].to(device)
        label = batch[1].to(device)
        output = model.forward(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print(f"Epoch {epoch + 1},  Train loss {np.mean(losses)}")

# Test
losses = []
model.eval()
predict_labels = []
true_labels = []
for batch in tqdm.tqdm(list(test_dataloader), desc=f"Testing"):
    data = batch[0].to(device)
    label = batch[1].to(device)
    output = model.forward(data)
    predict_label = np.argmax(output.cpu().detach().numpy(), axis=1)
    predict_labels.extend(predict_label)
    true_labels.extend(label.cpu().detach().numpy())
    loss = criterion(output, label)
    losses.append(loss.item())
# # acc, p, r, f1 = utlis.evaluate(true_labels, predict_labels)
# # report = utlis.classification_report(true_labels, predict_labels)
# print(f"Test loss: {np.mean(losses)}, Acc: {acc}, P: {p}, R: {r}, f1: {f1}")
# print(report)