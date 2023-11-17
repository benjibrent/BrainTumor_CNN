import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision.transforms as trans
from torchvision.transforms import Compose, ToTensor
from torchvision.datasets import ImageFolder

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

num_classes = 4
IMG_SIZE = (28, 28)

# hyperparams

BATCH_SIZE = 64
learning_rate = 0.01
num_epochs = 10  

# image transform
transform = Compose([
    trans.ToTensor(),
    trans.Resize(size = IMG_SIZE)
])


train_dataset = ImageFolder("./Training/", transform = transform)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)

val_dataset = ImageFolder("./Validation/", transform = transform)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True)

test_dataset = ImageFolder("./Testing/", transform = transform)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False)


class BrainTumorClassification(nn.Module):
    def __init__(self, input_size, out_size):
        super().__init__()
        self.con_stack_1 = nn.Conv2d(in_channels = input_size, out_channels = 6, kernel_size = 2, padding = 2)
        self.avg_pool_1 = nn.AvgPool2d(stride = 2, kernel_size = 2)
        self.con_stack_2 = nn.Conv2d(in_channels = 6, out_channels = 16,  kernel_size = 2)
        self.avg_pool_2 = nn.AvgPool2d(stride = 2, kernel_size = 2)
        
        self.linear_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = 784, out_features = 120),
            nn.Linear(in_features = 120, out_features = 84),
            nn.Linear(in_features = 84, out_features = out_size)
        )
        
    def forward(self, x):
        x = self.con_stack_1(x)
        x = self.avg_pool_1(x)
        x = self.con_stack_2(x)
        x = self.avg_pool_2(x)
        
        x = self.linear_stack(x)
        
        return x

# Define the baseline CNN model
class BaselineCNN(nn.Module):
    def __init__(self, input_size, out_size):
        super(BaselineCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=input_size, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(32 * 14 * 14, out_size)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x   
     
sample_x, sample_y = next(iter(train_loader))

loss_fn = nn.CrossEntropyLoss()
brain_tumor_cnn = BrainTumorClassification(input_size = sample_x.shape[1], out_size = num_classes)
#baseline_cnn = BaselineCNN(input_size = sample_x.shape[1], out_size = num_classes)

optimizer_brain_tumor = torch.optim.Adam(brain_tumor_cnn.parameters(), lr = learning_rate)
#optimizer_baseline = torch.optim.Adam(baseline_cnn.parameters(), lr = learning_rate)


def check_acc(y_probs, y_true):
    correct_samples = torch.eq(y_probs, y_true).sum().item()
    return correct_samples / len(y_probs)

# variables for plotting graphs
epoch_array, train_accuracies, train_losses, val_accuracies, val_losses = ([] for i in range(5))

# training the model
def train_loop(model, train_loader, val_loader, loss_fn, optimizer, epochs, device):
    model = model.to(device)
    model.train()

    for epoch in range(epochs):
        epoch_array.append(epoch+1)
        print(f"---- Epoch {epoch+1} -----")
        loss_train, acc_train = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            y_logits = model(x)
            
            y_probs = torch.softmax(y_logits, dim = 0).argmax(dim = 1)
            
            loss = loss_fn(y_logits, y)
            loss_train += loss
            acc_train += check_acc(y_probs, y)
            
            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        loss_train /= len(train_loader)
        acc_train /= len(train_loader)
        train_accuracies.append(acc_train)
        train_losses.append(loss_train)

        print(f"train_loss: {loss_train:.4f}, train_acc: {acc_train:.4f}")

        # validating model
        model.eval()
        loss_val, acc_val = 0.0, 0.0
        with torch.inference_mode():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_logits = model(x)
                
                y_probs = torch.softmax(y_logits, dim = 0).argmax(dim = 1)
                loss = loss_fn(y_logits, y)
                
                loss_val += loss
                acc_val += check_acc(y_probs, y)
                
            loss_val /= len(val_loader)
            acc_val /= len(val_loader)
            val_accuracies.append(acc_val)
            val_losses.append(loss_val)
        
        print(f"val_loss: {loss_val:.4f}, val_acc: {acc_val:.4f}\n")

    return model  
    
def test(model, data_loader, loss_fn, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    loss_test=0.0

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            predicted = torch.softmax(outputs, dim = 0).argmax(dim = 1)

            total_samples += y.size(0)
            total_correct += (predicted == y).sum().item()

            loss = loss_fn(outputs, y)   
            loss_test += loss
                
        loss_test /= len(data_loader)

    accuracy = total_correct / total_samples
    return accuracy, loss_test


# train and test baseline model

# baseline_cnn = train_loop(baseline_cnn, train_loader, val_loader, loss_fn, optimizer_baseline, epochs=num_epochs, device = "cpu")
# baseline_accuracy, baseline_loss = test(baseline_cnn, test_loader, loss_fn, device='cpu')
# print(f"Test Accuracy: {baseline_accuracy:.4f}")
# print(f"Test Loss: {baseline_loss:.4f}")


# plot graphs
def plot():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    with torch.no_grad():
        ax1.plot(epoch_array, train_accuracies, label='training')
        ax1.plot(epoch_array, val_accuracies, label='validation')
        #ax1.plot(epoch_array, [brain_tumor_accuracy] * len(epoch_array), '--', label='testing')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy at each epoch')
        ax1.legend()

        ax2.plot(epoch_array, train_losses, label='training')
        ax2.plot(epoch_array, val_losses, label='validation')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Loss at each epoch')
        ax2.legend()

        fig.suptitle(f"Learning rate: {learning_rate}")
        plt.tight_layout()
        plt.show()


# save model
#torch.save(brain_tumor_cnn.state_dict(), 'opt_model.pth')



if __name__ == '__main__':
    # train and test model and plot results

    # brain_tumor_cnn = train_loop(brain_tumor_cnn, train_loader, val_loader, loss_fn, optimizer_brain_tumor, epochs=num_epochs, device = "cpu")
    # brain_tumor_accuracy, brain_tumor_loss = test(brain_tumor_cnn, test_loader, loss_fn, device='cpu')
    # print(f"Test Accuracy: {brain_tumor_accuracy:.4f}")
    # print(f"Test Loss: {brain_tumor_loss:.4f}")
    # plot()

    # to load and make a prediction of an image
    model = BrainTumorClassification(input_size = sample_x.shape[1], out_size = num_classes)
    model.load_state_dict(torch.load('opt_model.pth'))
    model.eval()

    image = Image.open('image(2).jpg')
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)


    with torch.no_grad():
        output = model(image_tensor)


    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)
    print("Predicted class:", list(test_dataset.class_to_idx.keys())[predicted_class.item()])

    # test model on testing data

    brain_tumor_accuracy, brain_tumor_loss = test(model, test_loader, loss_fn, device='cpu')
    print(f"Test Accuracy: {brain_tumor_accuracy:.4f}")
    print(f"Test Loss: {brain_tumor_loss:.4f}")


