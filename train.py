import torch 
from torch import optim
import torch.nn as nn
from model import SimpleCNN
from dataprocessing import train_loader, test_loader
import matplotlib.pyplot as plt


def train(n_epochs):
    model = SimpleCNN()
    optimizer = optim.AdamW(model.parameters(), lr = 0.001, weight_decay = 1e-4)
    criterion = nn.CrossEntropyLoss()

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)

    loss_data = []
    for e in range(1,n_epochs+1):
        running_loss = 0
        total_samples = 0
        correct = 0
        for pixels, targets in train_loader:
            pixels, targets = pixels.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(pixels)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, prediction_indices = torch.max(output, dim=1)
            correct += (prediction_indices == targets).sum().item()
            total_samples += targets.shape[0]
        loss_data.append(loss.item())
        accuracy = 100 * correct / total_samples
        print(f'Epoch: [{e}/{n_epochs}] ; Accuracy: {accuracy} ; Loss: {running_loss / len(train_loader)}')

    plt.plot(range(1,n_epochs+1),loss_data)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    print(f"Training Loss: {running_loss/len(train_loader)}")

    model.eval()
    with torch.no_grad():
        validation_loss = 0
        total_samples = 0
        correct = 0
        for pixels, targets in test_loader:
            pixels, targets = pixels.to(device), targets.to(device)
            output = model(pixels)
            loss = criterion(output, targets)
            validation_loss+=loss
            total_samples += targets.shape[0]
            _, prediction_indices = torch.max(output, dim=1)
            correct += (prediction_indices == targets).sum().item()

        validation_loss = validation_loss/len(test_loader)
        accuracy = 100 * correct / total_samples
    
    print(f'validation_loss: {validation_loss} ; Accuracy: {accuracy}')
    torch.save(model.state_dict(), "model.pt")

#%%
if __name__ == "__main__":
    train(100)







