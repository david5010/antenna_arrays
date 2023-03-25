from models.FF_1H import *
from datasets.AntDataset import AntDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from models.transformers import TransformerRegressor

def train(model, optimizer, criterion, train_loader):
    model.train()
    total_loss = 0.0
    for i, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.unsqueeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss/len(train_loader)

def evaluate(model, criterion, test_loader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            total_loss += loss.item()

    return total_loss/len(test_loader)

def train_and_evaluate(model, optimizer, criterion, train_loader, test_loader, num_epochs):
    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        train_loss = train(model, optimizer, criterion, train_loader)
        test_loss = evaluate(model, criterion, test_loader)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    return train_losses, test_losses

    

if __name__ == '__main__':

    train_data_path = 'data/csv/paired/YZ_pair_Train.csv'
    test_data_path = 'data/csv/paired/YZ_pair_Test.csv'

    train_loader = DataLoader(AntDataset(train_data_path), batch_size = 32, shuffle=True)
    test_loader = DataLoader(AntDataset(test_data_path), batch_size = 32, shuffle = False)

    torch.manual_seed(0)
    # simple_FF = SimpleNN(2048, 1, 8, nn.ReLU)
    # learning_rate = 0.001
    # num_epochs = 100
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(simple_FF.parameters(), lr = learning_rate)

    # ReLU_train_losses, ReLU_test_losses = train_and_evaluate(simple_FF, optimizer, criterion, train_loader, test_loader,num_epochs)
    # torch.save(simple_FF.state_dict(), 'simple_ff_weights.pth')

    # # Plot A and B on the same figure
    # plt.plot(ReLU_train_losses, label = 'Train Loss')
    # plt.plot(ReLU_test_losses, label='Test Loss')

    # # Add axis labels and legend
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss (MSE)')
    # plt.legend()

    # # Show the plot
    # plt.show()

    input_dim = 2048
    nhead = 8
    nhid = 256
    nlayers = 3
    dropout = 0.1

    transformer_reg = TransformerRegressor(input_dim, nhead, nhid, nlayers, dropout)

    learning_rate = 0.001
    num_epochs = 100
    criterion = nn.MSELoss()
    optimizer = optim.Adam(transformer_reg.parameters(), lr = learning_rate)

    transformer_train_loss, transformer_test_loss = train_and_evaluate(transformer_reg, optimizer, criterion, train_loader, test_loader, num_epochs)
    torch.save(transformer_reg.state_dict(),'base_transformer_reg.pth')

    # Plot A and B on the same figure
    plt.plot(transformer_train_loss, label = 'Train Loss')
    plt.plot(transformer_test_loss, label='Test Loss')

    # Add axis labels and legend
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()

    # Show the plot
    plt.show()

    # linear_regression = LinearRegression(2048)

    # linear_train_losses, linear_test_losses = train_and_evaluate(linear_regression, optimizer, criterion, train_loader, test_loader,num_epochs)
    # plt.plot(linear_train_losses, label = 'Train Loss')
    # plt.plot(linear_test_losses, label='Test Loss')

    # # Add axis labels and legend
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss (MSE)')
    # plt.legend()

    # # Show the plot
    # plt.show()

    # Show the weights
    # weights = linear_regression.state_dict()['linear.weight'].detach().numpy().squeeze()
    # weights = weights[:1024] + weights[1024:] # sum of weights for Y and Z
    # bias= linear_regression.state_dict()['linear.bias'].detach().numpy()
    # plt.bar(range(len(weights)), weights)
    # plt.xlabel('Feature')
    # plt.ylabel('Weight')
    # plt.show()