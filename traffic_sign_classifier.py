import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class TrafficSignCNN(nn.Module):
    
    def __init__(self, gpu=False):
        super().__init__()

        self.gpu = gpu

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
       
        # pool layers
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        
        # dropout layers
        self.dropout = nn.Dropout(0.5)

        # Fully connected layer
        self.fullc1 = nn.Linear(in_features=1600, out_features= 800)
        self.fullc2 = nn.Linear(in_features=800, out_features= 200)
        self.fullc3 = nn.Linear(in_features=200, out_features= 43)

    def forward(self, x):
        # Convolution and pooling layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = x.view(-1, 1600) # flatten first fully connected layer input

        # Fully connected layers
        x = F.relu(self.fullc1(x))
        x = self.dropout(x)
        x = F.relu(self.fullc2(x))
        x = self.dropout(x)
        x = F.log_softmax(self.fullc3(x), dim=1)

        return x
    
    # evaluate the models performance on the test set, do not update the weights on the new data
    def __test_batches(self, test_loader : DataLoader, criterion : nn.CrossEntropyLoss, optimizer : torch.optim.Adam):
        print("Testing model")
        
        tests_correct = 0
        tests_count = 0
        with torch.no_grad():
            for batch, (x_test, y_test) in enumerate(test_loader):
                batch += 1
                tests_count += 1

                y_val = self.forward(x_test)
                predicted = torch.max(y_val.data, 1)[1]
                tests_correct = (predicted == y_test).sum()
                if batch % 100 == 0:
                    print(f"Test Batch {batch}")
        
        loss = criterion(y_val, y_test)
        print(f"Tests correct: {tests_correct} / {tests_count}, loss: {loss}")


    def __train_batches(self, train_loader : DataLoader, criterion: nn.CrossEntropyLoss, optimizer : torch.optim.Adam):
        print("Training model")
        
        for batch, (x_train, y_train) in enumerate(train_loader):
            batch += 1
             
            # predict
            y_pred = self.forward(x_train)
            
            # measure loss (error)
            loss = criterion(y_pred, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                print(f"Train Batch {batch}, Loss {loss}")


    # Train the model on train dataset
    # learning_rate is the gradient step, and decreasing it decreases chances for overstepping, 
    # but takes significantly longer to optimize (train)
    def train_model(self, train_loader : DataLoader, test_loader : DataLoader, epochs=5, learning_rate=0.001):
        # Loss function
        criterion = nn.CrossEntropyLoss()
        # Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for i in range(epochs):
            print(f"Epoch {i}")
            self.__train_batches(train_loader, criterion, optimizer)
            self.__test_batches(test_loader, criterion, optimizer)
        

    
    

def main():
    # Transformer to change dataset into tensors
    
    transform = transforms.Compose([
                transforms.Resize((32, 32)),  # Resize images to 32x32
                transforms.ToTensor(),        # Convert images to PyTorch tensors
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize images
                ])
    
    # Download and transform the gtsrb dataset
    train_data = datasets.GTSRB(root="gtsrb/train", split="train",
                                 download=True, transform=transform)
    test_data = datasets.GTSRB(root="gtsrb/test", split="test",
                                download=True, transform=transform)

    # Create batch sizes for the data
    train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

    model = TrafficSignCNN()
    model.train_model(train_loader, test_loader)


if __name__ == "__main__":
    main()