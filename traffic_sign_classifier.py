import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import yaml

# To input input for prediction
from PIL import Image

class TrafficSignCNN(nn.Module):
    
    def __init__(self, device="cpu", labels_path="labels.yaml", input_size=64):
        super().__init__()
        self.device = device
        self.num_classes = 43
        self.labels_dict = self.load_labels(labels_path)
        
        self.transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),  # Resize images to input_size*input_size
                transforms.ToTensor(),        # Convert images to PyTorch tensors
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
                ])
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
       
        # pool layers
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2, padding=0)
        
        # dropout layers
        self.dropout = nn.Dropout(0.5)

        # Fully connected layer
        self.fullc1 = nn.Linear(in_features=1024, out_features= 512)
        self.fullc2 = nn.Linear(in_features=512, out_features= 128)
        self.fullc3 = nn.Linear(in_features=128, out_features= self.num_classes)

    # load the labels of yaml format, and prase to a dictionary
    def load_labels(self, labels_path):
        if os.path.exists(labels_path):
            with open(labels_path, 'r') as file:
                return yaml.safe_load(file)
        else:
            return None

    # Save the trained model weights and biases
    def save_model(self, name="classifier_weights"):
        if not os.path.exists("weights"):
            os.makedirs("weights")
            
        torch.save(self.state_dict(), f"weights/{name}.pt")

    # Load pre-trained weights and biases to this model
    def load_model(self, path):
        self.load_state_dict(torch.load(path, weights_only=False))

    def forward(self, x):
        # Convolution and pooling layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)

        x = x.view(-1, 1024) # flatten first fully connected layer input

        # Fully connected layers
        x = F.relu(self.fullc1(x))
        x = self.dropout(x)
        x = F.relu(self.fullc2(x))
        x = self.dropout(x)
        x = self.fullc3(x)

        return x
    
    # evaluate the models performance on the test set, do not update the weights on the new data
    def __test_batches(self, test_loader : DataLoader, criterion : nn.CrossEntropyLoss, optimizer : torch.optim.Adam):
        print("Testing model")
        tests_correct = 0
        tests_total = 0
        with torch.no_grad():
            for batch, (x_test, y_test) in enumerate(test_loader):
                batch += 1

                x_test, y_test = x_test.to(self.device), y_test.to(self.device)

                outputs = self.forward(x_test)
                _, predicted = torch.max(outputs.data, 1)

                tests_total += y_test.size(0)
                
                tests_correct += (predicted == y_test).sum().item()

                if batch % 100 == 0:
                    print(f"Test Batch {batch}")
        
        loss = criterion(outputs, y_test)
        print(f"Tests correct: {tests_correct} / {tests_total}, loss: {loss}")

    # Pass an RGB immage, and forward through th neural network
    # returns a prediction with the label or index (depending if labels.yaml exists)
    def predict(self, image):
        image = self.transform(image)
        image = image.unsqueeze(0)
        image = image.to(self.device)

        with torch.no_grad():
            output = self.forward(image)
            output = torch.softmax(output, dim=1)

        val, predicted = torch.max(output, 1)
        
        if val < 0.5:
            return "no match"
        else:
            if self.labels_dict is not None:

                return f"{int(predicted)}: {self.labels_dict[int(predicted)]}, confidence: {val}"
            else:
                return int(predicted)

    def __train_batches(self, train_loader : DataLoader, criterion: nn.CrossEntropyLoss, optimizer : torch.optim.Adam):
        print("Training model")
        
        for batch, (x_train, y_train) in enumerate(train_loader):

            x_train, y_train = x_train.to(self.device), y_train.to(self.device)

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

    # Load the GTSRB dataset and return loader
    def load_dataset(self, batch_size=10):
    
        # Download and transform the gtsrb dataset
        train_data = datasets.GTSRB(root="gtsrb/train", split="train",
                                    download=True, transform=self.transform)
        test_data = datasets.GTSRB(root="gtsrb/test", split="test",
                                    download=True, transform=self.transform)

        # Create batch sizes for the data, support increased number of workers (multiprocessing), and moving tasks to GPU
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

        return train_loader, test_loader


    # Train the model on train dataset
    # learning_rate is the gradient step, and decreasing it decreases chances for overstepping, 
    # but takes significantly longer to optimize (train)
    def train_model(self, epochs=20, learning_rate=0.001):

        train_loader, test_loader = self.load_dataset()

        # Loss function
        criterion = nn.CrossEntropyLoss()
        # Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=0.001)

        for i in range(1, epochs+1):
            print(f"Epoch {i}")
            self.__train_batches(train_loader, criterion, optimizer)
            self.__test_batches(test_loader, criterion, optimizer)
    

def main():
    # Transformer to change dataset into tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print(f"GPU available, running on: {torch.cuda.get_device_name(device)}")

    model = TrafficSignCNN(device, input_size=64).to(device)
    #model.train_model(epochs=20, learning_rate=0.001)
    #model.save_model()
    model.load_model("weights/classifier_weights.pt")

    test_image = Image.open("test.jpg").convert("RGB")
    prediction = model.predict(test_image)
    print(prediction)


if __name__ == "__main__":
    main()