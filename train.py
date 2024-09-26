import torch
from torch.utils.data import DataLoader
from model import SimpleNN
from custom_dataset import RemoteDataset
from kusa import DatasetClient, DatasetSDKException

import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
PUBLIC_ID = os.getenv('PUBLIC_ID')
SECRET_KEY =  os.getenv('SECRET_KEY')
BATCH_SIZE = 100
NUM_EPOCHS = 5
LEARNING_RATE = 0.01
HIDDEN_SIZE = 128
NUM_CLASSES = 10  # Adjust based on your dataset

def main():
    # Initialize the SDK client
    client = DatasetClient(public_id=PUBLIC_ID, secret_key=SECRET_KEY)
    try:
        init_data = client.initialize()
        print(f"Total Rows: {init_data['totalRows']}")
        print("First 10 Rows:")
        print(init_data['first10Rows'])
    except DatasetSDKException as e:
        print(f"Initialization error: {e}")
        return

    # Create the dataset and dataloader
    dataset = RemoteDataset(client=client, batch_size=BATCH_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # Determine input size from first batch
    for inputs, labels in dataloader:
        input_size = inputs.shape[1]
        break

    # Initialize the model, loss function, and optimizer
    model = SimpleNN(input_size=input_size, hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            # Zero the gradients
            print(f"Labels: {labels}") 
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                avg_loss = running_loss / 100
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {avg_loss:.4f}")
                running_loss = 0.0

    print("Training complete.")

    # Evaluation
    evaluate(model, dataset, dataloader)

def evaluate(model, dataset, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Accuracy on the dataset: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
