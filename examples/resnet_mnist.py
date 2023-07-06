import torch
import keras

from torchmodels import ResNet

if __name__ == "__main__":
    # load mnist using keras
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 1, 28, 28).astype('float32') / 255.0
    x_test = x_test.reshape(10000, 1, 28, 28).astype('float32') / 255.0

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TRAIN_EPOCHS = 2
    BATCH_SIZE = 64
    CHANNELS = 1
    HEIGHT = 28
    WIDTH = 28
    NUM_RESIDUAL_BLOCKS = 4
    NUM_CLASSES = 10

    # Create a ResNet model
    model = ResNet(
            in_channels=CHANNELS, 
            in_height=HEIGHT, 
            in_width=WIDTH, 
            num_blocks=NUM_RESIDUAL_BLOCKS, 
            num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    # Create a dataloader
    train_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(x_train).float(),
            torch.from_numpy(y_train).long())
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # training loop
    model.train()
    for epoch in range(TRAIN_EPOCHS):
        epoch_loss = 0.0
        for i, (x, y) in enumerate(train_loader, start=1):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            y_pred = model(x)

            optimizer.zero_grad()
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            print(f"Epoch: {epoch+1}/{TRAIN_EPOCHS}, Iter: {i}/{len(train_loader)}, Loss: {loss.item() / i}    \r", end="")
        print(f"Epoch: {epoch+1}/{TRAIN_EPOCHS}, Iter: {i}/{len(train_loader)}, Loss: {loss.item() / i}    ")

    # test
    test_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(x_test).float(),
            torch.from_numpy(y_test).long())
    test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            y_pred = model(x)
            y_pred = torch.softmax(y_pred, dim=1)
            _, predicted = torch.max(y_pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    print(f"Test accuracy: {round(correct / total, 5)}")
