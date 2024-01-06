import torch
import keras

from torchmodels import MLP

if __name__ == "__main__":
    # load mnist using keras
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
    x_test = x_test.reshape(10000, 784).astype('float32') / 255.0

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TRAIN_EPOCHS = 4
    BATCH_SIZE = 64
    NUM_CLASSES = 10

    # create a MLP model
    model = MLP(dimensions=[784, 1024, 512, 256, 128, NUM_CLASSES])
    model = model.to(DEVICE)

    # create a dataloader
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

    print(f"Test accuracy: {correct / total}")
