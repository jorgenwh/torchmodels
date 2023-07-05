import torch

from torchmodels import ResNet

if __name__ == "__main__":
    BATCH_SIZE = 2
    CHANNELS = 3
    HEIGHT = 224
    WIDTH = 224
    NUM_RESIDUAL_BLOCKS = 2
    NUM_CLASSES = 1000

    # Create a ResNet model
    model = ResNet(
            in_channels=CHANNELS, 
            in_height=HEIGHT, 
            in_width=WIDTH, 
            num_blocks=NUM_RESIDUAL_BLOCKS, 
            num_classes=NUM_CLASSES)

    x = torch.randn(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
    y = model(x) 

    print(y.shape) # (BATCH_SIZE, NUM_CLASSES)
