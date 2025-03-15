# LeNet on FashionMNIST

## Overview
This project implements the LeNet model for image classification on the FashionMNIST dataset. The LeNet architecture, originally designed for digit classification, is adapted here to recognize different clothing items from FashionMNIST.

## Dataset
The FashionMNIST dataset consists of 70,000 grayscale images of size 28x28, categorized into 10 classes:
- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

The dataset is split into:
- Train size: 48000
- Validation size: 12000
- Test size: 10000

## Model Architecture
LeNet consists of the following layers:
1. **Convolutional Layer 1**: 6 filters, 5x5 kernel, followed by Sigmoid activation and Avg Pooling (2x2)
2. **Convolutional Layer 2**: 16 filters, 5x5 kernel, followed by Sigmoid activation and Avg Pooling (2x2)
3. **Fully Connected Layer 1**: 120 neurons with Sigmoid activation
4. **Fully Connected Layer 2**: 84 neurons with Sigmoid activation
5. **Output Layer**: 10 neurons (one for each class)

## Dependencies
Make sure to install the required dependencies before running the code:
```bash
pip install torch torchvision tensorboard
```

## Training
To train the model, open `lenet-train.ipynb` and execute the notebook cells. The training process includes:
1. Loading the FashionMNIST dataset
2. Splitting the dataset into training and validation sets
3. Training the LeNet model using cross-entropy loss and an optimizer
4. Saving checkpoints after each epoch

## Saving and Loading Checkpoints
To save a checkpoint after each epoch:
```python
def save_checkpoint(epoch, model, optimizer, loss):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, f'checkpoints/lenet_epoch_{epoch}.pth')
```
To load a checkpoint:
```python
def load_checkpoint(filepath, model, optimizer):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']
```

## Testing
To evaluate the model, open `lenet-test.ipynb` and execute the notebook cells. This will:
1. Load the trained model from the latest checkpoint
2. Run inference on the test dataset
3. Compute accuracy and loss

## Results
After training for several epochs, the model achieves a reasonable accuracy on the FashionMNIST test set. Future improvements may include using ReLU activations and batch normalization.

## Acknowledgments
This implementation is based on the classic LeNet-5 architecture introduced by Yann LeCun.

## License
This project is licensed under the MIT License.

