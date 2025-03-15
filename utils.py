import torch

def save_checkpoint(epoch, model, optimizer, loss):
    checkpoint_path = "checkpoints/lenet_epoch_{epoch}.pth"
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, checkpoint_path.format(epoch=epoch))
    print(f"Checkpoint saved at {checkpoint_path.format(epoch=epoch)}")

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """ Load checkpoint vào model và optimizer (nếu có) """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Checkpoint loaded from {checkpoint_path} (Epoch {checkpoint['epoch']})")
    return checkpoint['epoch'], checkpoint['loss']
