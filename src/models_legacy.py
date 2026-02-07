import segmentation_models_pytorch as smp
import torch
from pathlib import Path

def get_unet(encoder='resnet18', in_channels=16, classes=1, pretrained=False):
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights='imagenet' if pretrained else None,
        in_channels=in_channels,
        classes=classes,
        activation=None
    )
    return model

def load_checkpoint(model, path, device, strict=False):
    path = Path(path)
    if not path.exists():
        print(f"⚠️ Checkpoint not found: {path}")
        return model
        
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    else:
        model.load_state_dict(checkpoint, strict=strict)
    print(f"✅ Loaded checkpoint from {path}")
    return model
