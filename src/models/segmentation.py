import torch
import segmentation_models_pytorch as smp
from pathlib import Path

def get_model(encoder_name, in_channels, classes=1, activation=None):
    """Factory function to create the segmentation model."""
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet" if in_channels == 3 else None,
        in_channels=in_channels,
        classes=classes,
        activation=activation
    )

def load_model(model_path, encoder, in_channels, device):
    """Robust model loading function handling various checkpoint formats."""
    print(f"\n📥 Loading model: {model_path}")
    
    model = get_model(encoder, in_channels)
    
    path_obj = Path(model_path)
    if path_obj.exists():
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
            
        try:
            model.load_state_dict(state_dict, strict=False)
            print("   ✅ Loaded pretrained weights")
        except Exception as e:
            print(f"   ⚠️ Error loading state dict: {e}")
            print("   Trying strict=False...")
            model.load_state_dict(state_dict, strict=False)
    else:
        print("   ⚠️ No pretrained weights found, optimizing from scratch")
    
    return model.to(device)
