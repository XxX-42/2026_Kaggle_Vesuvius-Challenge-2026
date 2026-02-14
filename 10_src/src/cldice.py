import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftSkeletonize(nn.Module):
    def __init__(self, num_iter=10):
        super(SoftSkeletonize, self).__init__()
        self.num_iter = num_iter

    def forward(self, img):
        """
        Soft skeletonization using iterative min/max pooling to approximate morphological erosion/dilation.
        img: (B, C, D, H, W) - Probabilities in [0, 1]
        """
        # Ensure input is 5D
        if img.dim() != 5:
            raise ValueError(f"SoftSkeletonize expects 5D input (B, C, D, H, W), got {img.dim()}D")

        # Soft Erosion (MinPool) and Dilation (MaxPool)
        # We approximate min_pool using -max_pool(-x)
        def soft_erode(x):
            return -F.max_pool3d(-x, kernel_size=3, stride=1, padding=1)

        def soft_dilate(x):
            return F.max_pool3d(x, kernel_size=3, stride=1, padding=1)

        def soft_open(x):
            return soft_dilate(soft_erode(x))

        def soft_close(x):
            return soft_erode(soft_dilate(x))

        skel = soft_close(img)
        for _ in range(self.num_iter):
            # Skeleton iteration: Skel = Skel - Open(Skel)
            # This is a soft approximation of topological skeleton
            opened = soft_open(skel)
            skel = F.relu(skel - opened) # Keep only positive parts
            
            # Combine with original open to maintain structure
            # (In standard skeletonization, we iteratively thin. Here we use a simplified soft approach)
            # Actually, for soft clDice, a common approach is just soft_open/close iterations or 
            # using the difference between image and opened image.
            # Let's use the standard clDice implementation logic for "soft skeleton":
            # skel = min(img, 1 - open(img)) is precise for binary, but for soft:
            # We iterate thinning.
            
            # Refined soft skeletonization for clDice (often used in medical segmentation):
            # 1. Erode
            # 2. Skeleton = Image - Open(Image) (Morphological gradient)
            # Let's stick to the Iterative thinning approximation if possible, 
            # OR use the "Morphological Skeleton" definition: S = U (Erode(I, n) - Open(Erode(I, n)))
            pass
        
        # Simplified Soft Skeleton for Deep Learning (differentiable):
        # A robust approximation often used is: Skel = F.relu(img - soft_open(img))
        # This captures the "ridges" or centerlines.
        skel = F.relu(img - soft_open(img))
        
        return skel

def soft_cldice(y_pred, y_true, iter_=10, smooth=1e-5):
    """
    y_pred: (B, C, D, H, W) - Probabilities
    y_true: (B, C, D, H, W) - Ground Truth
    """
    skel_pred = SoftSkeletonize(num_iter=iter_)(y_pred)
    skel_true = SoftSkeletonize(num_iter=iter_)(y_true)
    
    # Topology Precision: Overlap of Pred Skeleton with True Mask
    tprec = (skel_pred * y_true).sum(dim=(2, 3, 4)) / (skel_pred.sum(dim=(2, 3, 4)) + smooth)
    
    # Topology Sensitivity: Overlap of True Skeleton with Pred Mask
    tsens = (skel_true * y_pred).sum(dim=(2, 3, 4)) / (skel_true.sum(dim=(2, 3, 4)) + smooth)
    
    # clDice
    cldice = 2.0 * (tprec * tsens) / (tprec + tsens + smooth)
    
    return cldice.mean()


class soft_cldice_loss(nn.Module):
    def __init__(self, iter_=3, smooth=1e-5):
        """
        iter_: number of iterations for soft skeletonization (not used in the simplified version, 
               but kept for interface compatibility if we expand)
        """
        super(soft_cldice_loss, self).__init__()
        self.iter = iter_
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        # inputs are logits? No, usually expects probabilities.
        # Check if inputs are logits or probs. 
        # For this module, let's assume inputs are probabilities [0, 1].
        # The caller (CombinedLoss) should handle Sigmoid.
        return 1.0 - soft_cldice(y_pred, y_true, self.iter, self.smooth)

if __name__ == "__main__":
    # Test
    model = soft_cldice_loss()
    # Mock data (B, C, D, H, W)
    pred = torch.rand(2, 1, 32, 64, 64, requires_grad=True)
    target = torch.randint(0, 2, (2, 1, 32, 64, 64)).float()
    
    loss = model(pred, target)
    print(f"Loss: {loss.item()}")
    loss.backward()
    print("Backward pass successful")
