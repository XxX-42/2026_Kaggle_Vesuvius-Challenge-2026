class MetricMonitor:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def fbeta_score(preds, targets, threshold=0.5, beta=0.5):
    preds = (preds > threshold).float()
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    
    score = ((1 + beta**2) * tp) / ((1 + beta**2) * tp + (beta**2) * fn + fp + 1e-7)
    return score
