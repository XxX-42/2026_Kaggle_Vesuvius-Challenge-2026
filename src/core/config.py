"""
Configuration Loader
Loads configs/config.yaml and provides a pythonic interface.
"""
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Any
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs/config.yaml"

@dataclass
class ConfigSchema:
    PROJECT_ROOT: Path = PROJECT_ROOT
    
    # Loaded from YAML
    DATA_ROOT: Path = field(init=False)
    Z_SLICES: List[int] = field(init=False)
    WINDOW_MIN: float = field(init=False)
    WINDOW_MAX: float = field(init=False)
    VALID_SPLIT: float = field(init=False)
    
    BATCH_SIZE: int = field(init=False)
    EPOCHS: int = field(init=False)
    LR: float = field(init=False)
    
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
    EXPERIMENT_NAME: str = "default"

    def __post_init__(self):
        # Load YAML
        if not CONFIG_PATH.exists():
            raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
            
        with open(CONFIG_PATH, 'r') as f:
            raw = yaml.safe_load(f)
            
        # Parse Data Section
        d = raw.get("data", {})
        # Handle relative path in YAML
        r_path = d.get("root", "data/native/train/1")
        self.DATA_ROOT = (self.PROJECT_ROOT / r_path).resolve()
        
        self.Z_SLICES = d.get("z_slices", [])
        if not self.Z_SLICES:
             raise ValueError("z_slices cannot be empty in config.yaml")
             
        self.WINDOW_MIN = float(d.get("window_min", 18000))
        self.WINDOW_MAX = float(d.get("window_max", 28000))
        self.VALID_SPLIT = float(d.get("valid_split_fraction", 0.2))
        
        # Parse Training Section
        t = raw.get("training", {})
        self.BATCH_SIZE = int(t.get("batch_size", 16))
        self.EPOCHS = int(t.get("epochs", 20))
        self.LR = float(t.get("lr", 1e-3))
        self.EXPERIMENT_NAME = t.get("experiment_name", "vesuvius")
        
        self.OUTPUT_DIR = self.PROJECT_ROOT / "outputs" / self.EXPERIMENT_NAME
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
    @property
    def IN_CHANNELS(self):
        return len(self.Z_SLICES)

# Singleton
cfg = ConfigSchema()
