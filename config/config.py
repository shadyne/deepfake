import torch
import os
import platform

class Config:

    # Data paths dengan struktur train/val/test
    DATA_DIR = 'data'
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    VAL_DIR = os.path.join(DATA_DIR, 'val')
    TEST_DIR = os.path.join(DATA_DIR, 'test')
    
    # Output paths
    OUTPUT_DIR = 'output'
    FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
    PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, 'predictions')
    METRICS_DIR = os.path.join(OUTPUT_DIR, 'metrics')
    VISUALIZATIONS_DIR = os.path.join(OUTPUT_DIR, 'visualizations')
    
    # Model paths
    MODELS_DIR = 'saved_models'
    CHECKPOINTS_DIR = os.path.join(MODELS_DIR, 'checkpoints')
    
    # Logs
    LOGS_DIR = 'logs'
    TENSORBOARD_DIR = os.path.join(LOGS_DIR, 'tensorboard')
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    NUM_WORKERS = 0 if platform.system() == 'Windows' else 2
    
    # Training hyperparameters - IMPROVED
    BATCH_SIZE = 8
    NUM_EPOCHS = 12
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 1e-4
    USE_AMP = True
    
    # Model configuration
    BACKBONE = 'xception'
    NUM_CLASSES = 4
    PRETRAINED = True
    
    
    # Image preprocessing
    IMAGE_SIZE = 224
    NORMALIZE_MEAN = [0.5, 0.5, 0.5]
    NORMALIZE_STD = [0.5, 0.5, 0.5]
    
    # Data augmentation
    USE_AUGMENTATION = True
    HORIZONTAL_FLIP_PROB = 0.5
    ROTATION_DEGREES = 10
    
    RESIDUAL_SPATIAL_KERNEL_SIZE = 5
    RESIDUAL_SPATIAL_SIGMA = 1.0
    DCT_BLOCK_SIZE = 8
    DCT_TOP_K = 10
    
    NORMALIZE_RESIDUAL = True
    NORMALIZE_DCT = True
    
    # Evaluation settings
    SAVE_CONFUSION_MATRIX = True
    SAVE_ROC_CURVE = True
    SAVE_PRECISION_RECALL_CURVE = True
    SAVE_PREDICTIONS = False
    SAVE_FEATURE_VISUALIZATIONS = False
    
    # Visualization settings
    PLOT_DPI = 200
    PLOT_STYLE = 'seaborn-v0_8-darkgrid'
    COLOR_PALETTE = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    # Logging
    USE_TENSORBOARD = False
    LOG_INTERVAL = 10
    SAVE_MODEL_EVERY_EPOCH = False
    EARLY_STOPPING_PATIENCE = 5
    
    # Validation during training
    VALIDATE_EVERY_EPOCH = True
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories"""
        dirs = [
            cls.OUTPUT_DIR,
            cls.FIGURES_DIR,
            cls.PREDICTIONS_DIR,
            cls.METRICS_DIR,
            cls.VISUALIZATIONS_DIR,
            cls.MODELS_DIR,
            cls.CHECKPOINTS_DIR,
            cls.LOGS_DIR,
            cls.TENSORBOARD_DIR
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
    
    @classmethod
    def get_model_path(cls, method):
        """Get model save path"""
        return os.path.join(cls.MODELS_DIR, f'{method}_best_model.pth')
    
    @classmethod
    def get_checkpoint_path(cls, method, epoch):
        """Get checkpoint path"""
        return os.path.join(cls.CHECKPOINTS_DIR, f'{method}_epoch_{epoch}.pth')
    
    @classmethod
    def get_log_path(cls, method):
        """Get tensorboard log path"""
        return os.path.join(cls.TENSORBOARD_DIR, method)
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("\n" + "="*70)
        print("SYSTEM CONFIGURATION".center(70))
        print("="*70)
        print(f"{'Device:':<25} {cls.DEVICE}")
        print(f"{'Platform:':<25} {platform.system()}")
        print(f"{'NUM_WORKERS:':<25} {cls.NUM_WORKERS}")
        print(f"{'Backbone:':<25} {cls.BACKBONE}")
        print(f"{'Image Size:':<25} {cls.IMAGE_SIZE}x{cls.IMAGE_SIZE}")
        print(f"{'Batch Size:':<25} {cls.BATCH_SIZE}")
        print(f"{'Epochs:':<25} {cls.NUM_EPOCHS}")
        print(f"{'Learning Rate:':<25} {cls.LEARNING_RATE}")
        print(f"{'Weight Decay:':<25} {cls.WEIGHT_DECAY}")
        print(f"{'Data Augmentation:':<25} {'Enabled' if cls.USE_AUGMENTATION else 'Disabled'}")
        print("="*70 + "\n")


class BaselineConfig(Config):
    """Configuration for baseline RGB model"""
    METHOD_NAME = 'baseline'
    DESCRIPTION = 'Xception with RGB input only'


class ResidualSpatialConfig(Config):
    """Configuration for residual spatial model"""
    METHOD_NAME = 'residual_spatial'
    DESCRIPTION = 'Xception with Residual Noise (Spatial Domain)'


class ResidualDCTConfig(Config):
    """Configuration for residual DCT model"""
    METHOD_NAME = 'residual_dct'
    DESCRIPTION = 'Xception with DCT coefficients from Residual Noise'