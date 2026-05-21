import torch.nn as nn
import timm


class XceptionBaseline(nn.Module):
    """
    Model Baseline: Xception + RGB
    Tahap 1: Training model dengan input RGB biasa
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(XceptionBaseline, self).__init__()
        
        # Load Xception pretrained dari timm
        self.backbone = timm.create_model('xception', pretrained=pretrained)
        
        # Xception punya 2048 features di layer terakhir
        num_features = self.backbone.get_classifier().in_features
        
        # Ganti classifier
        self.backbone.reset_classifier(num_classes=0)  # remove classifier
        
        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
        
        print(f"Model Xception Baseline dibuat (Pretrained: {pretrained})")
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Classify
        output = self.classifier(features)
        
        return output


class XceptionResidualSpatial(nn.Module):
    """
    Model Tahap 2: Xception + Residual Noise (Spatial Domain)
    Input: Residual noise hasil high-pass filter
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(XceptionResidualSpatial, self).__init__()
        
        # Backbone Xception
        self.backbone = timm.create_model('xception', pretrained=pretrained)
        num_features = self.backbone.get_classifier().in_features
        self.backbone.reset_classifier(num_classes=0)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
        
        print(f"Model Xception Residual Spatial dibuat (Pretrained: {pretrained})")
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output


class XceptionResidualDCT(nn.Module):
    """
    Model Tahap 3: Xception + DCT Residual
    Input: DCT coefficients dari residual noise
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(XceptionResidualDCT, self).__init__()
        
        # Backbone
        self.backbone = timm.create_model('xception', pretrained=pretrained)
        num_features = self.backbone.get_classifier().in_features
        self.backbone.reset_classifier(num_classes=0)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
        
        print(f"Model Xception Residual DCT dibuat (Pretrained: {pretrained})")
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output


def get_model(method, config):

    if method == 'baseline':
        model = XceptionBaseline(
            num_classes=config.NUM_CLASSES,
            pretrained=config.PRETRAINED
        )
    
    elif method == 'residual_spatial':
        model = XceptionResidualSpatial(
            num_classes=config.NUM_CLASSES,
            pretrained=config.PRETRAINED
        )
    
    elif method == 'residual_dct':
        model = XceptionResidualDCT(
            num_classes=config.NUM_CLASSES,
            pretrained=config.PRETRAINED
        )
    
    else:
        raise ValueError(f"Method '{method}' tidak dikenal. "
                        f"Pilih: baseline, residual_spatial, residual_dct")
    
    return model.to(config.DEVICE)