import torch
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


class XceptionFusion(nn.Module):
    """
    Model Tahap 4: Feature Fusion
    
    Menggabungkan fitur dari RGB, Residual Spatial, dan DCT
    """
    
    def __init__(self, num_classes=2, pretrained=True, fusion_type='concat'):
        super(XceptionFusion, self).__init__()
        
        self.fusion_type = fusion_type
        
        # Backbone untuk RGB
        self.backbone_rgb = timm.create_model('xception', pretrained=pretrained)
        num_features = self.backbone_rgb.get_classifier().in_features
        self.backbone_rgb.reset_classifier(num_classes=0)
        
        # Backbone untuk Residual Spatial
        self.backbone_residual = timm.create_model('xception', pretrained=pretrained)
        self.backbone_residual.reset_classifier(num_classes=0)
        
        # Backbone untuk DCT
        self.backbone_dct = timm.create_model('xception', pretrained=pretrained)
        self.backbone_dct.reset_classifier(num_classes=0)
        
        # Fusion layer
        if fusion_type == 'concat':
            # Concat: gabungkan semua features
            fusion_features = num_features * 3  # 2048 * 3 = 6144
            
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(fusion_features, 1024),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, num_classes)
            )
        
        elif fusion_type == 'add':
            # Add: jumlahkan features (harus dimensi sama)
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, num_classes)
            )
        
        print(f"Model Xception Fusion dibuat (Type: {fusion_type}, Pretrained: {pretrained})")
    
    def forward(self, rgb, residual, dct):
        # Extract features dari masing-masing stream
        feat_rgb = self.backbone_rgb(rgb)
        feat_residual = self.backbone_residual(residual)
        feat_dct = self.backbone_dct(dct)
        
        # Fusion
        if self.fusion_type == 'concat':
            # Concatenate
            fused = torch.cat([feat_rgb, feat_residual, feat_dct], dim=1)
        
        elif self.fusion_type == 'add':
            # Element-wise addition
            fused = feat_rgb + feat_residual + feat_dct
        
        # Classify
        output = self.classifier(fused)
        
        return output


def get_model(method, config):
    """
    Factory function untuk buat model sesuai metode
    
    Args:
        method: 'baseline', 'residual_spatial', 'residual_dct', atau 'fusion'
        config: Config object
    
    Returns:
        model: PyTorch model
    """
    
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
    
    elif method == 'fusion':
        model = XceptionFusion(
            num_classes=config.NUM_CLASSES,
            pretrained=config.PRETRAINED,
            fusion_type=config.FUSION_TYPE
        )
    
    else:
        raise ValueError(f"Method '{method}' tidak dikenal. "
                        f"Pilih: baseline, residual_spatial, residual_dct, fusion")
    
    return model.to(config.DEVICE)