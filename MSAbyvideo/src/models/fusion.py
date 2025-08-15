import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_CLASSES

class TransformerFusionModel(nn.Module):
    """
    Transformer-based fusion model for multimodal sentiment analysis.
    Uses Transformer encoders for individual modalities and a cross-attention
    module to combine them into a fused sentiment representation.
    Args:
        text_dim (int): Dimensionality of text features.
        audio_dim (int): Dimensionality of audio features.
        visual_dim (int): Dimensionality of visual features.
        hidden_dim (int): Hidden layer size for transformer encoders.
        num_heads (int): Number of attention heads in transformer layers.
        num_layers (int): Number of transformer encoder layers.
        dropout_rate (float): Dropout rate for regularization.
    """
    def __init__(
        self, 
        text_dim, 
        audio_dim, 
        visual_dim, 
        hidden_dim=256, 
        num_heads=8, 
        num_layers=4, 
        dropout_rate=0.3,
        num_classes=NUM_CLASSES
    ):
        super(TransformerFusionModel, self).__init__()
        
        # Local imports to avoid circular dependencies
        from src.models.text import TransformerTextEncoder
        from src.models.audio import TransformerAudioEncoder
        from src.models.visual import TransformerVisualEncoder
        from src.models.attention import MultimodalCrossAttention
        
        # Individual modality encoders
        self.text_encoder = TransformerTextEncoder(
            text_dim, hidden_dim, num_layers, num_heads, dropout_rate
        )
        
        self.audio_encoder = TransformerAudioEncoder(
            audio_dim, hidden_dim // 2, num_layers // 2, num_heads // 2, dropout_rate
        )
        
        self.visual_encoder = TransformerVisualEncoder(
            visual_dim, hidden_dim // 2, num_layers // 2, num_heads // 2, dropout_rate
        )
        
        # Multimodal cross-attention fusion
        self.fusion_module = MultimodalCrossAttention(
            hidden_dim, hidden_dim // 2, hidden_dim // 2, hidden_dim, num_heads, dropout_rate
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, features):
        """
        Forward pass for transformer-based fusion model.

        Args:
            text_features (Tensor): Input tensor for text modality.
            audio_features (Tensor): Input tensor for audio modality.
            visual_features (Tensor): Input tensor for visual modality.

        Returns:
            Tensor: Fused sentiment prediction of shape [batch_size, 1].
        """
        text_features = features["text"]
        audio_features = features["audio"]
        visual_features = features["vision"]

        # Encode each modality using respective transformer encoders
        text_encoded = self.text_encoder.get_encoded_features(text_features)
        audio_encoded = self.audio_encoder.get_encoded_features(audio_features)
        visual_encoded = self.visual_encoder.get_encoded_features(visual_features)
        
        # Cross-attention fusion
        fused_features, sentiment = self.fusion_module(
            text_encoded, audio_encoded, visual_encoded
        )
        
        return sentiment