import torch
import torch.nn as nn
from data.settings import ModelConfig, SystemConfig
from data.data_processing import fen_to_tensor

class ChessNet(nn.Module):
    def __init__(self, 
                 input_channels=ModelConfig.INPUT_CHANNELS,
                 base_channels=ModelConfig.BASE_CHANNELS,
                 dropout_rate=ModelConfig.DROPOUT_RATE):
        super().__init__()
        
        # Улучшенный конволюционный блок с разными нелинейностями
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),  # Новая нелинейность
            nn.Dropout2d(dropout_rate),
            
            nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.Mish(),  # Mish activation
            nn.Dropout2d(dropout_rate),
            
            nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.SiLU(),  # Swish activation
            nn.Dropout2d(dropout_rate),
            
            nn.Conv2d(base_channels*4, base_channels*8, 3, padding=1),
            nn.BatchNorm2d(base_channels*8),
            nn.LeakyReLU(0.1),  # Leaky ReLU
            nn.Dropout2d(dropout_rate)
        )
        
        # Улучшенные головы с дополнительными нелинейностями
        self.from_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels*8 * 8 * 8, ModelConfig.HEAD_HIDDEN_SIZE),
            nn.LayerNorm(ModelConfig.HEAD_HIDDEN_SIZE),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(ModelConfig.HEAD_HIDDEN_SIZE, ModelConfig.HEAD_HIDDEN_SIZE//2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(ModelConfig.HEAD_HIDDEN_SIZE//2, 64)
        )
        
        self.to_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels*8 * 8 * 8, ModelConfig.HEAD_HIDDEN_SIZE),
            nn.LayerNorm(ModelConfig.HEAD_HIDDEN_SIZE),
            nn.Mish(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(ModelConfig.HEAD_HIDDEN_SIZE, ModelConfig.HEAD_HIDDEN_SIZE//2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            
            nn.Linear(ModelConfig.HEAD_HIDDEN_SIZE//2, 64)
        )

    def forward(self, x):
        x = x.to(device=SystemConfig.DEVICE)
        features = self.conv_block(x)
        from_pred = self.from_head(features)
        to_pred = self.to_head(features)
        return from_pred, to_pred

    def predict(self, board):
        """Интерфейс для предсказания"""
        with torch.no_grad():
            fen_tensor = fen_to_tensor(board.fen()).unsqueeze(0)
            from_logits, to_logits = self.forward(fen_tensor)
            return torch.softmax(from_logits, dim=1), torch.softmax(to_logits, dim=1)