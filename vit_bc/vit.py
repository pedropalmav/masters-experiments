import math
import torch.nn as nn
from torchvision.models import VisionTransformer

class ViTBC(VisionTransformer):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        image_channels: int = 3,
        **kwargs 
    ):
        super().__init__(
            image_size,
            patch_size,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            **kwargs
        )
        self.conv_proj = nn.Conv2d(
            in_channels=image_channels, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
        )

        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)
        
if __name__ == '__main__':
    vit = ViTBC(
        image_size=8,
        patch_size=1,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        image_channels=7,
        num_classes=5
    )

    print(vit)