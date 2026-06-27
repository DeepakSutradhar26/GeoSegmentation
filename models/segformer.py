import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

class SegFormerBinary(nn.Module):
    def __init__(self):
        super().__init__()

        # Load pretrained SegFormer-B2
        self.backbone = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b2",
            num_labels=2,
            ignore_mismatched_sizes=True,
        )

        # Replace final layer: multi-class -> binary (1 output)
        in_ch = self.backbone.decode_head.classifier.in_channels
        self.backbone.decode_head.classifier = nn.Conv2d(in_ch, 1, kernel_size=1)

    def forward(self, x):
        # x: (B, 3, H, W)
        out    = self.backbone(pixel_values=x)
        logits = out.logits  # (B, 1, H/4, W/4)

        # Upsample back to input size
        logits = F.interpolate(
            logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return logits  # (B, 1, H, W) Apply sigmoid for probability