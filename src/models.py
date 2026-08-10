import timm
import torch
import torch.nn as nn


class MetadataMLP(nn.Module):
    """Projects the sparse metadata vector into a dense embedding."""

    def __init__(self, in_dim: int, out_dim: int = 32, hidden: int = 64, p: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(p),
            nn.Linear(hidden, out_dim), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class MelanomaViT(nn.Module):
    """ViT-B/16 backbone with an optional metadata fusion branch.

    Late fusion by concatenation is used rather than early fusion: the
    pretrained backbone stays unmodified, so image representations remain
    directly comparable across all experiments.
    """

    def __init__(self, meta_dim: int = 0, pretrained: bool = True,
                 meta_emb: int = 32, p: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model("vit_base_patch16_224",
                                          pretrained=pretrained, num_classes=0)
        img_dim = self.backbone.num_features  # 768
        self.use_meta = meta_dim > 0
        self.meta_mlp = MetadataMLP(meta_dim, meta_emb) if self.use_meta else None
        fuse_dim = img_dim + (meta_emb if self.use_meta else 0)
        self.head = nn.Sequential(
            nn.Linear(fuse_dim, 256), nn.GELU(), nn.Dropout(p), nn.Linear(256, 1),
        )

    def forward(self, img, meta=None):
        f = self.backbone(img)
        if self.use_meta:
            f = torch.cat([f, self.meta_mlp(meta)], dim=1)
        return self.head(f).squeeze(1)  # raw logit