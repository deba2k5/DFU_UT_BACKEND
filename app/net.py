"""UTNet: timm backbone + Texas grade head + stage (infection, ischemia) head.
Must stay identical to the class in the Kaggle training notebook."""
import timm
import torch
import torch.nn as nn


class UTNet(nn.Module):
    def __init__(self, arch, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0,
                                          drop_path_rate=0.1)
        was = self.backbone.training; self.backbone.eval()
        with torch.no_grad():                       # feature size works for any timm backbone
            n = self.backbone(torch.zeros(1, 3, 224, 224)).shape[-1]
        self.backbone.train(was)
        self.drop = nn.Dropout(0.3)
        self.grade_head = nn.Linear(n, 4)           # Texas grade 0-3
        self.stage_head = nn.Linear(n, 2)           # logits: [infection, ischemia]

    def forward(self, x):
        f = self.drop(self.backbone(x))
        return self.grade_head(f), self.stage_head(f)
