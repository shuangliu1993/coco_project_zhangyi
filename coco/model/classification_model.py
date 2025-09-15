import torch
from torch import nn

from transformers import SiglipVisionModel


class COCOClassificationModel(nn.Module):
    """
    input: image tensor  # N * 3 * H * W
    output: probability/logits of output classes  # N * num_classes
    """
    def __init__(self, num_classes):
        super(COCOClassificationModel, self).__init__()
        self.num_classes = num_classes
        self.vision_encoder = SiglipVisionModel.from_pretrained("google/siglip2-base-patch16-224")
        self.head = nn.Linear(768, num_classes)

    def forward(self, x):
        # 1. encode image
        image_embedding = self.vision_encoder(x).pooler_output  # N * 768

        # 2. get logits for each class
        logits = self.head(image_embedding)  # N * num_classes

        return logits


if __name__ == "__main__":
    model = COCOClassificationModel(num_classes=90)
    x = torch.rand([1, 3, 224, 224])
    logits = model(x)
    print(f"logits shape: {logits.shape}")
