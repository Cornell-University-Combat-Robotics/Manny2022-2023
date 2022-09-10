from torchvision import models

# alexnet = models.alexnet(pretrained=True)
resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
