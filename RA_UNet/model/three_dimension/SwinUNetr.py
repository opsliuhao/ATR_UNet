import torch
from monai.networks.nets import SwinUNETR


def createSwinUNETR(img_size, in_channels, out_channels, feature_size=48):
    model = SwinUNETR(
        img_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size
    )
    return model


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = createSwinUNETR((128, 128, 128), in_channels=1, out_channels=3, feature_size=48).to(device)
    X = torch.rand((1, 1, 128, 128, 128), dtype=torch.float32)
    X = X.to(device)
    y = model(X)
    print(y.shape)