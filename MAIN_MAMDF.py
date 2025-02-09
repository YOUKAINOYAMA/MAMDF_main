import torch
from torch import nn
import torch.fft
from torch.nn import functional as F
from torchvision import datasets, models




def get_cnn_model():
    cnn_model = models.resnet50(pretrained=True)
    num_ftrs = cnn_model.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    cnn_model.fc = nn.Linear(num_ftrs, 100)
    # cnn_model(b_img.to(device)).shape
    return cnn_model


# Global Filter
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FNetBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
        return x


class FNet(nn.Module):
    def __init__(self, dim, depth, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, FNetBlock()),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class cross_attention_block(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.satt_cross1 = nn.MultiheadAttention(100, num_heads=2)
        self.satt_cross2 = nn.MultiheadAttention(100, num_heads=2)
        self.fc = nn.Linear(200, 100)
        self.flatten = nn.Flatten()

    def forward(self, x, y):
        x_y = self.satt_cross1(x, y, x)[0]
        y_x = self.satt_cross2(y, x, y)[0]
        x_y = self.flatten(x_y)
        y_x = self.flatten(y_x)
        cross_xy = torch.cat([x_y, y_x], dim=1)
        cross_xy = self.fc(cross_xy)
        return cross_xy


class bigModel(nn.Module):
    def __init__(self, cnn_model):
        super().__init__()
        self.nor = nn.BatchNorm1d(20)
        self.lin1 = nn.Linear(19, 100)
        self.lin2 = nn.Linear(700, 100)
        self.lin3 = nn.Linear(100, 50)
        self.out = nn.Linear(50, 4)
        self.flatten = nn.Flatten()
        # feature extraction
        self.dnn = DNNModel()
        # resnet-50
        self.cnn1 = cnn_model
        self.cnn2 = cnn_model
        # self attention
        self.satt_struct = nn.MultiheadAttention(100, num_heads=2)
        self.satt_mri = nn.MultiheadAttention(100, num_heads=2)
        self.satt_pet = nn.MultiheadAttention(100, num_heads=2)
        # cross modal attention
        self.satt_cross1 = nn.MultiheadAttention(100, num_heads=2)
        self.satt_cross2 = nn.MultiheadAttention(100, num_heads=2)
        self.satt_cross3 = nn.MultiheadAttention(100, num_heads=2)
        self.att = nn.TransformerEncoderLayer(100, 2)
        self.flayer = FNet(100, 2, 10)
        self.wl1 = weightLayerReturnOne()
        self.wl3 = weightLayerReturnThree()

    def forward(self, x, b_img, b_img_pet, flag):
        # Structured data processing
        global x_last
        x = self.dnn(x)
        # MRI data processing
        x_img = self.cnn1(b_img)
        # PET data processing
        x_img_pet = self.cnn2(b_img_pet)

        # cross-attention
        x_img1 = self.satt_cross2(x, x_img_pet, x_img_pet)[0]
        x_img2 = self.satt_cross2(x, x_img, x_img)[0]
        # concat
        x_1, x_img1, x_img2 = self.wl3(x, x_img1, x_img2)
        # self-attention FNet
        x_mid = torch.stack([x_1, x_img1, x_img2], dim=1)
        x_a = self.flatten(self.att(x_mid))
        x_f = self.flatten(self.flayer(x_mid))
        # concat
        x_last = torch.cat([x, x_a, x_f], dim=1)
        x_last = F.relu(self.lin2(x_last))
        x_last = F.relu(self.lin3(x_last))
        x_last = self.out(x_last)
        return x_last





def main():
    f_dim = b_x.shape[1]  # 21
    n_class = 4
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")
    # cnn_model = CNNModel().to(device)
    cnn_model = get_cnn_model().to(device)
    print(cnn_model(b_img.to(device)).shape)
    dnn_block = DNNModel().to(device)
    print(dnn_block(b_x.to(device)).shape)
    eg_tensor = torch.rand(2, 10, 100)
    print(FNet(100, 2, 10)(eg_tensor).shape)
    big_model = bigModel(cnn_model).to(device)
    print(big_model(b_x.to(device), b_img.to(device), b_img_pet.to(device), flag).shape)


if __name__ == '__main__':
    main()
