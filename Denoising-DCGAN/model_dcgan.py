import torch
import torch.nn as nn
import torch.nn.functional as F


# def init_weights(m):
#     if isinstance(m, nn.Linear):
#         torch.nn.init.xavier_uniform(m.weight)
#         m.bias.data.fill_(0.01)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, device, weight_std=1e-3):
        # in_channel
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Projection in series if needed prior to shortcut
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding='same')
        torch.nn.init.trunc_normal_(self.conv1x1.weight, 0, 1)
        self.projection_block = nn.Sequential(
            self.conv1x1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ).to(device)


        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, [7,3], stride=1, padding='valid'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ).to(device)

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, [7,3], stride=1, padding='valid'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ).to(device)

    def forward(self, x):
        if self.out_channels != x.shape[1]:
            residual = self.projection_block(x)
        else:
            residual = x

        x = self.pad(x)
        x = self.conv_block1(x)
        x = self.pad(x)
        x = self.conv_block2(x)
        x = x + residual
        return x

    def pad(self, x):
        x = F.pad(x, (0, 0, 3, 3), mode='circular')
        x = F.pad(x, (1, 1, 0, 0), mode='constant')
        return x



# Generator Code
class Generator(nn.Module):
    def __init__(self, device, n_res_blocks=5, input_shape=[1, 7, 126]):
        super(Generator, self).__init__()
        self.in_shape = input_shape  # [n_channels, height, width]
        self.n_units = 64
        self.res_units = [64, 64, 64, 64, 64]
        self.device = device


        self.first_blocks = nn.Sequential(
            nn.Conv2d(in_channels=self.in_shape[0], out_channels=self.n_units, kernel_size=[7,3], stride=1, padding='valid'),
            nn.ReLU()
        ).to(device)

        self.res_blocks = nn.Sequential(
            ResBlock(in_channels=self.n_units, out_channels=self.res_units[0], device=device),
            *[ResBlock(in_channels=self.res_units[0], out_channels=self.res_units[i], device=device) for i in
              range(1, n_res_blocks)]
        ).to(device)

        self.final_blocks = nn.Sequential(
            nn.Conv2d(in_channels=self.res_units[-1], out_channels=self.n_units, kernel_size=[7,3], stride=1,
                      padding='valid'),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.n_units, out_channels=self.n_units, kernel_size=1, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.n_units, out_channels=input_shape[0], kernel_size=1, stride=1, padding='same'),
            # nn.Sigmoid()
            nn.Tanh()
        ).to(device)

    def forward(self, x):
        x = self.pad(x)
        x = self.first_blocks(x)
        x = self.res_blocks(x)  # circular padding already in res_block
        x = self.pad(x)
        x = self.final_blocks(x)

        return x

    def pad(self,x):
        x = F.pad(x, (0, 0, 3, 3), mode='circular')
        x = F.pad(x, (1, 1, 0, 0), mode='constant')
        return x


    def save_gen(self, epoch, save_dir=None):
        if save_dir is None:
            save_dir = ''
        elif not save_dir.endswith('/'):
            save_dir += '/'

        torch.save({'epoch': epoch, 'model_state_dict': self.state_dict()},
                   save_dir + f'Gen_Epoch_{epoch}.pt')
        return

    def load_model(self, gen_path):

        gen_state_dict = torch.load(gen_path, map_location=self.device)
        self.load_state_dict(gen_state_dict.get('model_state_dict'))

        return



# Discriminator Code
class Discriminator(nn.Module):
    def __init__(self, device, n_res_blocks=5, input_shape=[1, 7, 128]):
        super(Discriminator, self).__init__()
        # self.ngpu = ngpu
        self.layers = [64, 128, 256, 512]
        self.weights_std = 2.0
        self.in_shape = input_shape  # [n_channels, height, width]

        self.first_conv = nn.Conv2d(in_channels=1, out_channels=self.layers[0], kernel_size=3, stride=1, padding='same')
        torch.nn.init.trunc_normal_(self.first_conv.weight, 0, self.weights_std)

        self.first_layers = nn.Sequential(
            self.first_conv,
            nn.LeakyReLU(0.2)
        ).to(device)

        self.mid_blocks = nn.Sequential(
            nn.Conv2d(in_channels=self.layers[0], out_channels=self.layers[0], kernel_size=3, stride=[1, 2],
                      padding=[1, 0]),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(self.layers[0]),
            *[self.create_mid_block(self.layers[i - 1], self.layers[i], ker_size=3, stride=[1, 2], padding=[1, 0],
                                    weights_std=self.weights_std) for i in range(1, len(self.layers))]
        ).to(device)

        self.final_conv_blocks = nn.Sequential(
            self.create_end_block(in_channels=self.layers[-1], out_channels=self.layers[-1],
                                  ker_size=3, stride=1, padding='same', weights_std=self.weights_std),

            self.create_end_block(in_channels=self.layers[-1], out_channels=self.layers[-1],
                                  ker_size=1, stride=1, padding='same', weights_std=self.weights_std),

            self.create_end_block(in_channels=self.layers[-1], out_channels=1,
                                  ker_size=1, stride=1, padding='same', weights_std=self.weights_std)

        ).to(device)

        self.final_layers = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(in_features= (self.in_shape[2] / (2**len(self.layers)) * self.in_shape[1]), out_features=1024),
            nn.Linear(in_features=7 * 7, out_features=1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        ).to(device)

    def create_mid_block(self, in_channels, out_channels, ker_size, stride, padding, weights_std):
        conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=ker_size,
                               stride=stride, padding=padding)
        torch.nn.init.trunc_normal_(conv_layer.weight, 0, weights_std)
        l_relu = nn.LeakyReLU()
        bn_layer = nn.BatchNorm2d(out_channels)
        mid_block = nn.Sequential(conv_layer, l_relu, bn_layer)
        return mid_block

    def create_end_block(self, in_channels, out_channels, ker_size, stride, padding, weights_std):
        conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=ker_size,
                               stride=stride, padding=padding)
        torch.nn.init.trunc_normal_(conv_layer.weight, 0, weights_std)
        bn_layer = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU()
        end_block = nn.Sequential(conv_layer, bn_layer, relu)
        return end_block

    def forward(self, x):
        # Input dims [bs,1,7,128]
        x = self.first_layers(x)  # [bs,64,7,128]
        x = self.mid_blocks(x)  # [bs,512,7,7]
        x = self.final_conv_blocks(x)  # [bs,1,7,7]
        x = self.final_layers(x)
        return x
