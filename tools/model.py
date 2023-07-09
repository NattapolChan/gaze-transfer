import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle

# VGG net for feature-based loss
VGG = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
for param in VGG.parameters():
    param.requires_grad = False
VGG_3 = nn.Sequential(*list(VGG.children())[0][:5])
VGG_8 = nn.Sequential(*list(VGG.children())[0][:13])
VGG_13 = nn.Sequential(*list(VGG.children())[0][:22])
print(VGG_3)
print(VGG_8)
print(VGG_13)
print("VGG pretrained imagenet loaded successfully -- == ")


def init_weights(m):
    if isinstance(m, nn.Linear):
        print("Found Linear")
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

    if isinstance(m, nn.Conv2d):
        print("Found Conv")
        nn.init.xavier_uniform_(m.weight)
        if m.bias is None:
            return
        m.bias.data.fill_(0.00)

class ConvBlock(nn.Module):
    def __init__(self, input_channel, last = False):
        super(ConvBlock, self).__init__()
        self.Conv_1 = nn.Conv2d(in_channels=input_channel, 
                                out_channels=2*input_channel, 
                                kernel_size=(3,3), 
                                padding='same')
        self.Conv_2 = nn.Conv2d(in_channels=2*input_channel, 
                                out_channels=2*input_channel, 
                                kernel_size=(3,3), 
                                padding='same')
        self.Conv_1l = nn.Conv2d(in_channels=input_channel, 
                                 out_channels=input_channel, 
                                 kernel_size=(3,3), 
                                 padding='same')
        self.Conv_2l = nn.Conv2d(in_channels=input_channel, 
                                 out_channels=input_channel, 
                                 kernel_size=(3,3), 
                                 padding='same')
        self.Conv_skip_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, 
                      out_channels=2*input_channel, 
                      kernel_size=(1,1)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(2*input_channel)
        )
        self.Conv_skip_1l = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, 
                      out_channels=input_channel, 
                      kernel_size=(1,1)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(input_channel)
        )
        self.features = nn.Sequential(
            self.Conv_1,
            nn.LeakyReLU(),
            nn.BatchNorm2d(2*input_channel),
            self.Conv_2,
            nn.BatchNorm2d(2*input_channel)
        )
        if last:
            self.features = nn.Sequential(
                self.Conv_1l,
                nn.LeakyReLU(),
                nn.BatchNorm2d(input_channel),
                self.Conv_2l,
                nn.BatchNorm2d(input_channel)
            )
        self.last = last
        self.input_channel = input_channel

    def forward(self, input):
        if not self.last:
            input_skip = self.Conv_skip_1(input)
        else:
            input_skip = self.Conv_skip_1l(input)
        input = self.features(input)
        input = input_skip + input
        return nn.ReLU()(input)

class DeConvBlock(nn.Module):
    def __init__(self, input_channel, last = False):
        super(DeConvBlock, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, 
                      out_channels=input_channel//2, 
                      kernel_size=(3,3), 
                      padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(input_channel//2),
            nn.Conv2d(in_channels=input_channel//2, 
                      out_channels=input_channel//2, 
                      kernel_size=(3,3), 
                      padding='same'),
            nn.BatchNorm2d(input_channel//2)
        )
        if last: 
            self.features = nn.Sequential(
                nn.Conv2d(in_channels=input_channel, 
                          out_channels=input_channel//4, 
                          kernel_size=(3,3), 
                          padding='same'),
                nn.LeakyReLU(),
                nn.BatchNorm2d(input_channel//4),
                nn.Conv2d(in_channels=input_channel//4, 
                          out_channels=input_channel//4, 
                          kernel_size=(3,3), 
                          padding='same'),
                nn.BatchNorm2d(input_channel//4)
            )
        self.last = last
        self.input_channel = input_channel
        self.conv = nn.Conv2d(
            in_channels=self.input_channel, 
            out_channels=self.input_channel//2, 
            kernel_size=(1,1)
        )
        self.conv_last = nn.Conv2d(
            in_channels=self.input_channel, 
            out_channels=self.input_channel//4, 
            kernel_size=(1,1)
        )
        self.bn2 = nn.BatchNorm2d(self.input_channel//2)
        self.bn4 = nn.BatchNorm2d(self.input_channel//4)

    def forward(self, input):
        if not self.last:
            input_skip = self.conv(input)
            input_skip = nn.LeakyReLU()(input_skip)
            input_skip = self.bn2(input_skip)
        else:
            input_skip =  self.conv_last(input)
            input_skip = nn.LeakyReLU()(input_skip)
            input_skip = self.bn4(input_skip)
        input = self.features(input)
        input = input_skip + input
        return nn.ReLU()(input)

class GlobalAlignmentNetwork(nn.Module):
    # global alignment network, input: 2 images, output: 1 image
    def __init__(self, channels=16) -> None:
        super(GlobalAlignmentNetwork, self).__init__()
        self.encoder_shared = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=(3,3), padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(channels),
            ConvBlock(input_channel=channels),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            ConvBlock(input_channel=2*channels),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
        )
        self.encoder_main = nn.Sequential(
            nn.Conv2d(in_channels=channels*8, out_channels=channels*8, kernel_size=(3,3), padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(8*channels),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            ConvBlock(input_channel=channels*8),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            ConvBlock(input_channel=channels*16, last=True),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), # avgpool 1/3, 1/4 for Eyediap
            nn.Flatten(),
            nn.Linear(in_features=16*channels, out_features=3),
            nn.Tanh(),
        )
        self.channels = channels
    
    def affine_transformation(self, img, theta, height=36, width=60):
        batch = theta.size()[0]
        theta = torch.flatten(theta, start_dim=1)
        coef = theta[:,0].to(img.device)
        iden = torch.eye(2).repeat(batch, 1).view(batch, 2, 2).to(img.device)
        scaling_factor = iden * ( 0.4 * (coef.view(-1,1,1) + 1.0)/2 + 0.6) 
        translate = theta[:,1:] * 0.3
        translate = translate.view(-1,2,1)
        theta = torch.cat([scaling_factor, translate], dim = 2)
        grid = F.affine_grid(
            theta, torch.Size((batch, 1, height, width)), align_corners=True
        )
        grid = grid.type(torch.float32)
        img = img.type(torch.float32)
        roi = F.grid_sample(img, grid, align_corners=True)
        return roi

    def forward(self, input_i, input_o):
        input_i_e = self.encoder_shared(input_i)
        input_o_e = self.encoder_shared(input_o)

        input = torch.cat((input_i_e, input_o_e), dim=1)
        input = self.encoder_main(input)
        input_i_t = self.affine_transformation(input_i, input)
        return input_i_t

class GazeRedirectionNetwork(nn.Module):
    # gaze redirection network, input: image, 2 2-dim vectors, output: 1 image
    def __init__(self, channels=64, dim=(36,60)):
        super(GazeRedirectionNetwork, self).__init__()
        self.convhead = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=(3,3), padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(channels),
        )
        self.encoder = nn.Sequential(
            ConvBlock(input_channel=channels),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            ConvBlock(input_channel=2*channels),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            ConvBlock(input_channel=4*channels, last=True),
            # nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), #for Eyediap
        )
        self.decoder = nn.Sequential(
            DeConvBlock(input_channel=4*channels),
            nn.UpsamplingBilinear2d(scale_factor=2),
            DeConvBlock(input_channel=2*channels),
            nn.UpsamplingBilinear2d(scale_factor=2),
            DeConvBlock(input_channel=channels, last=True),
            nn.Conv2d(in_channels=channels//4, out_channels=channels//32, kernel_size=(1,1)),
            nn.Tanh(),
        )
        self.fc1 = nn.Linear(1, 9*15)
        self.fc2 = nn.Linear(1, 9*15)
        self.conv_out = nn.Conv2d(
            in_channels=4*channels+2, out_channels=4*channels, kernel_size=(3,3), padding='same'
        )
        self.bn = nn.BatchNorm2d(4*channels)

        self.channels = channels
        self.dim = dim

    def forward(self, input_image, input_yaw, input_pitch):
        input_image = self.convhead(input_image)
        input_image = self.encoder(input_image)
        batch_size = input_image.size(0)

        input_yaw = input_yaw.view(batch_size, 1)
        input_pitch = input_pitch.view(batch_size, 1)

        input_yaw = self.fc1(input_yaw)
        input_pitch = self.fc2(input_pitch)
        input_yaw = input_yaw.view(batch_size, 1, self.dim[0]//4, self.dim[1]//4)
        input_pitch = input_pitch.view(batch_size, 1, self.dim[0]//4, self.dim[1]//4)

        input_bottleneck = torch.cat((input_image, input_yaw, input_pitch), 1)

        input_deimage = self.conv_out(input_bottleneck)
        input_deimage = nn.LeakyReLU()(input_deimage)
        input_deimage = self.bn(input_deimage) 
        input_deimage = self.decoder(input_deimage)
        return input_deimage

class GazeRepresentationLearning(nn.Module): 
    # gaze representation learnings, input: 1 image, output 2-dim vector (yaw and pitch)
    def __init__(self, channels=[64,64,128,128], expand=[False,True,False,True]) -> None:
        super(GazeRepresentationLearning, self).__init__()

        self.features= nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=channels[0], kernel_size=(3,3), padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(channels[0]),
            # nn.Tanh(),
        )
        
        assert len(channels) == len(expand)

        for layer_index in range(len(channels)):
            self.features.add_module(
                f"pool_{layer_index}", 
                nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
            )
            self.features.add_module(
                f"ConvBlock_{layer_index}", 
                ConvBlock(input_channel=channels[layer_index], 
                last= (not expand[layer_index]))
            )

        self.decoder = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2,3), stride=(2,3)), # 1/3 1/4 for EyeDiap
            nn.Flatten(),
            nn.Linear(in_features=2*channels[-1], out_features=channels[-1]//8),
            nn.Linear(in_features=channels[-1]//8, out_features=2),
        )
        self.features.add_module("decoder", self.decoder)

        self.channels = channels[0]


    def forward(self, input):
        input = self.features(input)
        return input

class UnsupervisedGazeNetwork(nn.Module):
    # Entire unsupervised architecture 
    # (   
    #     combination of 
    #     1: 2 shared gaze representation learnings, input: 1 image, output 2-dim vector (yaw and pitch)
    #     2: global alignment network, input: 2 images, output: 1 image
    #     3: gaze redirection network, input: image, 2 2-dim vectors, output: 1 image
    # )

    def __init__(self, height=36, width=60) -> None:
        super(UnsupervisedGazeNetwork, self).__init__()
        self.shared_gazeEstimation = GazeRepresentationLearning()
        self.gazeRedirection = GazeRedirectionNetwork(dim=(height, width))
        self.align = GlobalAlignmentNetwork()
        self.VGG = [VGG_3, VGG_8, VGG_13]
        
    def forward(self, input_i, input_o):
        angle_i = self.shared_gazeEstimation(input_i)
        angle_o = self.shared_gazeEstimation(input_o)

        input_i_t = self.align(input_i, input_o)

        angle_dif = angle_i - angle_o
        angle_dif_yaw = angle_dif[:,0]
        angle_dif_pitch = angle_dif[:,1]
        grid_i_t = self.gazeRedirection(input_i_t,  angle_dif_yaw, angle_dif_pitch)
        grid_i_t = torch.permute(grid_i_t, (0, 2, 3, 1))
        output = F.grid_sample(input_i_t, grid_i_t)
        feature_i, feature_o = [], []
        for each in self.VGG:
            feature_i.append(each(input_o))
            feature_o.append(each(output))
        return output, feature_i, feature_o

if __name__ == "__main__":

    # test the model
    model = GazeRepresentationLearning()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input = torch.rand(size=(2,1,36,60)).to(device)
    model.to(device)
    outputs = model(input)
    assert outputs.size() == (2,2)
    print("GazeRepresentationLearning passed")
    print(model)

    model = GazeRedirectionNetwork()
    model.to(device)
    yaw, pitch = torch.rand(size=(2,1)).to(device), torch.rand(size=(2,1)).to(device)
    outputs = model(input, yaw, pitch)
    assert outputs.size() == (2,2,36,60)
    print("GazeRedirectionNetwork passed")
    print(model)

    model = GlobalAlignmentNetwork()
    model.to(device)
    output = model(input, input)
    assert output.size() == input.size()
    print("GlobalAlignmentNetwork passed")
    print(model)
