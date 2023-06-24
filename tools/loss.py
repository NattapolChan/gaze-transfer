import torch
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self, height=36, width=60, coef_pixel=0.5, coef_perceptual=0.02, coef_spatial=0.1, coef_warp_reg=0.25) -> None: 
        super(Loss, self).__init__()
        self.height= height
        self.width = width
        self.coef_pixel = coef_pixel
        self.coef_perceptual = coef_perceptual
        self.coef_spatial = coef_spatial
        self.loss__style = 0
        self.loss__feature = 0
        self.loss__pixel = 0

    def pixel_loss(self, image_i_t, image_o):
        discrepancy = torch.abs(image_i_t - image_o)
        
        self.loss__pixel = 1/torch.numel(image_i_t) * torch.sum(discrepancy)
        return self.loss__pixel
    
    def feature_loss(self, feature_i, feature_o):
        assert len(feature_i) == len(feature_o), f'uncompatible feature size: {len(feature_i)} and {len(feature_o)}'
        assert feature_i[0].size() == feature_o[0].size(), "intermediate feature got different size"
        self.loss__feature = 0
        for i in range(len(feature_i)):
            self.loss__feature += torch.linalg.norm(feature_i[i]-feature_o[i]) / torch.numel(feature_i[i])
        return self.loss__feature

    def style_loss(self, feature_i, feature_o):
        assert len(feature_i) == len(feature_o), f'uncompatible feature size: {len(feature_i)} and {len(feature_o)}'
        assert feature_i[0].size() == feature_o[0].size(), "intermediate feature got different size"
        accumulate = 0
        for each_layer in range(len(feature_i)):
            channel_size = feature_o[each_layer].size(1)
            spatial_size = feature_i[each_layer].size(2) * feature_i[each_layer].size(3)
            mat_i = torch.flatten(feature_i[each_layer], start_dim=1)
            mat_o = torch.flatten(feature_o[each_layer], start_dim=1)
            gram_i = 1/spatial_size * torch.matmul(mat_i, torch.transpose(mat_i, 0, 1))
            gram_o = 1/spatial_size * torch.matmul(mat_o, torch.transpose(mat_o, 0, 1))

            assert mat_i.size() == (channel_size, spatial_size), f"[LOSS]: mismatch size >> og matrix {mat_i.size()}"
            assert gram_i.size() == (channel_size, channel_size), f"[LOSS]: mismatch size >> gram matrix {gram_i.size()}"

            accumulate += 1/channel_size * 1/channel_size * 1/feature_o[each_layer].size(0) * torch.linalg.norm(gram_i-gram_o)
        return accumulate
        
    def forward(self, image_i_t, image_o, feature_i, feature_o):
        return self.coef_pixel*self.pixel_loss(image_i_t, image_o) + self.coef_perceptual*self.feature_loss(feature_i, feature_o) + self.coef_spatial * self.style_loss(feature_i, feature_o)
