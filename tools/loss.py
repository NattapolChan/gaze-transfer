import torch
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self, height=36, width=60, coef_pixel=0.5, coef_perceptual=0.5, coef_spatial=0.2, coef_warp_reg=0.25) -> None: 
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
        if len(feature_i) == 0: return 0
        assert len(feature_i) == len(feature_o), f'uncompatible feature size: {len(feature_i)} and {len(feature_o)}'
        assert feature_i[0].size() == feature_o[0].size(), "intermediate feature got different size"

        self.loss__feature = 0
        for i in range(len(feature_i)):
            _, c, w, h = feature_i[i].size()
            a = torch.linalg.norm(feature_i[i]-feature_o[i]) / (c * w * h)
            self.loss__feature += a 
        return self.loss__feature

    def get_gram(self, mat_i):

        assert len(mat_i.size()) == 4

        b, d, h, w = mat_i.size()
        mat_i = mat_i.view(b * d, h * w)
        
        gram_i = torch.mm(mat_i, mat_i.t())
        return gram_i / (h * w)

    def style_loss(self, feature_i, feature_o):
        if len(feature_i)==0: return 0
        assert len(feature_i) == len(feature_o), f'uncompatible feature size: {len(feature_i)} and {len(feature_o)}'
        assert feature_i[0].size() == feature_o[0].size(), "intermediate feature got different size"

        accumulate = 0
        for each_layer in range(len(feature_i)):
            b, c = feature_o[each_layer].size(0), feature_o[each_layer].size(1)
            gram_i = self.get_gram(feature_i[each_layer])
            gram_o = self.get_gram(feature_o[each_layer])

            a = 1/(c*c) * torch.linalg.norm(gram_i-gram_o)
            accumulate += a

        self.loss__style = accumulate
        return accumulate
        
    def forward(self, image_i_t, image_o, feature_i, feature_o):
        return self.coef_pixel*self.pixel_loss(image_i_t, image_o) + self.coef_perceptual*self.feature_loss(feature_i, feature_o) + self.coef_spatial * self.style_loss(feature_i, feature_o)
