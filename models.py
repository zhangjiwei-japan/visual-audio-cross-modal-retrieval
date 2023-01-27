import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models

class VGGNet(nn.Module):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet, self).__init__()
        self.vgg = models.vgg19_bn(pretrained=True)
        self.vgg_features = self.vgg.features
        self.fc_features = nn.Sequential(*list(self.vgg.classifier.children())[:-2])

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = self.vgg_features(x).view(x.shape[0], -1)
        features = self.fc_features(features)
        return features


class ImgNN(nn.Module):
    """Network to learn image representations"""
    def __init__(self, input_dim=4096, output_dim=1024):
        super(ImgNN, self).__init__()
        self.denseL1 = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        out = F.relu(self.denseL1(x))
        return out


class AudioNN(nn.Module):
    """Network to learn text representations"""
    def __init__(self, input_dim=1024, output_dim=1024):
        super(AudioNN, self).__init__()
        self.denseL1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = F.relu(self.denseL1(x))
        return out


class Cross_Modal_NN(nn.Module):
    """Network to learn text representations"""
    def __init__(self, img_input_dim=1024, img_output_dim=1024,
                 text_input_dim=128, text_output_dim=1024, minus_one_dim=512, output_dim=10):
        super(Cross_Modal_NN, self).__init__()
        self.img_net = ImgNN(img_input_dim, img_output_dim)
        self.text_net = AudioNN(text_input_dim, text_output_dim)
        self.linearLayer = nn.Linear(img_output_dim, minus_one_dim)
        self.linearLayer2 = nn.Linear(minus_one_dim, output_dim)

    def forward(self, img, text):
        view1_feature_out = self.img_net(img)
        view2_feature_out = self.text_net(text)
        view1_feature = self.linearLayer(view1_feature_out)
        view2_feature = self.linearLayer(view2_feature_out)

        view1_predict = self.linearLayer2(view1_feature)
        view2_predict = self.linearLayer2(view2_feature)
        return view1_feature, view2_feature, view1_predict, view2_predict


if __name__=='__main__':
    model_ft = Cross_Modal_NN(img_input_dim=1024, text_input_dim=128, output_dim=10)