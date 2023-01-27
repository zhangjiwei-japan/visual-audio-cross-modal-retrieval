import torch
import torch.nn as nn
from torch.autograd import Variable
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

class visualNN(nn.Module):
    """Network to learn image representations"""
    def __init__(self, input_dim=1024, output_dim=512):
        super(visualNN, self).__init__()
        self.denseL1 = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        out = F.relu(self.denseL1(x))
        return out

class AudioNN(nn.Module):
    """Network to learn audio representations"""
    def __init__(self, input_dim=128, output_dim=512):
        super(AudioNN, self).__init__()
        self.denseL1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = F.relu(self.denseL1(x))
        return out

class DEEP_C_CENTER_NN(nn.Module):
    """Network to learn audio representations"""
    def __init__(self, visual_input_dim=1024, visual_output_dim=512,
                 audio_input_dim=128, audio_output_dim=512, minus_one_dim=64, output_dim=10):
        super(DEEP_C_CENTER_NN, self).__init__()
        
        #encoder
        self.visual_net = visualNN(visual_input_dim, visual_output_dim) # 1024 512
        self.audio_net = AudioNN(audio_input_dim, audio_output_dim) # 128 512
        self.linearLayer1 = nn.Linear(visual_output_dim, minus_one_dim) # 512 64
        self.linearLayer2 = nn.Linear(minus_one_dim, output_dim) # 64 10
        #decoder

        self.linearLayer3 = nn.Linear(output_dim, minus_one_dim) # 10 64
        self.linearLayer4 = nn.Linear(minus_one_dim, visual_output_dim) # 64 512 
        self.visual_out = nn.Linear(visual_output_dim, visual_input_dim) # 512 1024
        self.audio_out = nn.Linear(audio_output_dim, audio_input_dim) # 512 128
        #mu,logvar
        self.mu = nn.Linear(minus_one_dim, minus_one_dim) # 64 64
        self.var = nn.Linear(minus_one_dim, minus_one_dim) # 64 64

    def reparametrization(self,view1_logvar,view2_logvar,view1_mu,view2_mu):
        if self.training:
            # print("training...")
            view1_std = torch.exp(view1_logvar / 2)
            view1_eps = Variable(view1_std.data.new(view1_std.size()).normal_())
            view1_feature_new = view1_eps.mul(view1_std).add_(view1_mu)

            view2_std = torch.exp(view2_logvar / 2)
            view2_eps = Variable(view2_std.data.new(view2_std.size()).normal_())
            view2_feature_new = view2_eps.mul(view2_std).add_(view2_mu)

            return view1_feature_new, view2_feature_new
        else:
            # print("evaluating...")

            return view1_mu,view2_mu

    def forward(self, visual, audio):
        view10_feature = self.visual_net(visual) # 1024 512
        view20_feature = self.audio_net(audio) # 128 512
        view1_feature = self.linearLayer1(view10_feature) #512 64 
        view2_feature = self.linearLayer1(view20_feature) #512 64

        view1_logvar = self.var(view1_feature)
        view2_logvar = self.var(view2_feature)
        view1_mu = self.mu(view1_feature)
        view2_mu = self.mu(view2_feature)

        new_view1_feature, new_view2_feature = self.reparametrization(view1_logvar,view2_logvar,view1_mu,view2_mu)

        view1_predict = self.linearLayer2(new_view1_feature) #64 10
        view2_predict = self.linearLayer2(new_view2_feature) #64 10

        # view1_mid = self.linearLayer3(view1_predict) #64 512
        # view2_mid = self.linearLayer3(view2_predict) #64 512
        view1_out_put = self.linearLayer4(new_view1_feature) #64 512
        view2_out_put = self.linearLayer4(new_view2_feature) #64 512

        visual_out_put = self.visual_out(view1_out_put) #512 1024
        audio_out_put = self.audio_out(view2_out_put) #512 128

        
        return view1_logvar,view2_logvar,view1_mu,view2_mu,new_view1_feature, new_view2_feature, view1_predict, view2_predict,visual_out_put,audio_out_put

if __name__=='__main__':
    input1 = torch.randn(128,1024)
    input2 = torch.randn(128,128)
    model_ft = DEEP_C_CENTER_NN(visual_input_dim=1024, audio_input_dim=128, output_dim=10)
    # model_ft.train()
    model_ft.eval()
    view1_logvar,view2_logvar,view1_mu,view2_mu,new_view1_feature, new_view2_feature, \
    view1_predict, view2_predict,visual_out_put,audio_out_put = model_ft(input1,input2)
    print(new_view1_feature.shape, new_view2_feature.shape)