import torch
import torch.nn as nn
from Center_Loss import center_loss

def calc_label(label_1, label_2):
    Sim = label_1.float().mm(label_2.float().t())
    return Sim

def loss_vae(re_x,x,mu,log_var):
    loss_MSE = nn.MSELoss()
    mse_loss = loss_MSE(re_x,x)
    KLD = 0.5 * torch.sum(torch.exp(log_var) + torch.pow(mu, 2) - 1. - log_var)
    loss = mse_loss + KLD
    return loss

def calc_loss(inputs_visual, inputs_audio,img_out_put,text_out_put,view1_logvar,view2_logvar,view1_mu,view2_mu,new_view1_feature, 
            new_view2_feature, view1_predict, view2_predict, labels_1, labels_2,samam,lamada,alpha, beta,gamma):
    

    view1_label= torch.argmax(labels_1, dim=1).float()
    view2_label=  torch.argmax(labels_2, dim=1).float()
    loss_center = 0.5* center_loss(new_view1_feature, view1_label, 0.5) + 0.5* center_loss(new_view2_feature,view2_label, 0.5)

    discriminate = ((view1_predict-labels_1.float())**2).sum(1).sqrt().mean() + ((view2_predict-labels_2.float())**2).sum(1).sqrt().mean() 

    cos = lambda x, y: x.mm(y.t()) / ((x ** 2).sum(1, keepdim=True).sqrt().mm((y ** 2).sum(1, keepdim=True).sqrt().t())).clamp(min=1e-6) / 2.
    view1_feature_new = new_view1_feature-torch.mean(new_view1_feature,dim=1,keepdim=True)
    view2_feature_new = new_view2_feature-torch.mean(new_view2_feature,dim=1,keepdim=True)
    theta11 = cos(view1_feature_new, view1_feature_new)
    theta12 = cos(view1_feature_new, view2_feature_new)
    theta22 = cos(view2_feature_new, view2_feature_new)
    Sim11 = calc_label(labels_1, labels_1).float()
    Sim12 = calc_label(labels_1, labels_2).float()
    Sim22 = calc_label(labels_2, labels_2).float()
    correlation1 = ((1+torch.exp(theta11)).log() - Sim11 * theta11).mean()
    correlation2 = ((1+torch.exp(theta12)).log() - Sim12 * theta12).mean()
    correlation3 = ((1 + torch.exp(theta22)).log() - Sim22 * theta22).mean()
    correlation = correlation1 + correlation2 + correlation3
    distance = ((new_view1_feature - new_view2_feature)**2).sum(1).sqrt().mean()

    kl_loss_view1 = loss_vae(img_out_put, inputs_visual,view1_mu,view1_logvar)
    kl_loss_view2 = loss_vae(text_out_put, inputs_audio,view2_mu,view2_logvar)

    VAE_loss = kl_loss_view1+kl_loss_view2

    im_loss = samam*VAE_loss + lamada *discriminate+ alpha * correlation  + beta * distance + gamma *loss_center

    return im_loss, samam*VAE_loss, lamada *discriminate, alpha * correlation ,  beta * distance, gamma *loss_center