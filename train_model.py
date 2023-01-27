import torch
import math
import torch.optim as optim
from loss_functions import *
from load_data import *
from torch.autograd import Variable
from model_vae import DEEP_C_CENTER_NN
from evaluate import fx_calc_map_label



def train_model(samam,lamada,alpha ,beta, gamma,lr,num_epoch,data_path):
    
    betas = (0.5, 0.999)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = DEEP_C_CENTER_NN(visual_input_dim=1024, audio_input_dim=128, output_dim=num_class)
    params_to_update = list(model_ft.parameters()) 
    warmup_t = 50
    n_t = 0.5
    lambda_warmup = lambda epoch: (0.9*epoch / warmup_t+0.1) if epoch < warmup_t else  0.1  if n_t * (1+math.cos(math.pi*(epoch - warmup_t)/(num_epoch-warmup_t)))<0.1 else n_t * (1+math.cos(math.pi*(epoch - warmup_t)/(num_epoch-warmup_t)))
    optimizer = optim.Adam(params_to_update, lr=lr, betas=betas)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_warmup)
    visual_train,audio_train,lab_train,visual_test,audio_test,lab_test,y_train_onehot, y_test_onehot  = load_data_visual_audio(data_path,num_class)

    batch_size = 128
    train_step = len(visual_train)//batch_size

    for epoch in range(num_epoch):
        model_ft.train()
        running_loss =0

        for batch_idx in range(train_step):
            optimizer.zero_grad()
            audio_label_input = torch.from_numpy(y_train_onehot[batch_idx*batch_size:(batch_idx+1)*batch_size])
            # print(X_train[batch_idx*batch_size:(batch_idx+1)*batch_size,:].shape)
            audio_label_input = audio_label_input.float()
            # audio_data_input = audio_data_input.cuda()
            audio_label_input = Variable(audio_label_input)
            label_audio= audio_label_input.to("cpu")

            audio_data_input = torch.from_numpy(audio_train[batch_idx*batch_size:(batch_idx+1)*batch_size,:])
            # print(X_train[batch_idx*batch_size:(batch_idx+1)*batch_size,:].shape)
            audio_data_input = audio_data_input.float()
            # audio_data_input = audio_data_input.cuda()
            audio_data_input = Variable(audio_data_input)
            inputs_audio= audio_data_input.to("cpu")

            visual_label_input = torch.from_numpy(y_train_onehot[batch_idx*batch_size:(batch_idx+1)*batch_size])
            # print(X_train[batch_idx*batch_size:(batch_idx+1)*batch_size,:].shape)
            visual_label_input = visual_label_input.float()
            # audio_data_input = audio_data_input.cuda()
            visual_label_input = Variable(visual_label_input)
            label_visual= visual_label_input.to("cpu")

            visual_data_input = torch.from_numpy(visual_train[batch_idx*batch_size:(batch_idx+1)*batch_size,:])
            visual_data_input = visual_data_input.float()
            # audio_data_input = audio_data_input.cuda()
            visual_data_input = Variable(visual_data_input)
            inputs_visual= visual_data_input.to("cpu")

            label_input = torch.from_numpy(lab_train[batch_idx*batch_size:(batch_idx+1)*batch_size])
            # print(X_train[batch_idx*batch_size:(batch_idx+1)*batch_size,:].shape)
            label_input = label_input.float()
            # audio_data_input = audio_data_input.cuda()
            label_input = Variable(label_input)
            label= label_input.to("cpu")          
        
            logvar_view1,logvar_view2,mu_view1,mu_view2,new_view1_feature,new_view2_feature,\
            view1_predict,view2_predict,visual_out_put,text_out_put = model_ft(inputs_visual, inputs_audio)
            loss, VAE_loss,term1,term2,term3,loss_center = calc_loss(inputs_visual, inputs_audio,visual_out_put,text_out_put,logvar_view1,logvar_view2,mu_view1,mu_view2,new_view1_feature, new_view2_feature, view1_predict, view2_predict, label_visual, label_audio, samam,lamada,alpha, beta,gamma)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # scheduler.step()

        print('Epoch {}/{},samam {},lamada {},alpha {},beta {},gamma {},lr {:.5f}, all_loss: {:.4f}'.format(epoch, num_epoch, samam,lamada,alpha ,beta, gamma ,optimizer.param_groups[0]['lr'] ,running_loss))
        if epoch >0 and epoch%5==0:
            torch.save(model_ft.state_dict(), 'save_models/vae_visual_ccn_{}_ave.pth'.format(epoch))

def evaluate_model(model_path,data_path):
    
    model_ft = DEEP_C_CENTER_NN(visual_input_dim=1024, audio_input_dim=128, output_dim=num_class)
    model_ft.load_state_dict(torch.load(model_path))
    model_ft.eval()
    
    # load_path = "D:/DATASETS/Dataset_audiovisual/vegas_feature.h5" # vegas_feature.h5  AVE_feature_updated_squence.h5
    visual_train,audio_train,lab_train,visual_test,audio_test,lab_test,y_train_onehot, y_test_onehot  = load_data_visual_audio(data_path,num_class)
    print('...Evaluation on testing data...')
    with torch.no_grad():
        audio_label_input = torch.from_numpy(y_test_onehot[:])

        audio_label_input = audio_label_input.float()

        audio_label_input = Variable(audio_label_input)
        label_audio= audio_label_input.to("cpu")

        audio_data_input = torch.from_numpy(audio_test[:,:])

        audio_data_input = audio_data_input.float()

        audio_data_input = Variable(audio_data_input)
        inputs_audio= audio_data_input.to("cpu")
 
        visual_label_input = torch.from_numpy(y_test_onehot[:])

        visual_label_input = visual_label_input.float()

        visual_label_input = Variable(visual_label_input)
        label_visual= visual_label_input.to("cpu")

        visual_data_input = torch.from_numpy(visual_test[:,:])
        visual_data_input = visual_data_input.float()
 
        visual_data_input = Variable(visual_data_input)
        inputs_visual= visual_data_input.to("cpu")

        logvar_view1,logvar_view2,mu_view1,mu_view2,new_view1_feature, new_view2_feature, \
        view1_predict, view2_predict,visual_out_put,text_out_put = model_ft(inputs_visual, inputs_audio)
        view1_feature = new_view1_feature.detach().cpu().numpy()
        view2_feature = new_view2_feature.detach().cpu().numpy()
        # view1_predict = view1_predict.detach().cpu().numpy()
        # view2_predict = view2_predict.detach().cpu().numpy()
        view1_label = torch.argmax(view1_predict, dim=1).detach().cpu().numpy()
        view2_label = torch.argmax(view2_predict, dim=1).detach().cpu().numpy()
        label = torch.argmax(label_visual, dim=1).detach().cpu().numpy()

        print('...Evaluation...')

        visual_to_audio = fx_calc_map_label(view1_feature, view2_feature, label)
        print('...visual to Audio MAP = {}'.format(visual_to_audio))
        audio_to_visual = fx_calc_map_label(view2_feature, view1_feature, label)
        # result_show8 =  keras.metrics.top_k_categorical_accuracy(view2_label, view1_label, k=5)
        print('...Audio to visual MAP = {}'.format(audio_to_visual))
        MAP = (visual_to_audio + audio_to_visual) / 2.
        print('...Average MAP = {}'.format(MAP))
       

import time
if __name__=='__main__':
    data_path = "data_sets/ave_feature_norm.h5" # vegas_feature.h5  AVE_feature_updated_squence.h5
    now =  time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    lr = 1e-4 #3.5e-3 
    num_epoch = 500
    alpha = 0.001
    beta =  0.1
    gamma = 0.01
    samam = 0.0001
    lamada = 1
    num_class = 15
    model_path = 'save_models/vae_visual_ccn_'+ str(num_epoch-5)+"_ave.pth"
    train_model(samam,lamada,alpha,beta, gamma ,lr,num_epoch,data_path)
    evaluate_model(model_path,data_path)