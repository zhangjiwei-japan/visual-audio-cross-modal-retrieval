import torch
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from torch.utils.data import Dataset,DataLoader,TensorDataset
# import torch.nn.functional as F
import h5py

def load_data_visual_audio(load_path,num_class):
    f = h5py.File(load_path,'r')
    f.keys()
    lab_test = f["lab_test"]
    visual_test = f["visual_test"]
    audio_test = f["audio_test"]
    visual_train = f["visual_train"]
    audio_train = f["audio_train"]
    lab_train = f["lab_train"] 

    # 标签onehot编码
    y_train_onehot, y_test_onehot = to_categorical(lab_train, num_classes=num_class),to_categorical(lab_test, num_classes=num_class)
    # y_train_onehot, y_test_onehot = F.one_hot(lab_train, num_classes=10),F.one_hot(lab_test, num_classes=10)

    return visual_train,audio_train,lab_train,visual_test,audio_test,lab_test,y_train_onehot, y_test_onehot 

def load_dataset_train(load_path,train_size):
    f = h5py.File(load_path,'r')
    f.keys()
    visual_train = f["visual_train"]
    audio_train = f["audio_train"]
    lab_train = f["lab_train"]

    lab_train = torch.tensor(lab_train)
    lab_train = lab_train.view(lab_train.size(0))
    lab_train = lab_train.long()
    train_visual = TensorDataset(torch.tensor(visual_train).float(), lab_train)
    train_audio = TensorDataset(torch.tensor(audio_train).float(), lab_train)

    data_loader_visual = DataLoader(dataset=train_visual, batch_size=train_size, shuffle=False)
    data_loader_audio = DataLoader(dataset=train_audio, batch_size=train_size, shuffle=False)

    return data_loader_visual,data_loader_audio

def load_dataset_test(load_path,test_size):
    f = h5py.File(load_path,'r')
    f.keys()
    lab_test = f["lab_test"]
    visual_test = f["visual_test"]
    audio_test = f["audio_test"]
    
    # print(np.any(np.isnan(lab_train)))
    # print(len(lab_test))
    # print(np.any(np.isnan(visual_test)))
    lab_test = torch.tensor(np.array(lab_test))
    lab_test = lab_test.view(lab_test.size(0))
    lab_test = lab_test.long()
    test_visual = TensorDataset(torch.tensor(visual_test).float(), lab_test)
    test_audio = TensorDataset(torch.tensor(audio_test).float(), lab_test)

    data_loader_visual = DataLoader(dataset=test_visual, batch_size=test_size, shuffle=False)
    data_loader_audio = DataLoader(dataset=test_audio, batch_size=test_size, shuffle=False)

    return data_loader_visual,data_loader_audio

def save_data(path,visual_train,audio_train,lab_train,visual_test,audio_test,lab_test):
    with h5py.File(path, 'w') as f:
        f.create_dataset("lab_train", data=lab_train)
        f.create_dataset("lab_test", data=lab_test)  
        f.create_dataset("visual_test", data=visual_test)
        f.create_dataset("audio_test", data=audio_test)
        f.create_dataset("visual_train", data=visual_train)
        f.create_dataset("audio_train", data=audio_train)
        
        f.close()  

if __name__=='__main__':
    save_path = 'ave_feature_norm.h5'# 'vegas_feature_nrom.h5'
    data_path = "data_sets/AVE_feature_updated_squence.h5" # vegas_feature.h5 
    num_class = 15
    visual_train,audio_train,lab_train,visual_test,audio_test,lab_test,y_train_onehot, y_test_onehot  = load_data_visual_audio(data_path,num_class)
    save_data(save_path,visual_train,audio_train,lab_train,visual_test,audio_test,lab_test)