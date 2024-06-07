import numpy as np
import pandas as pd
import librosa
import os
import torch
import torch
import librosa
import argparse
from sklearn.preprocessing import LabelEncoder

def extract_melspec(file_name):
    audio, sr = librosa.load(file_name)
    n_fft = 2048
    hop_length = 512

    mel_signal = librosa.feature.melspectrogram(y=audio, sr=sr, hop_length=hop_length, n_fft=n_fft, window='hann', n_mels = 88)
    logspec = librosa.power_to_db(mel_signal, ref = 1.0)
    x = logspec
    norm = (x-np.min(x))/(np.max(x)-np.min(x))
    if norm.shape[1] != 87:
       print("Wrong shape:"+file_name)
    #show_spectogram(norm, sr)
    return norm


import re
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
    return sorted(data, key=alphanum_key)


def extract_features(audio, img, thresh, fname):
    features = []

    for i, file in enumerate(sorted_alphanumeric(os.listdir(audio))):        
        file_name = os.fsdecode(file)
        #print("Audio Path:", file_name) 
        idx = int(file_name[6:-4])
        class_label = 0
        if idx > thresh:
            class_label = 1

        img_path = img + '/frame'+file_name[6:-4] + '.jpg'
        print("extract data pair ", idx, " from ", fname[:-3])  
        melspec = extract_melspec(audio + file_name)
        features.append([img_path, melspec, class_label])

    # Convert into a Panda dataframe 
    featuresdf = pd.DataFrame(features, columns=['img', 'audio','class_label'])

    # Convert features and corresponding classification labels into numpy arrays
    img = np.array(featuresdf.img.tolist())
    audio = np.array(featuresdf.audio.tolist())
    y = np.array(featuresdf.class_label.tolist())

    # Encode the classification labels
    le = LabelEncoder()
    yy = le.fit_transform(y)
    
    audio = torch.unsqueeze(torch.from_numpy((audio)),1)
    #img = torch.from_numpy(img)
    label = torch.from_numpy(yy)

    print(img.shape, audio.shape, y.shape)
    file = torch.save([img, audio, label], fname)
    

def combine_datasets(file1, file2, fname):
    img1, audio1, labels1 = torch.load(file1)
    img2, audio2, labels2 = torch.load(file2)
    images = np.append(img1, img2)
    audio = torch.concat((audio1, audio2))
    labels = np.append(labels1, labels2)
    print(images.shape, audio.shape, labels.shape)
    torch.save([images, audio, labels], fname)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type = int, default=1, help='1 to create dataset, 0 to combine datasets')
    parser.add_argument('--audio_f', type = str, default="data/TRAIN/TI_audio/", help='folder with audio samples')
    parser.add_argument('--img_f', type = str, default="data/TRAIN/TI_img/", help='folder with image frames')
    parser.add_argument('--thresh', type = int, default=4436, help='frame/sample number where class changes from 0 to 1')
    parser.add_argument('--fname', type = str, default="TI.pt", help='file name of created dataset')
    parser.add_argument('--dataset1', type = str, default="C1_YT1.pt", help='first dataset to combine')
    parser.add_argument('--dataset2', type = str, default="H2.pt", help='second dataset to combine')



    args = parser.parse_args()
    if args.task:
        extract_features(args.audio_f, args.img_f, args.thresh, args.fname)
    else:
        combine_datasets(args.dataset1, args.dataset2, args.fname)
