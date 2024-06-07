import moviepy.editor
import numpy as np
import librosa.effects
import soundfile as sf
import argparse
from pydub.utils import make_chunks


#convert .mp4 file to .wav file
def video_to_audio(videofile, audiofile):
    #Load the Video
    video = moviepy.editor.VideoFileClip(videofile)

    #Extract the Audio
    audio = video.audio

    #Export the Audio
    audio.write_audiofile(audiofile)


from pydub import AudioSegment

#extract part of the audio file
def split_audio(file, name, t1, t2):
    audio1 = AudioSegment.from_mp3(file)
    audio1 = audio1[t1:t2]
    audio1.export(name, format="wav")


#split_audio("YT/Calm chicken sounds.wav", "YT/Calm chicken sounds3.wav", 1890000, 2490000 )

def mp3_to_wav(file, name):
    audio1 = AudioSegment.from_mp3(file)
    audio1.export(name, format="wav")


from pytube import YouTube


def download_audio_from_YT(ref, fname):
    video = YouTube(ref)
    stream = video.streams.filter(only_audio=True)
    #print(stream)
    stream = video.streams.get_by_itag(140)
    stream.download(filename=fname)

#download_audio_from_YT('https://www.youtube.com/watch?v=mkCyXuawdFk', 'Chicken Alarm Call Over Hawk.wav')
#split_audio("C.wav", "Chicken Alarm Call Over Hawk.wav", 5000,30000)

import pickle

def get_data_from_pkl():
    with open('Continuous_segment0.pkl', 'rb') as f:
        data = pickle.load(f)
        X = data['x']
        y = data['y']
        #print(X[0].shape, len(y))

import re
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
    return sorted(data, key=alphanum_key)

import os 

def sample_generator(audio, folder, name):
    myaudio = AudioSegment.from_file(audio , "wav") 
    chunk_length_ms = 2000 # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec
    #Export all of the individual chunks as wav files

    for i, chunk in enumerate(chunks):
            chunk_name = name + "{0}.wav".format(i)
            print("exporting", chunk_name)
            chunk.export(folder + chunk_name, format="wav")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--net', type=str, default='canet', help='net type')
    parser.add_argument('--mp3', type = int, default=1, help='mp3 file or not')
    parser.add_argument('--mp4', type = int, default=0, help='mp4 file or not')
    parser.add_argument('--split', type = int, default=1, help='split file or not')
    parser.add_argument('--start', type = int, default=0, help='start of file')
    parser.add_argument('--end', type = int, default=5000, help='end of file')
    parser.add_argument('--input_data',type=str, default='audio.mp3', help='input file path')
    parser.add_argument('--chunk_name',type=str, default='a', help='name of audio chunks')
    parser.add_argument('--save_folder',type=str, default='./alarm calls/', help='saved path of input data')

    args = parser.parse_args()

    audio_path = args.input_data[:-4] + '.wav'
    if args.mp4:
        print(audio_path)
        video_to_audio(args.input_data, audio_path)
    elif args.mp3:
        mp3_to_wav(args.input_data, audio_path)
    if args.split:
        split_audio(audio_path, audio_path, args.start, args.end)
    sample_generator(audio_path, args.save_folder, args.chunk_name)
        


