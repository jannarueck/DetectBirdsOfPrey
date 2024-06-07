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


import cv2
from pydub import AudioSegment
def fuse_data(video, audio_f, image_f, thresh = 20000):
    cap = cv2.VideoCapture(video)
    audio = video[:-4] + '.wav'
    video_to_audio(video, audio)
    audio1 = AudioSegment.from_mp3(audio)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Frames per second: {0}".format(fps))
    frame_len = (1 / fps) * 1000  # in milliseconds
    i=0
    j = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        if i*frame_len > 2000:
            print("Save frame "+ str(j))
            cv2.imwrite(image_f +'/frame'+str(j)+'.jpg',frame)
            audio2 = audio1[(i*frame_len)-2000:i*frame_len]
            audio2.export(audio_f + '/sample'+str(j)+'.wav', format="wav")
            j += 1
        if j > thresh:
            break 
        i+=1
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type = str, default="../data/TEST/C2.mp4", help='input video')
    parser.add_argument('--audio_f', type = str, default="../data/TEST/cam2_audio", help='folder to save audio samples')
    parser.add_argument('--img_f', type = str, default="../data/TEST/cam2_img", help='folder to save video frames')
    parser.add_argument('--thresh', type = int, default=20000, help='threshold of maximum samples/frames to be extracted')


    args = parser.parse_args()

    fuse_data(args.video, args.audio_f, args.img_f, args.thresh)
        


