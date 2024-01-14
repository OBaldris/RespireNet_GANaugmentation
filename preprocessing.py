import os
from pydub import AudioSegment
import os
import librosa
import numpy as np
from scipy.interpolate import interp1d
import librosa.display
import matplotlib.pyplot as plt
import cv2
import cmapy
import librosa
import cv2
import numpy as np


##README
#just chose the directory containing the wav files and the directory where you want to save the output



def cut_audio_for_directory(input_folder, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process each pair of .wav and .txt files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            wav_path = os.path.join(input_folder, filename)
            txt_path = os.path.join(input_folder, filename.replace(".wav", ".txt"))

            # Check if the corresponding .txt file exists
            if os.path.exists(txt_path):
                cut_audio(wav_path, txt_path, output_folder)
            else:
                print(f"Warning: No corresponding .txt file found for {filename}")

def cut_audio(input_wav, input_txt, output_folder):
    # Load audio file
    audio = AudioSegment.from_wav(input_wav)

    # Read time points from the text file
    with open(input_txt, 'r') as file:
        time_points = [line.strip().split('\t') for line in file.readlines()]

    # Cut the audio file at specified time points
    for i, (start_time, end_time, mark1, mark2) in enumerate(time_points):
        start_time, end_time = float(start_time), float(end_time)
        time_difference = end_time - start_time

        # Skip saving if time difference is lower than one second
        if mark1 == '1' and mark2 == '1':
            segment = audio[int(start_time * 1000):int(end_time * 1000)]
             # Save the segment to a new file
            output_file = f"{output_folder}/mark{mark1}{mark2}_{os.path.splitext(os.path.basename(input_wav))[0]}_segment_{i + 1}_mark{mark1}{mark2}.wav"
            segment.export(output_file, format="wav")
            print(f"Segment {i + 1} of {os.path.basename(input_wav)} saved to {output_file}")
        # Save the mel spectrogram to a new file
            img=create_mel(output_file, n_mels=28, f_min=50, f_max=4000, nfft=2048, hop=512, resz=1)
            output_image = f"{output_folder}/mark{mark1}{mark2}_{os.path.splitext(os.path.basename(input_wav))[0]}_segment_{i + 1}_mark{mark1}{mark2}.png"
            cv2.imwrite(output_image, img)
            print(f"Image {i + 1} of {os.path.basename(input_wav)} saved to {output_image}")


def create_mel(input_wav, n_mels=28, f_min=50, f_max=4000, nfft=2048, hop=512, resz=1, fixed_size=28):
    y, sr = librosa.load(input_wav)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmin=f_min, fmax=f_max, n_fft=nfft, hop_length=hop)
    S = librosa.power_to_db(S, ref=np.max)
    S = (S-S.min()) / (S.max() - S.min())
    S *= 255
    img = cv2.applyColorMap(S.astype(np.uint8), cmapy.cmap('magma'))
    height, width, _ = img.shape
    if resz > 0:
        img = cv2.resize(img, (fixed_size, height*resz), interpolation=cv2.INTER_LINEAR)
    img = cv2.flip(img, 0)
    return img





# Example usage

input_folder_path = "/mnt/c/users/oriol/cpia/respirenet/data/ICBHI_dataset"
output_folder_path = "/mnt/c/users/oriol/cpia/respirenet/data/data_output"

cut_audio_for_directory(input_folder_path, output_folder_path)



