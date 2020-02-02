import os
import argparse
import string, random
from shutil import copyfile
import scipy.io.wavfile as wavfile
from wav_functions import readwav

printable = set(string.printable)


def augment_data(wav_data, factor, audio_seconds):
    # get different audios until we reach self.data_augmentation
    audios = []
    centers = []
    for id in range(0,factor):
        # selecting a center randomly but taking into consideration there needs to be data enough "in the left" and "in the right"
        samples_to_retain = audio_seconds * 16000
        valid_centers = range(0,len(wav_data))[samples_to_retain/2:-samples_to_retain/2]
        center = random.choice(valid_centers)

        # if the center is already taken, let's find another one.
        while center in centers: center = random.choice(valid_centers)

        centers.append(center)
        audios.append(wav_data[center-samples_to_retain/2:center+samples_to_retain/2])

    return audios

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--folder_path", required = True, help='path of the folder where to take the data and augment')
    parser.add_argument("--factor", required = True, help = 'data augmentation factor')
    parser.add_argument('--output_folder', required = True, help = 'folder where to store the augmented data')
    parser.add_argument('--audio_seconds', required = True, help='audio of the resulting augmented .wav files')

    args = parser.parse_args()

    youtubers = [youtuber for youtuber in os.listdir(args.folder_path) if not youtuber.startswith('.')]

    id_number = 0

    for youtuber in youtubers:

        working_path = os.path.join(args.folder_path,youtuber)

        # creating if necessary a folder with the youtuber name in the output_folder
        if not os.path.exists(os.path.join(args.output_folder,youtuber)):
            os.makedirs(os.path.join(args.output_folder,youtuber))

        audios = [audio for audio in os.listdir(working_path) if audio.endswith('.wav')]

        for audio in audios:

            number = audio.split('_')[-1][:-4]

            # getting the paths of the original audio and face
            audio_path = os.path.join(working_path,audio)
            corresponding_face = os.path.join(working_path,'cropped_face_frame_'+number+'.png')

            # reading the audio file and converting it into an array
            fm, _, wav_data = readwav(audio_path)

            if fm != 16000:
                raise ValueError('Sampling rate is expected to be 16 KHz!')

            if len(wav_data) < 16000 * int(args.audio_seconds):
                raise ValueError('The original audio is shorter than the desired output')

            # obtaining the audios as a result of the data augmentation
            wav_vectors = augment_data(wav_data, int(args.factor), int(args.audio_seconds))

            for wav_vec in wav_vectors:

                new_path = os.path.join(args.output_folder,youtuber)

                # converting wav_vec into a .wav file and storing it
                wav_file = wavfile.write(filename=os.path.join(new_path,'preprocessed_frame_'+str(id_number)+'.wav'), rate=16000, data=wav_vec)
                new_face_path = os.path.join(new_path,'cropped_face_frame_'+str(id_number)+'.png')


                # copying the faces from the older path to the newer

                copyfile(corresponding_face, new_face_path)

                # updating id_number to avoid overwritting files
                id_number += 1
