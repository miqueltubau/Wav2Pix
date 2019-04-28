import os
import argparse
import cPickle as pickle
import string,random

printable = set(string.printable)

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_path' , required=True, help='path where the dataset is stored')
parser.add_argument('--pickle_path', required=True, help='folder where to store the .pckl file')
parser.add_argument('--save_name', required=True, help='name of the pickle file')

args = parser.parse_args()

# obtaining the list of youtubers under study
youtubers = os.listdir(args.dataset_path)

# creating the pickles
if not os.path.exists(args.pickle_path): 
    os.mkdir(args.pickle_path)
    os.mkdir(args.pickle_path+'/faces')
    os.mkdir(args.pickle_path+'/audios')

face_file = open(os.path.join(args.pickle_path,'faces/'+args.save_name)+'.pkl', 'wb')
audio_file = open(os.path.join(args.pickle_path,'audios/'+args.save_name)+'.pkl', 'wb')

total_faces = []
total_audios = []
for youtuber in youtubers:
    print('storing paths from {0}'.format(youtuber))
    working_path = os.path.join(args.dataset_path,youtuber)

    # storing faces and audios in lists
    faces = [os.path.join(working_path,face) for face in os.listdir(working_path) if face.endswith('.png')]
    audios = [os.path.join(working_path,audio) for audio in os.listdir(working_path) if audio.endswith('.wav')]

    # as there are some .wav files damaged in loics directory, I should skip that audios and do not store them
    for face in  faces:
        audio = face.replace('cropped_face', 'preprocessed').replace('.png', '.wav').replace('.jpg', '.wav')
        # adding the face and its corresponding audio
        total_faces.append(face)
        total_audios.append(audio)

# storing both lists in the corresponding file
pickle.dump(total_faces, face_file)
pickle.dump(total_audios,audio_file)

face_file.close()
audio_file.close()
