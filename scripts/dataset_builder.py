from torch.utils.data import Dataset, DataLoader
import cPickle as pickle
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import yaml
import scipy.io.wavfile as wavfile
import string
import random
import unicodedata

printable = set(string.printable)

class dataset_builder(Dataset):

    def __init__(self, transform = None, inference = False, audio_seconds = 1):

        # opening the file that contains the path with the images/audios.
        with open('config.yaml', 'r') as file:
            config = yaml.load(file)

        # selecting the paths for the images/audios we will work with according to whether we are training or testing
        if not inference:
            self.faces = pickle.load(open(config['train_faces_path'], 'rb'))
            self.audios = pickle.load(open(config['train_audios_path'], 'rb'))

            print 'Total amount of training samples: {0} faces | {1} audios '.format(len(self.faces),len(self.audios))
            print
        else:
            self.faces = pickle.load(open(config['inference_faces_path'], 'rb'))
            self.audios = pickle.load(open(config['inference_audios_path'], 'rb'))

            print 'Total amount of inference samples: {0} faces | {1} audios '.format(len(self.faces),len(self.audios))
            print

        # initializing some useful variables
        self.audio_samples = audio_seconds * 16000 #desired audio_samples to work with for each .wav file
        self.transform = transform
        self.inference = inference
        self.youtubers = list(set([path.split('/')[-2] for path in self.faces]))  # be carefull, this is hardcoded. Each youtuber has its own directory with the images/audios inside
        self.num_youtubers = len(self.youtubers)


    def __len__(self):
        return len(self.faces)

    def __getitem__(self, idx):

        # getting image paths. # in practice, when inference that won't be a path but a 0
        face_path = self.faces[idx]
        format_path = self.format_filename(face_path)
        format_path = format_path.replace(" ", "") \
            .replace("'", "").replace('"', '').replace('(', '').replace(')', '').replace('#', '') \
            .replace('&', '').replace(';', '').replace('!', '').replace(',', '').replace('$', '').replace('?', '')

        # getting audio paths.
        audio_path = self.audios[idx]
        fm, wav_data = wavfile.read(filter(lambda x: x in printable, audio_path))
        if fm != 16000:
            raise ValueError('Sampling rate is expected to be 16kHz!')

        if self.audio_samples > len(wav_data):
            raise ValueError('Desired audio length is larger than the .wav file duration')
        elif self.audio_samples < len(wav_data):
            # cut the audio
            wav_data = self.cut_audio(wav_data)

        # applying some processing to the audio
        wav_data = self.abs_normalize_wave_minmax(wav_data)
        wav_data = self.pre_emphasize(wav_data)

        # opening the corresponding image and a wrong_face
        cropped_face = Image.open(filter(lambda x: x in printable, format_path).replace('.jpg', '.png'))
        wrong_face_path = self.get_dismatched_face(audio_path)
        wrong_face = Image.open(wrong_face_path)

        # storing youtuber identity in a onehot vector. It will be usefull, for example, for softmax the loss computation
        youtuber = face_path.split('/')[-2]
        onehot = self.youtubers.index(youtuber)

        sample = {'onehot': self.to_categorical(onehot), 'wrong_face':np.array(wrong_face),
                  'face':np.array(cropped_face), 'audio': wav_data, 'audio_path':audio_path}

        # rescale the sample if asked
        if self.transform:
            sample = self.transform(sample)

        # normalizing and standardizing the images
        sample['face'] = sample['face'].sub_(127.5).div_(127.5)
        sample['wrong_face'] = sample['wrong_face'].sub_(127.5).div_(127.5)

        return sample


    def cut_audio(self, wav_data):
        # initially len(wav_data)=64000
        samples_to_retain = self.audio_samples

        # computing samples to remove
        samples_to_remove = len(wav_data) - samples_to_retain

        return wav_data[int(samples_to_remove/2):-int(samples_to_remove/2)]
    def to_categorical(self, token):
        """ 1-hot encodes a tensor """
        return np.eye(self.num_youtubers, dtype='uint8')[token]

    def format_filename(self, filename):
        try:
            filename = filename.decode('utf-8')
            s = ''.join((c for c in unicodedata.normalize('NFD', unicode(filename)) if unicodedata.category(c) != 'Mn'))
            return s.decode()
        except (UnicodeEncodeError, UnicodeDecodeError):
            return filename

    def abs_normalize_wave_minmax(self, wavdata):
        x = wavdata.astype(np.int32)
        imax = np.max(np.abs(x))
        x_n = x / imax # tira molts warnings!
        return x_n

    def pre_emphasize(self, x, coef=0.95):
        if coef <= 0:
            return x
        x0 = np.reshape(x[0], (1,))
        diff = x[1:] - coef * x[:-1]
        concat = np.concatenate((x0, diff), axis=0)
        return concat

    def get_dismatched_face(self,audio_path):
        selected_face = random.choice(self.faces)
        if selected_face.split('/')[-2] == audio_path.split('/')[-2]:
            # if randomly we have chosen the right face, let's choose another one
            selected_face = self.get_dismatched_face(audio_path)
        return selected_face


class Rescale(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If int, output generated is an
        square image (output_size, output_size, channels). If tuple, output matches with
        output_size (output_size[0], output_size[1], channels).
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        onehot, face, audio, p, wf = sample['onehot'], sample['face'], sample['audio'], \
                                    sample['audio_path'], sample['wrong_face']


        img = transforms.ToPILImage()(face)
        wrong_img = transforms.ToPILImage()(wf)

        img = transforms.Resize((self.output_size, self.output_size))(img)
        wrong_img = transforms.Resize((self.output_size, self.output_size))(wrong_img)

        img = np.array(img, dtype=float)
        wrong_img = np.array(wrong_img, dtype=float)


        # (height, width, channels) -->> (channels, height, width)
        img = img.transpose(2, 0, 1)
        wrong_img = wrong_img.transpose(2, 0, 1)

        return {'onehot': torch.from_numpy(onehot).float(),
                'face': torch.from_numpy(np.array(img)).float(),
                'audio': torch.from_numpy(audio).float(), 'audio_path': p,
                'wrong_face': torch.from_numpy(wrong_img).float()}

