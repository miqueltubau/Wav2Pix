import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models.discriminator import discriminator
from models.generator import generator
from scripts.utils import Utils, Logger, from_onehot_to_int
from scripts.dataset_builder import dataset_builder, Rescale
from PIL import Image
import os
import numpy as np

class Trainer(object):
    def __init__(self, vis_screen, save_path, l1_coef, l2_coef, pre_trained_gen,
                 pre_trained_disc, batch_size, num_workers, epochs, inference, softmax_coef, image_size, lr_D, lr_G, audio_seconds):

        # initializing the generator and discriminator modules.
        self.generator = generator(image_size, audio_seconds*16000).cuda()
        self.discriminator = discriminator(image_size).cuda()

        # if pre_trained_disc is true, load the already learned parameters. Else, initialize weights randomly
        if pre_trained_disc:
            self.discriminator.load_state_dict(torch.load(pre_trained_disc))
        else:
            self.discriminator.apply(Utils.weights_init)

        # the same as with the discriminator
        if pre_trained_gen:
            self.generator.load_state_dict(torch.load(pre_trained_gen))
        else:
            self.generator.apply(Utils.weights_init)

        # initializing other parameters
        self.inference = inference
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.beta1 = 0.5
        self.num_epochs = epochs
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.softmax_coef = softmax_coef
        self.lr_D = lr_D
        self.lr_G = lr_G


        # building the data_loader
        self.dataset = dataset_builder(transform=Rescale(int(self.image_size)), inference = self.inference, audio_seconds = audio_seconds)
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
                                      num_workers=self.num_workers)

        # defining optimizers. Keeping all the parameters from the list of Module.parameters() for which requires_grad is TRUE
        self.optimD = torch.optim.Adam(filter(lambda p: p.requires_grad, self.discriminator.parameters()), lr=self.lr_D, betas=(self.beta1, 0.999))
        self.optimG = torch.optim.Adam(filter(lambda p: p.requires_grad, self.generator.parameters()), lr=self.lr_G, betas=(self.beta1, 0.999))

        # initializing a Logger in which we will create Log files and defining the directory name in which store checkpoints.
        self.logger = Logger(vis_screen, save_path)
        self.checkpoints_path = 'checkpoints'
        self.save_path = save_path

    def train(self):

        # initializing some loss functions that will be used
        criterion = nn.MSELoss()
        l2_loss = nn.MSELoss()
        l1_loss = nn.L1Loss()

        print('Training...')
        for epoch in range(self.num_epochs):
            for sample in self.data_loader:

                # getting each key value of the sample in question (each sample is a dictionary)
                right_images = sample['face']
                onehot = sample['onehot']
                raw_wav = sample['audio']
                wrong_images = sample['wrong_face']
                id_labels = from_onehot_to_int(onehot) # list with the position of the youtuber which the audio in question belongs


                # defining the inputs as Variables and allocate them into the GPU
                right_images = Variable(right_images.float()).cuda()
                raw_wav = Variable(raw_wav.float()).cuda()
                wrong_images = Variable(wrong_images.float()).cuda()
                onehot = Variable(onehot.float()).cuda()
                id_labels = Variable(id_labels).cuda()


                # tensor of 64 (num of samples per batch) ones and zeros that will be used to compute D loss.
                real_labels = torch.ones(right_images.size(0))
                fake_labels = torch.zeros(right_images.size(0))

                # ======== One sided label smoothing ==========
                # Helps preventing the discriminator from overpowering the
                # generator adding penalty when the discriminator is too confident
                # =============================================
                smoothed_real_labels = torch.FloatTensor(Utils.smooth_label(real_labels.numpy(), -0.1)) # so smooth_real_labels will now be 0.9

                # allocating the three variables into GPU
                real_labels = Variable(real_labels).cuda()
                smoothed_real_labels = Variable(smoothed_real_labels).cuda()
                fake_labels = Variable(fake_labels).cuda()

                # ======= #
                # TRAIN D #
                # ======= #

                # setting all the gradients to 0
                self.discriminator.zero_grad()

                # feeding G only with wav file
                fake_images, z_vector, _ = self.generator(raw_wav)

                # feeding D with the generated images and z vector whose dimensions will be needed
                # for the concatenation in the last hidden layer
                outputs, _ = self.discriminator(fake_images, z_vector)

                # computing D loss when feeding fake images
                fake_score = outputs # log file purposes
                fake_loss = criterion(outputs, fake_labels)

                # feeding D with the real images and z vector again
                outputs, activation_real = self.discriminator(right_images, z_vector)

                # computing D loss when feeding real images
                real_score = outputs
                real_loss = criterion(outputs, smoothed_real_labels)

                # feeding D with real images but not corresponding to the wav under training
                outputs, _ = self.discriminator(wrong_images, z_vector)
                # computing D loss when feeding real images but not the ones corresponding to the input audios
                wrong_loss = criterion(outputs, fake_labels)
                wrong_score = outputs

                # the discriminator loss function is the sum of the three of them
                d_loss = real_loss + fake_loss + wrong_loss

                d_loss.backward()

                self.optimD.step()

                # ======= #
                # TRAIN G #
                # ======= #

                # setting all the gradients to 0
                self.generator.zero_grad()

                # feeding G only with wav file
                fake_images, z_vector, softmax_scores = self.generator(raw_wav)

                # feeding D with the generated images and z vector. Storing intermediate layer activations for loss computation purposes
                outputs, activation_fake = self.discriminator(fake_images, z_vector)

                # feeding D with the real images and z vector.  Storing intermediate layer activations for loss computation purposes
                _, activation_real = self.discriminator(right_images, z_vector)


                activation_fake = torch.mean(activation_fake, 0)
                activation_real = torch.mean(activation_real, 0)

                # ======= Generator Loss function============
                # This is a customized loss function, the first term is the mean square error loss
                # The second term is feature matching loss, this measure the distance between the real and generated
                # images statistics by comparing intermediate layers activations
                # The third term is L1 distance between the generated and real images, this is helpful for the conditional case
                # because it links the embedding feature vector directly to certain pixel values.
                # ===========================================

                # computing first the part of the loss related to the softmax classifier after the embedding
                softmax_criterion = nn.CrossEntropyLoss()
                softmax_loss = softmax_criterion(softmax_scores, id_labels)


                g_loss = criterion(outputs, real_labels) \
                         + self.l2_coef * l2_loss(activation_fake, activation_real.detach()) \
                         + self.l1_coef * l1_loss(fake_images, right_images)\
                         + self.softmax_coef * softmax_loss  # we have seen softmax_loss starts around 2 and g_loss around 20... That's why we've scaled by 10

                # applying backpropagation and updating parameters.
                g_loss.backward()
                self.optimG.step()

            # store the info in the logger at each epoch
            self.logger.log_iteration_gan(epoch, d_loss, g_loss, real_score, fake_score, wrong_score)

            # storing the parameters for every 10 epochs
            if (epoch) % 10 == 0:
                Utils.save_checkpoint(self.discriminator, self.generator, self.checkpoints_path, self.save_path, epoch)

    def predict(self):
        print('Starting inference...')

        starting_id = 0 # this would be the lower_bound id for the next image to be stored

        for id, sample in enumerate(self.data_loader):

            # id is the identifier of the batch. sample is a dictionary of 5 keys.
            right_images = sample['face']
            onehot = sample['onehot']
            raw_wav = sample['audio']
            paths = sample['audio_path']

            # retaining the right youtuber's name from the onehot vector and the id
            token = (onehot == 1).nonzero()[:, 1]
            ids = [path.split('_')[-1][:-4] for path in paths]

            txt = [self.dataset.youtubers[idx] + '_' + str(id) for idx,id in zip(token,ids)]

            if not os.path.exists('results/{0}'.format(self.save_path)):
                os.makedirs('results/{0}'.format(self.save_path))

            # storing raw_wav as a variable into the GPU
            raw_wav = Variable(raw_wav.float()).cuda()

            # feed the audio into the generator
            fake_images, _, _ = self.generator(raw_wav)

            for image, t in zip(fake_images, txt):
                im = image.data.mul_(127.5).add_(127.5).permute(1, 2, 0).cpu().numpy()
                rgb = np.empty((self.image_size, self.image_size, 3), dtype=np.float32)
                # bgr --> rgb
                rgb[:,:,0] = im[:,:,2]
                rgb[:,:,1] = im[:,:,1]
                rgb[:,:,2] = im[:,:,0]
                im = Image.fromarray(rgb.astype('uint8'))
                im.save('results/{0}/{1}.jpg'.format(self.save_path, t.replace("/", "")[:100]))







