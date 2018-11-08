# SPEECH-CONDITIONED FACE GENERATION USING GENERATIVE ADVERSARIAL NETWORKS

## Intoduction
Speech is a rich biometric signal that contains information about the identity, gender and emotional state of the speaker. In this work, we explore its potential to generate face images of a speaker by conditioning a Generative Adversarial Network (GAN) with raw speech input. We propose a deep neural network that is trained from scratch in an end-to-end fashion, generating a face directly from the raw speech waveform without any additional identity information (e.g reference image or one-hot encoding). Our model is trained in a self-supervised fashion by exploiting the audio and visual signals naturally aligned in videos. With the purpose of training from video data, we present a novel dataset collected for this work, with high-quality videos of ten youtubers with notable expressiveness in both the speech and visual signals.

We used [this](https://github.com/franroldans/tfm-franroldan-wav2pix) project as baseline.

<figure><img src=''></figure>

## Dependencies

- Python 2.7
- PyTorch 

This implementation only supports running with GPUs.

## Usage
### Training

`python runtime.py`

**Arguments:**
- `lr_D` : The learning rate of the disciminator. default = `0.0004`
- `lr_G` : The learning rate of the generator. default = `0.0001`
- `vis_screen` : The visdom env name for visualization. default = `gan`
- `save_path` : Name of the directory (inside **checkpoints**) where the parameters of them odel will be stored.
- `l1_coef` : L1 loss coefficient in the generator loss fucntion. default=`50`
- `l2_coef` : Feature matching coefficient in the generator loss fucntion. default=`100`
- `pre_trained_disc` : Discriminator pre-tranined model path used for intializing training.
- `pre_trained_gen` : Generator pre-tranined model path used for intializing training.
- `batch_size` : Batch size. default= `64`
- `num_workers`: Number of dataloader workers used for fetching data. default = `8`
- `epochs` : Number of training epochs. default=`200`
- `softmax_coef`: Paramete for the scale of the loss of the classifier on top of the embedding
- `image_size` : Number of pixels per dimension. They are assumed to be squared. Two possible values: `64 | 128`. default = `64`
- `inference` : Boolean for choosing whether train or test. default = `False`

## References

