from scripts.trainer import Trainer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--l1_coef", default=50, type=float)
parser.add_argument("--l2_coef", default=100, type=float)
parser.add_argument("--vis_screen", default='gan')
parser.add_argument("--save_path", default='')
parser.add_argument("--inference", default=False, action='store_true')
parser.add_argument('--pre_trained_disc', default=None)
parser.add_argument('--pre_trained_gen', default=None)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--epochs', default=201, type=int)
parser.add_argument('--softmax_coef', default = 20, type=float, help = 'Regularization parameter for the softmax loss')
parser.add_argument('--image_size', default = 128, type = int)
parser.add_argument('--lr_D', default = 0.0004, type=float, help = 'learning rate of the disciminator')
parser.add_argument('--lr_G', default = 0.0001, type=float, help = 'learning rate of the generator')
parser.add_argument('--audio_seconds', default = 1, type=float, help='desired audio duration to fed the network with')


args = parser.parse_args()

trainer = Trainer(vis_screen=args.vis_screen,
                  save_path=args.save_path,
                  l1_coef=args.l1_coef,
                  l2_coef=args.l2_coef,
                  pre_trained_disc=args.pre_trained_disc,
                  pre_trained_gen=args.pre_trained_gen,
                  batch_size=args.batch_size,
                  num_workers=args.num_workers,
                  epochs=args.epochs,
                  inference = args.inference,
                  softmax_coef = args.softmax_coef,
                  image_size = args.image_size,
                  lr_D = args.lr_D,
                  lr_G = args.lr_G,
                  audio_seconds = args.audio_seconds
                  )

# TRAINING OR PREDICTING #
if not args.inference:
    trainer.train()
else:
    trainer.predict()

