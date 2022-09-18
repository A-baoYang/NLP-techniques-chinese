import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from train import train
from hparams import hparams

here = os.path.dirname(os.path.abspath(__file__))


def main():
    train(hparams)


if __name__ == '__main__':
    main()
