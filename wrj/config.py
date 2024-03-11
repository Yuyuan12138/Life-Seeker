import torch
import argparse


class Config():
    d = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device=d)

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-data', default='4mC_C.equisetifolia')
        parser.add_argument('-epochs', default=400)
        parser.add_argument('-bsize', default=512)
        parser.add_argument('-lr', default=1e-3)
        parser.add_argument('-loop', default=114514)
        args = parser.parse_args()
        self.epochs = args.epochs
        self.lr = args.lr
        self.batch_size = args.bsize
        self.data_name = args.data
        self.data_path = '../data/DNA/iDNA_ABF/tsv/4mC/'
        self.loop = args.loop
