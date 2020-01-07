import torch
import torch.nn as nn
from Generators import *
from Discriminators import *
from Classifier import *


def data_path_length(start, end, G: Generator, D: Discriminator, classifier: Classifier):
    pass


def perceptual_path_length(start, end, G: Generator, D: Discriminator, inception_model):
    pass


def intra_data_path_length(start, end, G: Generator, D: Discriminator, classifier: Classifier):
    pass


def intra_perceptual_path_length(start, end, G: Generator, D: Discriminator, inception_model):
    pass
