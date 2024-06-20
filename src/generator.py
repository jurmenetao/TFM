import numpy as np
import time
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Reshape, Input
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop
from matplotlib import pyplot as plt
import os
from datetime import datetime
import os
from keras.saving import load_model
import random

import pandas as pd



class generator:
  """
  Esta clase permite construir el generador de manera simple.
  """
  def __init__(self, dropout = 0.4, depth = 256, latent_dim = 100, additional_blocks = 1, channels = 3, image_size = 64):
    self.dropout = dropout
    self.depth = depth
    self.latent_dim = latent_dim
    self.additional_blocks = additional_blocks
    self.dim = image_size // 2**int(np.log2(7))
    self.channels = channels
    self.generator_model = self.build_model()


  def build_model(self):
    """
    Construye el generador a partir de los parámetros especificados.
    """
    gen_input = Input(shape = (self.latent_dim))
    gen = Dense(self.dim*self.dim*self.depth)(gen_input)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((self.dim, self.dim, self.depth))(gen)
    gen = Dropout(self.dropout)(gen)

    # Add upsampling blocks
    gen = UpSampling2D()(gen)
    gen = Conv2DTranspose(int(self.depth/2), 5, padding='same')(gen)
    gen = BatchNormalization(momentum=0.9)(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = UpSampling2D()(gen)
    gen = Conv2DTranspose(int(self.depth/4), 5, padding='same')(gen)
    gen = BatchNormalization(momentum=0.9)(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    # Add aditional blocks.
    for block in range(self.additional_blocks):
      gen = Conv2DTranspose(int(self.depth/(4*2**(self.additional_blocks-1))), 5, padding='same')(gen)
      gen = BatchNormalization(momentum=0.9)(gen)
      gen = LeakyReLU(alpha=0.2)(gen)


    gen_output = Conv2D(self.channels, 5, activation='tanh', padding='same')(gen)
    model = Model(gen_input, gen_output)

    return model