import numpy as np
import time
from keras.datasets import mnist
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


class discriminator:
  def __init__(self, depth = 32, dropout = 0.4, img_rows = 64, img_cols = 64, img_channels =3, additional_layers= 3):
      """
      Esta clase permite construir discriminadores de manera simple.
      """

      self.depth = depth
      self.dropout = dropout
      self.input_shape = (img_rows, img_cols, img_channels)
      self.additional_layers = additional_layers
      self.discriminator_model = self.build_discriminator()


  def build_discriminator(self):
      """
      Construye el discriminador de acuerdo con las características especificadas.
      """
      
      dis_input = Input(shape = self.input_shape)

      dis = Conv2D(self.depth, 5, strides=2, padding='same')(dis_input)
      dis = LeakyReLU(alpha = 0.2)(dis)
      dis =  Dropout(self.dropout)(dis)

      for i in range(1,self.additional_layers):
          dis = Conv2D(self.depth*2**i, 5, strides=2, padding='same')(dis)
          dis = LeakyReLU(alpha = 0.2)(dis)
          dis =  Dropout(self.dropout)(dis)

      dis = Flatten()(dis)
      dis = Dropout(0.4)(dis)
      dis_output = Dense(1, activation='sigmoid')(dis)

      model = Model(dis_input, dis_output)

      return model