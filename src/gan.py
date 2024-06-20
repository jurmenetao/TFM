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




class GAN:

  def __init__(self, generator, discriminator, path_resultados, data_source = "Datos/data_npz_64x64_compressed_todas/todas.npz", epochs = 100, latent_dim = 100, sample_interval = 5, batch_size = 128, learning_rate=0.0002, beta_1=0.5):
    """
    Esta clase permite construir la estructura de la red adversarial y entrenarla. 
    """
    
    self.data = np.load(data_source)
    self.data = self.data["img"]
    self.num_imagenes = len(self.data)
    print("Se han cargado "+ str(self.num_imagenes)+ " imagenes.")

    self.generator = generator
    self.discriminator = discriminator
    # self.data_dir = data_dir
    self.epochs = epochs
    self.sample_interval = sample_interval
    self.path_resultados = path_resultados
    self.batch_size = batch_size
    self.latent_dim = latent_dim
    self.evolution = []
    self.training_time = 0
    self.learning_rate = learning_rate
    self.beta_1 = beta_1
    # self.imagenes_disponibles = os.listdir(self.data_dir)

    self.opt = Adam(learning_rate= self.learning_rate,
                    beta_1= self.beta_1)
    self.gan = self.build_gan()
    self.discriminator.compile(loss=['binary_crossentropy'],
                    optimizer=self.opt,
                    metrics=['accuracy'])
    self.verificar_y_crear_carpeta()

  def build_gan(self):
    """
    Construye la estructura de la red adversarial.
    """
    model = Sequential()
    model.add(self.generator)
    self.discriminator.trainable = False
    model.add(self.discriminator)
    self.discriminator.trainable = True
    model.compile(loss='binary_crossentropy',
                  optimizer = Adam(learning_rate=0.0001, beta_1=0.5),
                  metrics=['accuracy'])
    return model

  def verificar_y_crear_carpeta(self):
    """
    Verifica si una carpeta existe en la ruta especificada y la crea si no existe.
    """
    if not os.path.exists(self.path_resultados):
        try:
            os.makedirs(self.path_resultados)
            print(f"Carpeta '{self.path_resultados}' creada exitosamente.")
        except OSError as e:
            print(f"Error al crear la carpeta '{self.path_resultados}': {e}")
    else:
        print(f"La carpeta '{self.path_resultados}' ya existe.")


  def sample_images(self, epoch, save_models = False):
      """
      Genera una figura con resultados de imágenes generadas por la red en el paso actual.
      """
      r, c = 2, 5
      noise = np.random.normal(0, 1, (r * c, 100))
      #sampled_labels = np.arange(0, 10)#.reshape(-1, 1)
      gen_imgs = self.generator.predict(noise)

      # Rescale images 0 - 1
      gen_imgs = 0.5 * gen_imgs + 0.5

      fig, axs = plt.subplots(r, c)
      cnt = 0
      for i in range(r):
          for j in range(c):
              axs[i,j].imshow(gen_imgs[cnt,:,:,:])
              axs[i,j].axis('off')
              cnt += 1
      fig.suptitle(str(epoch))
      fig.savefig(self.path_resultados+"/%d.png" % epoch)
      plt.close()

      if save_models:
        self.generator.save(self.path_resultados + "/generator_"+str(epoch)+".h5")
        self.discriminator.save(self.path_resultados + "/discriminator_"+str(epoch)+".h5")

  def sample_real_images(self):
    """
    Realiza un muestreo y devuelve un conjunto de imágenes reales.
    """
    return self.data[random.sample(range(self.num_imagenes), self.batch_size),:,:,:]

  def train(self, save_models = False, save_model_interval = 100):
    """
    Ejecuta el proceso de entrenamiento de la red adversarial.
    """
    start_time = time.time()
    try:
      real_labels = np.zeros([self.batch_size, 1])
      false_labels = np.ones([self.batch_size, 1])
      time_contador = time.time()
      for epoch in range(self.epochs):
        # idx = np.random.randint(0, self.data.shape[0], self.batch_size)
        # imgs = self.data[idx]

        imgs = self.sample_real_images()
        print(str(time.time()-time_contador))
        time_contador = time.time()
        noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))

        gen_imgs = self.generator.predict(noise)

        randomness_labels = 0.005 * np.random.uniform(size=[self.batch_size, 1])
        d_loss_real = self.discriminator.train_on_batch(imgs,real_labels+randomness_labels)
        d_loss_fake = self.discriminator.train_on_batch(gen_imgs,false_labels+randomness_labels)

        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        d_acc_reals = self.discriminator.evaluate(imgs, real_labels)[1]
        d_acc_fakes = self.discriminator.evaluate(gen_imgs, false_labels)[1]

        g_loss = self.gan.train_on_batch(noise, real_labels)
        self.evolution.append([d_loss_real[0], d_loss_fake[0], d_acc_reals, d_acc_fakes, g_loss[0], g_loss[1]])
        print ("%d [D loss: %f, reals_acc.: %.2f%%, false_acc.:%.2f%%] [G loss: %f - %.2f%%]" % (epoch, d_loss[0], 100*d_acc_reals, 100*d_acc_fakes, g_loss[0], g_loss[1]))
        print(str(time.time()-time_contador))
        time_contador = time.time()
        if epoch % self.sample_interval == 0:
          self.sample_images(epoch, save_models=False)
        if save_models and epoch % save_model_interval ==0:
          self.generator.save(self.path_resultados + "/generator_"+str(epoch)+".h5")
          self.discriminator.save(self.path_resultados + "/discriminator_"+str(epoch)+".h5")
      self.training_time = time.time()-start_time
      self.finish_training(epoch)

    except Exception as e:
      self.training_time = time.time()-start_time
      self.finish_training(epoch)
      print("Error: \n" + str(e))




  def finish_training(self, epoch):
    """
    Guarda los datos del entrenamiento al finalizar el entrenamiento.
    """

    evolution = pd.DataFrame(self.evolution)
    evolution.columns = ["d_loss_real", "d_loss_fake", "d_acc_reals", "d_acc_fakes", "g_loss", "g_acc"]

    evolution.to_csv(self.path_resultados + "/evolucion.csv")
    # Descomentar para guardar información sobre el uso de GPUs
    # gpu_info = !nvidia-smi 
    # pd.DataFrame(gpu_info).to_csv(self.path_resultados + "/info_gpu.csv") 

    pd.DataFrame.from_dict({"training_time": [self.training_time],
                            "batch_size" : [self.batch_size],
                            "epochs" : [self.epochs],
                            "learning_rate" : [self.learning_rate],
                            "beta_1" : [self.beta_1]})\
                .to_csv(self.path_resultados + "/training_batchsize_learningrate_beta.csv")

    self.generator.save(self.path_resultados + "/generator_"+str(epoch)+"_final.h5")
    self.discriminator.save(self.path_resultados + "/discriminator_"+str(epoch)+"_final.h5")
