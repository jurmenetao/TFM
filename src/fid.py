import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize
from keras.datasets import cifar10
from keras.saving import load_model
import random
import os



def generate_images(carpeta, noise = np.random.normal(0, 1, (10000, 100))):
  """
  Recupera o genera conjuntos de imágenes para la evaluación cuantitativa.
  """

  if not os.path.exists("Resultados/"+carpeta +"/imagenes FID.npz"):
      try:
          # os.makedirs("Resultados/"+carpeta +"/imagenes FID")
          # print(f"Carpeta Resultados/"+carpeta +"/imagenes FID creada exitosamente.")
          try:
            modelo_recuperado = load_model( "Resultados/" + carpeta + "/generator_499_final.h5")
          except:
            pass
          try:
            modelo_recuperado = load_model( "Resultados/" + carpeta + "/generator_999_final.h5")
          except:
            pass
          try:
            modelo_recuperado = load_model( "Resultados/" + carpeta + "/generator_14999_final.h5")
          except:
            pass
          gen_imgs = modelo_recuperado.predict(noise)
          np.savez_compressed(os.path.join("Resultados/"+carpeta +"/imagenes FID.npz"), img=gen_imgs)
          print("Imagenes creadas y guardadas")
      except OSError as e:
          print(f"Error al crear la carpeta Rsultados/"+carpeta +"/imagenes FID: {e}")
  else:
      print(f" Resultados/"+carpeta +"/imagenes FID.npz ya existe.")
      gen_imgs = np.load("Resultados/"+carpeta +"/imagenes FID.npz")
      gen_imgs = gen_imgs["img"]

  return gen_imgs


def scale_images(images, new_shape):
 """
 Escala las imágenes para adecuarlas al tamaño de entrada del modelo Inception V3. 
 """
 images_list = list()
 for image in images:
    # resize with nearest neighbor interpolation
    new_image = resize(image, new_shape, 0)
    # store
    images_list.append(new_image)
 return asarray(images_list)


def sample_real_images(data, sample_n=10000):
  """
  Realiza un muestreo de imágenes reales.
  """
  return data[random.sample(range(len(data)), sample_n),:,:,:]



# Para la implementación de este apartado se ha empleado 
# https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
# calculate frechet inception distance
def calculate_fid(model, images1, images2):
 """
 Calcula el valor del FID, dados los conjuntos de imágenes y el modelo Inception-V3.
 """
 # calculate activations
 act1 = model.predict(images1)
 act2 = model.predict(images2)
 # calculate mean and covariance statistics
 mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
 mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
 # calculate sum squared difference between means
 ssdiff = np.sum((mu1 - mu2)**2.0)
 # calculate sqrt of product between cov
 covmean = sqrtm(sigma1.dot(sigma2))
 # check and correct imaginary numbers from sqrt
 if iscomplexobj(covmean):
  covmean = covmean.real
 # calculate score
 fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
 return fid