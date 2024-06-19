# TFM: Aumentación de datos para imágenes: de las transformaciones geométricas a la implementación de un modelo generativo

Este repositorio contiene las bases del código y los resultados del desarrollo del caso de estudio para el trabajo final del Máster en Ciencia de Datos de la Universitat Oberta de Catalunya.

## Contenidos del repositorio

- Datos/data_npz_64x64_compressed_todas/todas.npz: Conjunto de imágnes pretratadas, normalizadas entre -1 y 1, y en formato preparado para consultarlas de manera eficiente desde la clase del generador.
- src:
    - discriminator.py: Script con la clase para generar discriminadores con distinto número de bloques convolucionales.
    - generator.py: Script con la clase para generar generadores con distinto número de bloques convolucionales.
    - gan.py: Script con la clase GAN, que se encarga de crear el marco adversarial entre generador y discriminador y entrenarlos, así como de generar la carpeta de resultados y guardar las imágenes.




![generador 1 drawio](https://github.com/jurmenetao/TFM/assets/97030334/f9b8339e-8bef-4423-9127-40ce2716c1c1)
![generador 2 drawio](https://github.com/jurmenetao/TFM/assets/97030334/9d79980a-9fbd-4618-af1e-acf13ce79e3b)
![discriminador 1 drawio](https://github.com/jurmenetao/TFM/assets/97030334/25377d41-f0cc-4a3f-b522-428b0198221f)
![discriminador2 drawio](https://github.com/jurmenetao/TFM/assets/97030334/65ad68c7-da98-4fb9-98ef-8b6f39a51b78)