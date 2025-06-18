import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure

# Cargar la imagen
image = cv2.imread('foto1.jpg')

# Convertir de BGR a RGB para visualización
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Mostrar la imagen original
plt.figure(figsize=(10, 10))
plt.imshow(image_rgb)
plt.title('Imagen Original')
plt.axis('off')
plt.show()