import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure

# Cargar la imagen
def load_image(image_path):
    """Carga la imagen y la convierte a RGB para visualización"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("No se pudo cargar la imagen. Verifica la ruta.")
    return image, cv2.cvtColor(image, cv2.COLOR_BGR2RGB)




if __name__ == "__main__":
    # Configuración
    input_path = "foto1.jpg"
    output_path = "documento_escaneado.jpg"

    try:
        # Paso 1: Cargar imagen
        image, image_rgb = load_image(input_path)

        # Visualización
        plt.figure(figsize=(15, 10))
        plt.subplot(1, 2, 1), plt.imshow(image_rgb), plt.title("Original")
        plt.show()

    except Exception as e:
        print(f"Error: {str(e)}")