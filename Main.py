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

def preprocess_image(image):
    """Preprocesamiento: reduce ruido y detecta bordes"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    return edged, gray


def find_document_contour(edged_image):
    """Encuentra el contorno del documento (4 esquinas)"""
    contours, _ = cv2.findContours(edged_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            return approx
    return None


def correct_perspective(image, doc_contour):
    """Corrige la perspectiva usando homografía"""

    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    pts = doc_contour.reshape(4, 2)
    rect = order_points(pts)

    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped



if __name__ == "__main__":
    # Configuración
    input_path = "foto1.jpg"
    output_path = "documento_escaneado.jpg"

    try:
        # Paso 1: Cargar imagen
        image, image_rgb = load_image(input_path)
        # Paso 2: Preprocesamiento
        edged, gray = preprocess_image(image)

        # Paso 3: Detección de documento
        doc_contour = find_document_contour(edged)

        if doc_contour is not None:
            # Paso 4: Corrección de perspectiva
            warped = correct_perspective(image, doc_contour)

            # Visualización
            plt.figure(figsize=(15, 10))
            plt.subplot(1, 2, 1), plt.imshow(image_rgb), plt.title("Original")
            #plt.subplot(1, 2, 2), plt.imshow(enhanced_bw, cmap='gray'), plt.title("Documento Mejorado")
            plt.show()
        else:
            print("Error: No se detectó un documento válido en la imagen.")


    except Exception as e:
        print(f"Error: {str(e)}")