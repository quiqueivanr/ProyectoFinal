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
    """Preprocesamiento mejorado"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Ecualización de histograma para mejorar contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(blurred)
    # Detección de bordes con parámetros adaptativos
    v = np.median(equalized)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    edged = cv2.Canny(equalized, lower, upper)
    # Operación de cierre para conectar bordes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    return edged, gray


def find_document_contour(edged_image, min_area_ratio, epsilon_ratio):
    """Encuentra el contorno del documento (4 esquinas)"""

    # 1. Obtener contorno
    contours, _ = cv2.findContours(edged_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("❌ Error: No se detectaron contornos - Revisa el preprocesamiento")
        print("Solución: Ajusta los parámetros de cv2.Canny() o mejora la iluminación")
        return None

    # 2. Filtrar por área mínima
    height, width = edged_image.shape
    image_area = height * width
    min_area = image_area * min_area_ratio
    max_area = image_area * 0.9  # Máximo 90% del área


    print(f"\nℹ️ Análisis de contornos:")
    print(f"- Tamaño imagen: {width}x{height} ({image_area} px)")
    print(f"- Área mínima requerida: {min_area:.0f} px")

    # 3. Filtrado mejorado
    valid_contours = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        print(f"Contorno {i + 1}: {area:.0f} px ({(area / image_area) * 100:.1f}%)")

        if min_area <= area <= max_area:
            valid_contours.append(contour)

    if not valid_contours:
        largest_area = max(cv2.contourArea(c) for c in contours)
        print(f"\n⚠️ Todos los contornos muy pequeños (Mayor encontrado: {largest_area:.0f} px)")
        print(f"Recomendaciones:")
        print(f"- Reduce min_area_ratio (actual: {min_area_ratio})")
        print(f"- Acerca más el documento a la cámara")
        print(f"- Usa cv2.Canny() con umbrales más bajos (ej: 20, 60)")
        return None

    # 4. Ordenar y analizar top 3 contornos
    valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:3]

    for i, contour in enumerate(valid_contours):
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon_ratio * peri, True)

        print(f"\n🔍 Analizando contorno #{i + 1}:")
        print(f"- Lados: {len(approx)}")
        print(f"- Área: {cv2.contourArea(contour):.0f} px")

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            convex = cv2.isContourConvex(approx)
            solidity = cv2.contourArea(contour) / float(w * h)

            print(f"- Aspect Ratio: {aspect_ratio:.2f}")
            print(f"- Convexo: {'Sí' if convex else 'No'}")
            print(f"- Solidez: {solidity:.2f}")

            # Criterios más flexibles
            if 0.5 < aspect_ratio < 2.0 and convex and 0.6 < solidity < 1.3:
                print("¡Documento detectado correctamente!")
                return approx
            else:
                print("❌ No cumple criterios geométricos")
        else:
            print(f"❌ Tiene {len(approx)} lados (se requieren 4)")

    print("\n⚠️ Ningún contorno cumplió todos los requisitos")
    print("Posibles soluciones:")
    print("- Aumenta epsilon_ratio (actual: {epsilon_ratio})")
    print("- Verifica que el documento esté plano y bien visible")
    print("- Prueba con otra iluminación o fondo uniforme")
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

def enhance_document(warped_image):
    """Mejora el contraste y binariza el documento"""
    gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return thresh


def color_enhancement(warped_image):
    """Mejora el color y nitidez (opcional para documentos a color)"""
    lab = cv2.cvtColor(warped_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return enhanced


def save_results(enhanced_image, output_path):
    """Guarda la imagen procesada"""
    cv2.imwrite(output_path, enhanced_image)




def apply_color_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Rangos para fondos blancos/grises (ajusta según tu caso)
    lower = np.array([0, 0, 150], dtype="uint8")  # Tonos claros
    upper = np.array([180, 30, 255], dtype="uint8")
    mask = cv2.inRange(hsv, lower, upper)
    masked = cv2.bitwise_and(image, image, mask=mask)
    return masked



if __name__ == "__main__":
    # Configuración
    input_path = "foto2.jpg"
    output_path_bw = "documento_bw.jpg"
    output_path_color = "documento_color.jpg"

    try:
        # Paso 1: Cargar imagen
        image, image_rgb = load_image(input_path)

        # Paso extra: Máscara de color
        #masked = apply_color_mask(image)
        #edged, _ = preprocess_image(masked if masked is not None else image)

        # Paso 2: Preprocesamiento
        edged, gray = preprocess_image(image)

        # Paso 3: Detección de documento
        doc_contour = find_document_contour(edged, min_area_ratio=0.02, epsilon_ratio=0.05 )

        if doc_contour is not None:
            # Paso 4: Corrección de perspectiva
            warped = correct_perspective(image, doc_contour)
            # Muestra la imagen corregida antes de mejorarla
            cv2.imshow("Perspectiva Corregida", warped)
            cv2.waitKey(2000)  # Muestra por 2 segundos
            cv2.destroyAllWindows()

            # Paso 5: Mejora (blanco y negro)
            enhanced_bw = enhance_document(warped)

            # Paso 6 (Opcional): Mejora de color
            enhanced_color = color_enhancement(warped)

            # Guardar ambos resultados
            save_results(enhanced_bw, output_path_bw)
            save_results(enhanced_color, output_path_color)

            # Visualización
            plt.figure(figsize=(15, 10))
            plt.subplot(1, 3, 1), plt.imshow(image_rgb), plt.title("Original")
            plt.subplot(1, 3, 2), plt.imshow(enhanced_bw, cmap='gray'), plt.title("Documento BN")
            plt.subplot(1, 3, 3), plt.imshow(cv2.cvtColor(enhanced_color, cv2.COLOR_BGR2RGB)), plt.title("Documento Color")
            plt.show()
            print("✅ ¡Documento escaneado correctamente!")

        else:
            print("Error: No se detectó un documento válido en la imagen.")

    except Exception as e:
        print(f"Error: {str(e)}")