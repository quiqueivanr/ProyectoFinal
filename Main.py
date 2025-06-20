import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure


# Cargar la imagen
def cargar_imagen(image_path):
    """Carga la imagen y la convierte a RGB para visualización"""
    imagen = cv2.imread(image_path)
    if imagen is None:
        raise ValueError("No se pudo cargar la imagen. Verifica la ruta.")
    return imagen, cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)


def Preprocesar_Imagen(imagen):
    """Preprocesamiento mejorado con control de iluminación"""
    # Convertir a escala de grises
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Corrección gamma para mejorar contraste
    gamma = 1.5 if np.mean(gray) < 100 else 0.8
    gray = np.power(gray / 255.0, gamma) * 255.0
    gray = gray.astype(np.uint8)

    # Reducción de ruido adaptativo
    blur_size = 5 if imagen.shape[1] > 1000 else 3
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

    # Ecualización CLAHE para mejorar contraste local
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    equalized = clahe.apply(blurred)

    # Detección de bordes con parámetros adaptativos
    v = np.median(equalized)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(equalized, lower, upper)

    # Operación morfológica para conectar bordes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)

    return edged, gray


def encontrar_contorno(edged_image, min_area_ratio=0.02, epsilon_ratio=0.02):
    """Encuentra el contorno del documento con criterios mejorados"""
    # Encontrar contornos
    contours, _ = cv2.findContours(edged_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("❌ Error: No se detectaron contornos")
        return None

    # Calcular área de la imagen
    height, width = edged_image.shape
    image_area = height * width
    min_area = image_area * min_area_ratio
    max_area = image_area * 0.95  # Más flexible que 0.9

    # Filtrar contornos por área
    valid_contours = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]

    if not valid_contours:
        print(f"⚠️ No hay contornos en el rango de área {min_area:.0f}-{max_area:.0f} px")
        # Usar el contorno más grande como fallback
        valid_contours = [max(contours, key=cv2.contourArea)]

    # Ordenar por área descendente
    valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:3]

    # Evaluar los mejores contornos
    for i, contour in enumerate(valid_contours):
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon_ratio * peri, True)

        # Criterios más flexibles para formas cuadriláteras
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            convex = cv2.isContourConvex(approx)
            solidity = cv2.contourArea(contour) / float(w * h)

            # Rangos más amplios para aceptar documentos
            if 0.3 < aspect_ratio < 3.0 and solidity > 0.5:
                print(f"✅ Documento detectado (Contorno {i + 1})")
                print(f"- Lados: {len(approx)}")
                print(f"- Área: {cv2.contourArea(contour):.0f} px")
                print(f"- Aspect Ratio: {aspect_ratio:.2f}")
                print(f"- Solidez: {solidity:.2f}")
                return approx

    print("⚠️ Usando el mejor contorno disponible (puede no ser perfecto)")
    return cv2.convexHull(valid_contours[0])


def corregir_perspectiva(imagen, doc_contour):
    """Corrige perspectiva manteniendo el máximo contenido posible"""

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

    # Calcular dimensiones del documento
    (tl, tr, br, bl) = rect
    width_top = np.sqrt(((tr[0] - tl[0]) ** 2 + (tr[1] - tl[1]) ** 2))
    width_bottom = np.sqrt(((br[0] - bl[0]) ** 2 + (br[1] - bl[1]) ** 2))
    max_width = max(int(width_top), int(width_bottom))

    height_left = np.sqrt(((tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2))
    height_right = np.sqrt(((tr[0] - br[0]) ** 2 + (tr[1] - br[1]) ** 2))
    max_height = max(int(height_left), int(height_right))

    # Puntos de destino con margen adicional
    margin = int(max(max_width, max_height) * 0.05)  # 5% de margen
    dst = np.array([
        [margin, margin],
        [max_width - 1 + margin, margin],
        [max_width - 1 + margin, max_height - 1 + margin],
        [margin, max_height - 1 + margin]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(imagen, M,
                                 (max_width + 2 * margin, max_height + 2 * margin),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
    return warped


def mejorar_documento(warped_image, output_type='color'):
    """Mejora el documento con opción para color o blanco/negro"""
    if output_type == 'bw':
        # Procesamiento para blanco y negro
        gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)

        # Mezcla de umbralización adaptativa y global
        thresh_adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        _, thresh_global = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Combinar resultados
        enhanced = cv2.bitwise_and(thresh_adapt, thresh_global)
        enhanced = cv2.bitwise_not(enhanced)

        # Mejorar calidad de texto
        kernel = np.ones((1, 1), np.uint8)
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)
        enhanced = cv2.medianBlur(enhanced, 3)
    else:
        # Procesamiento para color
        lab = cv2.cvtColor(warped_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Mejorar iluminación
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Reducción de ruido en canales de color
        a = cv2.medianBlur(a, 3)
        b = cv2.medianBlur(b, 3)

        # Mejorar nitidez
        lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Enfoque suave
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)

    return enhanced


def guardar_resultados(enhanced_image, output_path):
    """Guarda la imagen con compresión óptima"""
    cv2.imwrite(output_path, enhanced_image, [cv2.IMWRITE_JPEG_QUALITY, 95])


if __name__ == "__main__":
    # Configuración
    input_path = "foto2.jpg"
    output_path_bw = "documento_bw.jpg"
    output_path_color = "documento_color.jpg"

    try:
        # Paso 1: Cargar imagen
        print("🔍 Cargando imagen...")
        imagen, image_rgb = cargar_imagen(input_path)

        # Mostrar tamaño original
        print(f"📐 Tamaño original: {imagen.shape[1]}x{imagen.shape[0]}")

        # Paso 2: Preprocesamiento mejorado
        print("🛠️ Procesando imagen...")
        edged, gray = Preprocesar_Imagen(imagen)

        # Paso 3: Detección de documento con parámetros adaptativos
        print("🔎 Buscando documento...")
        doc_contour = encontrar_contorno(edged)

        if doc_contour is not None:
            # Paso 4: Corrección de perspectiva mejorada
            print("✂️ Recortando y corrigiendo perspectiva...")
            warped = corregir_perspectiva(imagen, doc_contour)

            # Mostrar información del resultado
            print(f"📏 Tamaño documento: {warped.shape[1]}x{warped.shape[0]}")

            # Paso 5: Mejora del documento (versión color)
            print("🎨 Mejorando documento (color)...")
            enhanced_color = mejorar_documento(warped, 'color')

            # Paso 6: Mejora del documento (versión blanco/negro)
            print("⚫⚪ Creando versión blanco/negro...")
            enhanced_bw = mejorar_documento(warped, 'bw')

            # Guardar resultados
            print("💾 Guardando resultados...")
            guardar_resultados(enhanced_color, output_path_color)
            guardar_resultados(enhanced_bw, output_path_bw)

            # Visualización
            plt.figure(figsize=(18, 12))

            plt.subplot(2, 2, 1)
            plt.imshow(image_rgb)
            plt.title("Imagen Original")
            plt.axis('off')

            plt.subplot(2, 2, 2)
            plt.imshow(edged, cmap='gray')
            plt.title("Bordes Detectados")
            plt.axis('off')

            plt.subplot(2, 2, 3)
            plt.imshow(cv2.cvtColor(enhanced_color, cv2.COLOR_BGR2RGB))
            plt.title("Documento Color Mejorado")
            plt.axis('off')

            plt.subplot(2, 2, 4)
            plt.imshow(enhanced_bw, cmap='gray')
            plt.title("Documento Blanco/Negro")
            plt.axis('off')

            plt.tight_layout()
            plt.show()

            print("✅ Proceso completado con éxito!")
        else:
            print("❌ No se pudo detectar un documento válido. Intente con otra imagen.")

    except Exception as e:
        print(f"❌ Error: {str(e)}")