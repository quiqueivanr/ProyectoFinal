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
    """Preprocesamiento mejorado para bordes más precisos"""
    # Convertir a escala de grises
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Mejorar contraste con CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Reducción de ruido más agresiva
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)

    # Detección de bordes con Canny adaptativo
    v = np.median(blurred)
    lower = int(max(0, 0.7 * v))
    upper = int(min(255, 1.3 * v))
    edged = cv2.Canny(blurred, lower, upper)

    # Operación morfológica para cerrar pequeños huecos
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    return edged, gray


def encontrar_contorno(edged_image, min_area_ratio=0.02, epsilon_ratio=0.02):
    """Encuentra el contorno del documento con criterios mejorados"""
    # Encontrar contornos
    contours, _ = cv2.findContours(edged_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("❌ Error: No se detectaron contornos")
        return None

    # Ordenar contornos por área (mayor a menor)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # Si no tiene 4 lados, forzar aproximación a 4 puntos
        if len(approx) != 4:
            # Aproximación más agresiva hasta obtener 4 puntos
            for epsilon in [0.03, 0.04, 0.05]:
                approx = cv2.approxPolyDP(contour, epsilon * peri, True)
                if len(approx) == 4:
                    break

        # Si ahora tiene 4 lados, verificar calidad
        if len(approx) == 4:
            if cv2.isContourConvex(approx):
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                solidity = cv2.contourArea(contour) / float(w * h)

                if 0.5 < aspect_ratio < 2.0 and solidity > 0.7:
                    print(f"✅ Contorno válido encontrado con {len(approx)} puntos")
                    return approx

    # Si no se encontró contorno válido, usar el mayor con 4 puntos forzados
    print("⚠️ Usando mejor contorno disponible (puede no ser perfecto)")
    largest_contour = contours[0]
    peri = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, 0.05 * peri, True)  # Aproximación más agresiva

    # Si aún no tiene 4 puntos, crear un rectángulo desde el bounding rect
    if len(approx) != 4:
        x, y, w, h = cv2.boundingRect(largest_contour)
        approx = np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]])

    return approx


def corregir_perspectiva(imagen, doc_contour):
    """Corrección de perspectiva con verificación de puntos"""
    # Asegurarse que el contorno tiene exactamente 4 puntos
    if doc_contour is None:
        raise ValueError("No se proporcionó un contorno válido")

    pts = doc_contour.reshape(-1, 2)
    if pts.shape[0] != 4:
        # Si no tiene 4 puntos, crear un rectángulo desde el bounding rect
        x, y, w, h = cv2.boundingRect(doc_contour)
        pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype="float32")

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

    # Puntos de destino SIN margen
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    # Transformación de perspectiva
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(imagen, M, (max_width, max_height))

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


def recortar_bordes(imagen, porcentaje=2):
    """Recorta un pequeño porcentaje de los bordes para eliminar artefactos"""
    h, w = imagen.shape[:2]
    crop_h = int(h * porcentaje / 100)
    crop_w = int(w * porcentaje / 100)
    return imagen[crop_h:h-crop_h, crop_w:w-crop_w]

if __name__ == "__main__":
    # Configuración
    input_path = "prueba.jpg"
    output_BN = "documento_BN.jpg"
    output_color = "documento_color.jpg"

    try:
        # Paso 1: Cargar imagen
        print("Cargando imagen...")
        imagen, image_rgb = cargar_imagen(input_path)

        # Mostrar tamaño original
        print(f"Tamaño original: {imagen.shape[1]}x{imagen.shape[0]}")

        # Paso 2: Preprocesamiento mejorado
        print("Procesando imagen...")
        edged, gray = Preprocesar_Imagen(imagen)

        # Paso 3: Detección de documento con parámetros adaptativos
        print(" Buscando documento...")
        doc_contour = encontrar_contorno(edged)

        if doc_contour is not None:
            # Paso 4: Corrección de perspectiva mejorada
            print("Recortando y corrigiendo perspectiva...")
            warped = corregir_perspectiva(imagen, doc_contour)

            # Recortar bordes residuales
            final = recortar_bordes(warped, porcentaje=1)  # Ajusta el porcentaje según necesites

            # Mostrar información del resultado
            print(f"Tamaño documento: {warped.shape[1]}x{warped.shape[0]}")

            # Paso 5: Mejora del documento (versión color)
            print("Mejorando documento (color)...")
            enhanced_color = mejorar_documento(final, 'color')

            # Paso 6: Mejora del documento (versión blanco/negro)
            print("Creando versión blanco/negro...")
            enhanced_bw = mejorar_documento(warped, 'bw')

            # Guardar resultados
            print("💾 Guardando resultados...")
            guardar_resultados(enhanced_color, output_color)
            guardar_resultados(enhanced_bw, output_BN)

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

            print("✅ Documento escaneado con éxito!")
        else:
            try:
                warped = corregir_perspectiva(imagen, doc_contour)
            except Exception as e:
                print(f"⚠️ Error al corregir perspectiva: {str(e)}. Usando imagen original.")
                warped = imagen.copy()

    except Exception as e:
        print(f"❌ Error: {str(e)}")