import cv2


def detect_moving_objects(binary_image, min_area=1200):
    """
    Detecta objetos en movimiento por contornos.
    Ignora áreas pequeñas.
    """

    # Busca contornos en la imagen binaria.
    contours, _ = cv2.findContours(
        binary_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detections = []

    for contour in contours:

        area = cv2.contourArea(contour)

        # Descarta ruido pequeño.
        if area < min_area:
            continue

        # Obtiene caja del objeto.
        x, y, w, h = cv2.boundingRect(contour)

        # Calcula centro del objeto.
        cx = x + w // 2
        cy = y + h // 2

        detections.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "cx": cx,
            "cy": cy,
            "area": area
        })

    return detections