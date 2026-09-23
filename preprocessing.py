import cv2
import numpy as np


# Parámetros del preprocesamiento.

MOTION_THRESHOLD = 40
BLUR_KERNEL = 5
DILATION_KERNEL = 5
DILATION_ITERATIONS = 2


def preprocess_motion(previous_frame, current_frame):
    """
    Detecta movimiento entre dos frames.
    """

    # Convierte a escala de grises.

    previous_gray = cv2.cvtColor(
        previous_frame,
        cv2.COLOR_BGR2GRAY
    )

    current_gray = cv2.cvtColor(
        current_frame,
        cv2.COLOR_BGR2GRAY
    )


    # Suaviza la imagen.

    previous_gray = cv2.GaussianBlur(
        previous_gray,
        (BLUR_KERNEL, BLUR_KERNEL),
        0
    )

    current_gray = cv2.GaussianBlur(
        current_gray,
        (BLUR_KERNEL, BLUR_KERNEL),
        0
    )


    # Calcula la diferencia entre frames.

    difference = cv2.absdiff(
        previous_gray,
        current_gray
    )


    # Separa movimiento del fondo.

    _, threshold = cv2.threshold(
        difference,
        MOTION_THRESHOLD,
        255,
        cv2.THRESH_BINARY
    )


    # Quita ruido pequeño.

    median = cv2.medianBlur(
        threshold,
        5
    )


    # Agranda zonas relevantes.

    kernel = np.ones(
        (DILATION_KERNEL, DILATION_KERNEL),
        np.uint8
    )

    dilated = cv2.dilate(
        median,
        kernel,
        iterations=DILATION_ITERATIONS
    )


    return difference, threshold, median, dilated