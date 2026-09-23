import cv2
import numpy as np


class VideoStabilizer:

    def __init__(self):

        self.previous_gray = None


    def stabilize(self, frame):
        """
        Estabiliza el frame usando el anterior.
        """

        # Convierte a escala de grises.

        current_gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        # Si es el primer frame, lo guarda y sale.

        if self.previous_gray is None:

            self.previous_gray = current_gray

            return frame.copy()


        # Busca puntos útiles para seguir movimiento.

        previous_points = cv2.goodFeaturesToTrack(
            self.previous_gray,
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=30,
            blockSize=3
        )


        if previous_points is None:

            self.previous_gray = current_gray

            return frame.copy()


        # Calcula cómo se movieron esos puntos.

        current_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.previous_gray,
            current_gray,
            previous_points,
            None
        )


        if current_points is None:

            self.previous_gray = current_gray

            return frame.copy()


        # Filtra solo puntos confiables.

        status = status.reshape(-1)

        good_previous = previous_points[
            status == 1
        ]

        good_current = current_points[
            status == 1
        ]


        # Se necesitan varios puntos para estimar el movimiento.

        if len(good_previous) < 4:

            self.previous_gray = current_gray

            return frame.copy()


        # Estima el desplazamiento entre frames.

        transformation, _ = cv2.estimateAffinePartial2D(
            good_previous,
            good_current,
            method=cv2.RANSAC
        )


        if transformation is None:

            self.previous_gray = current_gray

            return frame.copy()


        # Invierte ese movimiento para corregir el frame.

        inverse_transformation = cv2.invertAffineTransform(
            transformation
        )


        height, width = frame.shape[:2]


        # Aplica la corrección al frame.

        stabilized = cv2.warpAffine(
            frame,
            inverse_transformation,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )


        # Guarda el nuevo estado para el siguiente frame.

        self.previous_gray = current_gray


        return stabilized