import cv2

MAX_CAMERAS = 2

class UtilsResolver:
    def getDevices(self):
        # Inicializar una lista para almacenar los dispositivos encontrados
        devices = []

        try:
            # Recorrer los índices de las cámaras disponibles
            for index in range(MAX_CAMERAS):  # Puedes ajustar el rango según la cantidad de cámaras que esperas
                cap = cv2.VideoCapture(index, cv2.CAP_V4L2)  # Intentar abrir la cámara en el índice dado
                if cap.isOpened():
                    # Si se abre correctamente, obtener información sobre el dispositivo
                    device_info = {
                        'index': index,
                        'name': f"Camera {index}",
                        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        'fps': int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else None,
                    }
                    devices.append(device_info)
                    cap.release()  # Liberar la cámara después de obtener la información

            return {'code': 1, 'message': 'Se han obtenido las cámaras disponibles correctamente', 'data': devices}
        except:
            return {'code': -1, 'message': 'Error al obtener las cámaras', 'data': []}
