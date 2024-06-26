# Author: John Vanegas - github.com/johnjaider1000
import os
import re
import cv2
import time
import numpy as np
from statistics import mean
import requests
from glob import glob
from io import BytesIO
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.lite.python.interpreter import Interpreter

replaces_dict = {
    "0": "O",
}


class AccuracyInvalidError(Exception):
    def __init__(self, file, accuracy):
        self.message = f"Accuracy inválido: Accuracy: {accuracy} - Archivo: {file}"
        super().__init__(self.message)


class Inference:
    def __init__(self, model_path, labelmap_path=None, equalize=False):
        file_path_model = (
            os.path.join(model_path, "detect.tflite")
            if os.path.isdir(model_path)
            else model_path
        )
        label_path_model = (
            os.path.join(model_path, "labelmap.txt")
            if os.path.isdir(model_path)
            else (
                labelmap_path
                if labelmap_path is not None
                else os.path.join(os.path.dirname(model_path), "labelmap.txt")
            )
        )

        self.equalize = equalize
        self.model = self.load_model(file_path_model)
        self.labels = self.load_labelmap(label_path_model)
        print("Model loaded: ", file_path_model)

    def load_labelmap(self, lblpath):
        with open(lblpath, "r") as f:
            labels = [line.strip() for line in f.readlines()]
            return labels

    def boxes_overlap(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        diff = abs(x1 - x2)
        diff_tolerance = w1 if w1 < w2 else w2

        return diff < (diff_tolerance * 0.1)

    def load_model(self, model_path):
        # Load the Tensorflow Lite model into memory
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.outname = self.output_details[0]["name"]

        if "StatefulPartitionedCall" in self.outname:  # This is a TF2 model
            self.model_ids = 1, 3, 0
        else:  # This is a TF1 model
            self.model_ids = 0, 1, 2

        self.height = self.input_details[0]["shape"][1]
        self.width = self.input_details[0]["shape"][2]

        self.float_input = self.input_details[0]["dtype"] == np.float32

        self.input_mean = 127.5
        self.input_std = 127.5

    def is_url(self, string):
        # Patrón de expresión regular para verificar si la cadena es una URL
        url_pattern = re.compile(
            r"^(?:http|ftp)s?://"  # http:// or https://
            r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # domain...
            r"localhost|"  # localhost...
            r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ipv4
            r"(?::\d+)?"  # optional port
            r"(?:/?|[/?]\S+)$",
            re.IGNORECASE,
        )

        return re.match(url_pattern, string) is not None

    def is_cv2_image(self, variable):
        return isinstance(variable, np.ndarray) and len(variable.shape) == 3

    def equalize_image_size(self, image):
        try:
            height, width, _ = image.shape
            if width != height:
                if width > height:
                    y_offset = (width - height) // 2
                    new_img = cv2.copyMakeBorder(
                        image,
                        y_offset,
                        y_offset,
                        0,
                        0,
                        cv2.BORDER_CONSTANT,
                        value=(0, 0, 0),
                    )
                    x_offset, y_offset = 0, y_offset
                else:
                    x_offset = (height - width) // 2
                    new_img = cv2.copyMakeBorder(
                        image,
                        0,
                        0,
                        x_offset,
                        x_offset,
                        cv2.BORDER_CONSTANT,
                        value=(0, 0, 0),
                    )
                    x_offset, y_offset = x_offset, 0
                # Devolvemos la imagen y las coordenadas de desplazamiento
                return new_img, x_offset, y_offset
            else:
                # Si la imagen ya es cuadrada, simplemente devolvemos la imagen original
                return image, 0, 0
        except Exception as e:
            print(f"Error al procesar la imagen: {e}")
            return None, 0, 0

    def draw_boxes(self, image, boxes, text=None):
        xmin, ymin, xmax, ymax = boxes
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 255), 2)
        if text is not None and text != "":
            fontSize = 0.6
            labelSize, baseLine = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, fontSize, 2
            )  # Get font size
            label_ymin = max(
                ymin, labelSize[1] + 10
            )  # Make sure not to draw label too close to top of window
            cv2.rectangle(
                image,
                (xmin, label_ymin - labelSize[1] - 10),
                (xmin + labelSize[0], label_ymin + baseLine - 10),
                (255, 0, 255),
                cv2.FILLED,
            )  # Draw white box to put label text in
            cv2.putText(
                image,
                text,
                (xmin, label_ymin - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontSize,
                (255, 255, 255),
                2,
            )  # Draw label text

    def draw_detections(self, image, detections):
        for detection in detections:
            name, accuracy, xmin, ymin, xmax, ymax = detection
            self.draw_boxes(
                image, (xmin, ymin, xmax, ymax), f"({round(accuracy)}%) {name}"
            )

    def sort_predictions(self, detections):
        sorted_data = sorted(detections, key=lambda x: (x[2], x[3], x[4], x[5]))
        return sorted_data

    def predict(
        self,
        image_path,
        min_conf=0.5,
        x_init=0,
        y_init=0,
        show_image=False,
        draw_boxes=False,
    ):
        image = None
        if self.is_cv2_image(image_path):
            image = image_path
        elif self.is_url(image_path):
            response = requests.get(image_path)
            if response.status_code == 200:
                image_bytes = BytesIO(response.content)
                image_array = np.frombuffer(image_bytes.getvalue(), dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            else:
                print("No se pudo descargar", image_path)
        else:
            image = cv2.imread(image_path)

        image_temp = image

        x_offset = 0
        y_offset = 0

        if self.equalize:
            image_temp, x_offset, y_offset = self.equalize_image_size(image)

        image_rgb = cv2.cvtColor(image_temp, cv2.COLOR_BGR2RGB)
        imH, imW, _ = image.shape
        imtemp_H, imtemp_W, _ = image_temp.shape
        image_resized = cv2.resize(image_rgb, (self.width, self.height))
        input_data = np.expand_dims(image_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if self.float_input:
            input_data = (np.float32(input_data) - self.input_mean) / self.input_std

        boxes_idx, classes_idx, scores_idx = self.model_ids

        # Perform the actual detection by running the model with the image as input
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()

        # Retrieve detection results
        boxes = self.interpreter.get_tensor(self.output_details[boxes_idx]["index"])[
            0
        ]  # Bounding box coordinates of detected objects
        classes = self.interpreter.get_tensor(
            self.output_details[classes_idx]["index"]
        )[
            0
        ]  # Class index of detected objects
        scores = self.interpreter.get_tensor(self.output_details[scores_idx]["index"])[
            0
        ]  # Confidence of detected objects

        detections = []
        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if (scores[i] > min_conf) and (scores[i] <= 1.0):
                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(0, (boxes[i][0] * imtemp_H) - y_offset)) + y_init
                xmin = int(max(0, (boxes[i][1] * imtemp_W) - x_offset)) + x_init
                ymax = int(min(imH, (boxes[i][2] * imtemp_H) - y_offset)) + y_init
                xmax = int(min(imW, (boxes[i][3] * imtemp_W) - x_offset)) + x_init

                # cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (255, 0, 255), 2)

                # Draw label
                object_name = self.labels[
                    int(classes[i])
                ]  # Look up object name from "labels" array using class index
                label = "%s: %d%%" % (
                    object_name,
                    int(scores[i] * 100),
                )  # Example: 'person: 72%'
                # label = object_name
                # fontSize = 0.6
                # labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fontSize, 2) # Get font size
                # label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                # cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 0, 255), cv2.FILLED) # Draw white box to put label text in
                # cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255, 255, 255), 2) # Draw label text
                detections.append(
                    [object_name, int(scores[i] * 100), xmin, ymin, xmax, ymax]
                )

        if draw_boxes:
            self.draw_detections(image, detections)

        if show_image:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.show()

        return self.sort_predictions(detections), image


class LPR:
    def __init__(
        self, plates_model, ocr_model, equalize_plates=False, equalize_ocr=False
    ):
        self.platesInference = Inference(plates_model, equalize=equalize_plates)
        self.ocrInference = Inference(ocr_model, equalize=equalize_ocr)

    def encode_roi(self, roi):
        # Codificar la imagen cortada en formato JPEG
        _, encoded_image = cv2.imencode(".jpg", roi)
        # Verificar si la imagen codificada tiene un tamaño válido
        if encoded_image.size > 0:
            # Decodificar la imagen codificada
            decoded_image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)

            # Verificar si la decodificación fue exitosa y la forma de la imagen decodificada
            if decoded_image is not None:
                return decoded_image
            else:
                print("Error al decodificar la imagen.")
        else:
            print("El array codificado está vacío.")

    def get_roi(self, image, boxes):
        xmin, ymin, xmax, ymax = boxes
        roi = image[ymin:ymax, xmin:xmax]
        return roi

    def get_text(self, ocr_detections):
        text = ""
        for detection in ocr_detections:
            name, accuracy, xmin, ymin, xmax, ymax = detection
            text += name
        return text

    def print_boxes(self, image, boxes, text=None):
        xmin, ymin, xmax, ymax = boxes
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 255), 2)
        if text is not None and text != "":
            fontSize = 0.6
            labelSize, baseLine = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, fontSize, 2
            )  # Get font size
            label_ymin = max(
                ymin, labelSize[1] + 10
            )  # Make sure not to draw label too close to top of window
            cv2.rectangle(
                image,
                (xmin, label_ymin - labelSize[1] - 10),
                (xmin + labelSize[0], label_ymin + baseLine - 10),
                (255, 0, 255),
                cv2.FILLED,
            )  # Draw white box to put label text in
            cv2.putText(
                image,
                text,
                (xmin, label_ymin - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontSize,
                (255, 255, 255),
                2,
            )  # Draw label text

    def print_detections(self, image, detections):
        for plate in detections:
            boxes = plate["boxes"]
            text = plate["text"]
            print("text", text)
            self.print_boxes(image, boxes, text)

    def overlap(self, box1, box2, tolerance=0.1):
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2

        # Calcula el ancho y la altura de cada cuadro
        w1 = xmax1 - xmin1
        h1 = ymax1 - ymin1
        w2 = xmax2 - xmin2
        h2 = ymax2 - ymin2

        # Calcula la diferencia entre las coordenadas x de los dos cuadros
        diff_x = abs(xmin1 - xmin2)

        # Define la tolerancia de diferencia basada en el ancho del cuadro más pequeño
        diff_tolerance = min(w1, w2)

        # print('diff_x', diff_x, 'diff_tolerance', (diff_tolerance * tolerance))

        # Compara la diferencia horizontal calculada con un porcentaje de la tolerancia de diferencia
        return diff_x < (diff_tolerance * tolerance)

    def clear_lines(self, data, tolerance=0.4):
        if len(data) <= 6:
            return data

        # Inicializar la lista de líneas vacía
        lines = []

        # Iterar sobre los datos
        min_element_height = min(data, key=lambda line: line[5] - line[3])
        min_height = (
            min_element_height[5] - min_element_height[3]
        )  # Extraer la anchura de min_width

        for item in data:
            # Flag para indicar si se añadió el elemento a una línea existente
            added_to_line = False

            # Iterar sobre las líneas existentes
            for line in lines:
                vertical_tolerance = (
                    min_height * tolerance
                )  # Tolerance as a percentage of the minimum height
                # Comprobar si el elemento pertenece a esta línea
                if abs(line[-1][3] - item[3]) <= vertical_tolerance:
                    line.append(item)
                    added_to_line = True
                    break

            # Si el elemento no se añadió a ninguna línea existente, crear una nueva línea para él
            if not added_to_line:
                lines.append([item])

        # Ordenar las líneas basadas en el valor mínimo de ymin
        lines.sort(key=lambda x: x[0][3])
        return lines[0] if lines else []

    def clear_predictions(self, values=[], plate_boxes=None, tolerance=0.6):
        # print('detections=', values, 'plate_boxes=',plate_boxes)
        if len(values) == 0:
            return values

        plate_center = None
        if plate_boxes is not None:
            pxmin, pymin, pxmax, pymax = plate_boxes
            plate_width = pxmax - pxmin
            plate_center = (plate_width // 2) + pxmin

        # Limpiará las líneas, es posible que el modelo reconozca carácteres de la ciudad, entonces priorizaré las placas
        if plate_center is not None:
            values = self.clear_lines(values, tolerance)

        # min_width_element = min(values, key=lambda item: item[4] - item[2])
        # min_width = min_width_element[4] - min_width_element[2]  # Extraer la anchura de min_width
        min_width = mean(item[4] - item[2] for item in values)

        horizontal_tolerance = min_width * tolerance

        n = len(values)
        to_remove = set()

        letter_items = set()
        number_items = set()

        # Iterar sobre todos los elementos en la lista
        for i in range(n):
            # Últimos i elementos ya están en su posición correcta
            for j in range(0, n - i - 1):
                # Intercambiar si el elemento actual es mayor que el siguiente
                name1, accuracy1, xmin1, ymin1, xmax1, ymax1 = values[j]
                name2, accuracy2, xmin2, ymin2, xmax2, ymax2 = values[j + 1]

                interference = min(xmax1, xmax2) - max(xmin1, xmin2) + 1

                # Si el accuracy del elemento actual es mayor al siguiente y se pisan los cables remueve el siguiente
                if accuracy1 > accuracy2 and interference >= horizontal_tolerance:
                    to_remove.add(j + 1)

                # Si el accuracy del elemento actual es menor al siguiente y se pisan los cables remueve el elemento actual
                elif accuracy1 < accuracy2 and interference >= horizontal_tolerance:
                    to_remove.add(j)

                if plate_center is not None:
                    is_first_part = xmin1 - ((xmax1 - xmin1) * 0.50) <= plate_center

                    # if is_first_part:
                    if is_first_part and name1.isalpha() or (len(letter_items) < 3):
                        # if is_first_part and name1.isalpha() or (len(letter_items) < 3 and values > 6):
                        # Actualizar aquí con el diccionario de caracteres
                        if not name1.isalpha() and name1 in replaces_dict:
                            values[j][0] = replaces_dict[name1]
                        letter_items.add(j)
                    else:
                        number_items.add(j)

        # Eliminar elementos marcados para eliminar
        temp_values = [item for i, item in enumerate(values) if i not in to_remove]
        if plate_center is None:
            # Eliminar elementos marcados para eliminar
            return temp_values

        # print('temp_values:', temp_values)

        letters = [
            item
            for i, item in enumerate(values)
            if i in letter_items and i not in to_remove
        ]
        numbers = [
            item
            for i, item in enumerate(values)
            if i in number_items and i not in to_remove
        ]

        numbers.append(temp_values[-1])

        # print('letters:', letters)
        # print('numbers:', numbers)

        # Agrega el último item

        if len(letters) > 3:
            # Tengo que recorrerlos y verificar cual está más cerca al centro de place_center, así evitaré predicciones con bajo accuracy erroneas fuera del marco de la placa
            new_array = []

            # Recorremos el array original en reversa
            for i in range(len(letters) - 3, len(letters)):
                new_array.append(letters[i])

            letters = new_array

        if len(numbers) > 3:
            # Tengo que recorrerlos y verificar cual está más cerca al centro de place_center, así evitaré predicciones con bajo accuracy erroneas fuera del marco de la placa
            new_array = []

            # Recorremos el array original en reversa
            for i in range(3):
                new_array.append(numbers[i])

            numbers = new_array

        # values = [item for i, item in enumerate(values) if i not in to_remove]

        filtered_values = letters + numbers

        return filtered_values

    def clear_plates(self, values=[]):
        to_remove = set()
        n = len(values)
        for i in range(n):
            for j in range(0, n - i - 1):
                name1, accuracy1, xmin1, ymin1, xmax1, ymax1 = values[j]
                name2, accuracy2, xmin2, ymin2, xmax2, ymax2 = values[j + 1]

                overlap = self.plate_overlap(
                    (xmin1, ymin1, xmax1, ymax1), (xmin2, ymin2, xmax2, ymax2)
                )
                if overlap and accuracy1 > accuracy2:
                    to_remove.add(j + 1)
                elif overlap and accuracy1 < accuracy2:
                    to_remove.add(j)

        filtered_values = [item for i, item in enumerate(values) if i not in to_remove]
        return filtered_values

    def plate_overlap(self, box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Verificar si las cajas se cruzan en el eje X
        overlap_x = x1_min < x2_max and x1_max > x2_min

        # Verificar si las cajas se cruzan en el eje Y
        overlap_y = y1_min < y2_max and y1_max > y2_min

        # Las cajas se cruzan si hay superposición en ambos ejes
        return overlap_x and overlap_y

    def are_boxes_intersecting(self, box1, box2, tolerance=0.1):
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2

        # Calcular las dimensiones de las cajas
        width1 = xmax1 - xmin1
        height1 = ymax1 - ymin1
        width2 = xmax2 - xmin2
        height2 = ymax2 - ymin2

        # Aplicar la tolerancia a las dimensiones de las cajas
        tolerance_width1 = width1 * tolerance
        tolerance_height1 = height1 * tolerance
        tolerance_width2 = width2 * tolerance
        tolerance_height2 = height2 * tolerance

        # Expandir las cajas según la tolerancia
        xmin1 -= tolerance_width1
        ymin1 -= tolerance_height1
        xmax1 += tolerance_width1
        ymax1 += tolerance_height1
        xmin2 -= tolerance_width2
        ymin2 -= tolerance_height2
        xmax2 += tolerance_width2
        ymax2 += tolerance_height2

        # Verificar si hay intersección en los ejes X e Y
        x_intersect = not (xmax1 < xmin2 or xmax2 < xmin1)
        y_intersect = not (ymax1 < ymin2 or ymax2 < ymin1)

        # Si hay intersección en ambos ejes, las cajas se están cruzando
        return x_intersect and y_intersect

    def getTotalAccuracy(self, detections):
        total = 0
        for detection in detections:
            total += detection["accuracy"]
            for item in detection["ocr_detections"]:
                total += item[1]

        if total == 0:
            return total

        return total / (len(detections) * 6 + len(detections))

    def getOcrTotalAccuracy(self, detection):
        total = detection["accuracy"]
        for item in detection["ocr_detections"]:
            total += item[1]

        if total == 0:
            return total

        return total / 7

    def predict(
        self,
        image_path,
        min_conf=0.5,
        show_image=False,
        show_boxes=False,
        process_text=True,
        extract_roi=False,
    ):
        # Busca la placa:
        detections = []
        plates_detections, image = self.platesInference.predict(
            image_path, min_conf=min_conf
        )
        plates_detections = self.clear_plates(plates_detections)

        for plate in plates_detections:
            name, accuracy, xmin, ymin, xmax, ymax = plate
            plate_boxes = (xmin, ymin, xmax, ymax)
            roi = self.get_roi(image, plate_boxes)
            encoded_roi = self.encode_roi(roi)
            if process_text:
                ocr_detections, _ = self.ocrInference.predict(
                    encoded_roi, min_conf=min_conf, x_init=xmin, y_init=ymin
                )
                if len(ocr_detections) > 6:
                    print(
                        "Más de 6 detecciones encontradas:",
                        ocr_detections,
                        "plate_boxes:",
                        plate_boxes,
                    )
                    ocr_detections = self.clear_predictions(ocr_detections, plate_boxes)
                else:
                    ocr_detections = self.clear_predictions(ocr_detections, plate_boxes)
            else:
                ocr_detections = []

            detection = {
                "text": self.get_text(ocr_detections),
                "name": name,
                "accuracy": accuracy,
                "ocr_detections": ocr_detections,
                "boxes": (xmin, ymin, xmax, ymax),
            }

            # detections['total_accuracy'] = self.getOcrTotalAccuracy(detection)

            if extract_roi:
                detection["roi"] = encoded_roi

            if len(detection["ocr_detections"]) >= 3:
                print("OMITED:", detection)
                detections.append(detection)

        total_accuracy = self.getTotalAccuracy(detections)
        if total_accuracy <= 50:
            return []

        if show_boxes:
            self.print_detections(image, detections)

        if show_image:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.show()

        return detections
