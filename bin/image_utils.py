import os
import cv2
import random
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from glob import glob
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
from bin.shared_utils import removeKey, checkDatasetHealth, prettify_xml, update_xml_info, get_image_path, get_image_size

def load_boxes_from_xml(xml_path):
    boxes = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        boxes.append((name, (xmin, ymin, xmax, ymax)))
    return boxes

def draw_boxes_on_image(image_path, boxes):
    # Leer la imagen con OpenCV
    image = cv2.imread(image_path)
    
    # Iterar sobre las cajas delimitadoras y el texto
    final_text = ''
    for name, (xmin, ymin, xmax, ymax) in boxes:
        # Dibujar la caja delimitadora con OpenCV
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 255, 255), 1)
        
        # Añadir texto a la imagen con OpenCV
        font_scale = 0.8
        if name == 'plate':
            ymin += ymax - ymin + 30
        cv2.putText(image, name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 3)
        final_text += name + ('_' if name == 'plate' else '')
    
    # Mostrar la imagen con matplotlib
    print(final_text)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Desactivar los ejes
    plt.show()

def draw_xml_boxes(image, xml_boxes):
    draw = ImageDraw.Draw(image)
    for obj in xml_boxes:
        boxes = obj['boxes']
        xmin, ymin, xmax, ymax = boxes
        draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=(255, 0, 255), width=2)
    return image

def show_xml_targets(xml_path):
    boxes = load_boxes_from_xml(xml_path)
    image_path = get_image_path(xml_path)
    if not os.path.exists(image_path):
        return 'Image not exists'
    draw_boxes_on_image(image_path, boxes)

def recalculateBoxes(xml_path, overlapping_object, newboxes, scale):
    # Parent coords
    parentCoords = overlapping_object['boxes']
    xminParent, yminParent, xmaxParent, ymaxParent = parentCoords
    xminNew, yminNew, xmaxNew, ymaxNew = newboxes
    
    xml_boxes = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for obj in root.findall('object'):
        name = obj.find('name')
        if name is not None and name.text != overlapping_object['name']:
            selector = name.text

            box = obj.find('bndbox')
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            item_boxes = (xmin, ymin, xmax, ymax)
            
            # Child coords
            #item_boxes = current_boxes[name.text]
            xmin, ymin, xmax, ymax = item_boxes
            xmin -= xminParent
            ymin -= yminParent
            xmax -= xminParent
            ymax -= yminParent

            xmin = int(xmin * scale) + xminNew
            ymin = int(ymin * scale) + yminNew
            xmax = int(xmax * scale) + xminNew
            ymax = int(ymax * scale) + yminNew
            xml_boxes.append({"name":selector, "boxes":(xmin, ymin, xmax, ymax)})
        elif name is not None and name.text == overlapping_object['name']:
            selector = name.text
            xml_boxes.append({"name":selector, "boxes":newboxes})
            
    return xml_boxes

def find_overlapping_object(xml_file):
    # Parsear el archivo XML
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Obtener todos los objetos
    objects = root.findall('object')

    # Inicializar variables para el objeto superpuesto
    overlapping_object = None
    overlapping_area = 0

    # Iterar sobre cada objeto y calcular el área de su bounding box
    for obj in objects:
        # Obtener las coordenadas del bounding box
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)

        # Calcular el área del bounding box
        area = (xmax - xmin) * (ymax - ymin)

        # Verificar si el área es mayor que el área de superposición actual
        if area > overlapping_area:
            overlapping_area = area
            overlapping_object = obj

    if overlapping_object is not None:
        name = overlapping_object.find('name').text
        xmin = int(overlapping_object.find('bndbox/xmin').text)
        ymin = int(overlapping_object.find('bndbox/ymin').text)
        xmax = int(overlapping_object.find('bndbox/xmax').text)
        ymax = int(overlapping_object.find('bndbox/ymax').text)
        return {"name":name, "boxes":(xmin,ymin,xmax,ymax)}
    
    return None

# Actualiza los boxes de un archivo xml
# xml_path = Archivo
# newcoors = [{name: 'plate', 'boxes': (xmin, ymin, xmax, ymax)}]
def update_xml_coors(xml_path, newcoors):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for element in root.findall('object'):
        name = element.find('name')
        if name is not None:
            for obj in newcoors:
                if obj['name'] == name.text and not obj.get('used', False):
                    obj['used'] = True
                    boxes = obj['boxes']
                    (xmin, ymin, xmax, ymax) = boxes

                    element.find('bndbox/xmin').text = str(xmin)
                    element.find('bndbox/ymin').text = str(ymin)
                    element.find('bndbox/xmax').text = str(xmax)
                    element.find('bndbox/ymax').text = str(ymax)
                    break  # Salir del bucle para evitar que se sobrescriban los valores del XML

    # Convertir el árbol XML a una cadena formateada
    xml_string = ET.tostring(root, encoding='utf-8')
    dom = minidom.parseString(xml_string)
    xml_formatted = dom.toprettyxml(indent='  ')

    # Escribir el XML formateado en el archivo
    with open(xml_path, 'w') as f:
        f.write(xml_formatted)

def select_area(image_path, center_x, center_y, boxes, area_size=320):
    # Abrir la imagen
    image = Image.open(image_path)

    # Calcular las coordenadas del área
    xmin = max(0, center_x - area_size // 2)
    ymin = max(0, center_y - area_size // 2)
    xmax = min(image.width, xmin + area_size)
    ymax = min(image.height, ymin + area_size)

    # Desempaquetar los valores de boxes
    x1, y1, x2, y2 = boxes
    
    scale_factor = 1
    
    # Verificar si los cuadros delimitadores están dentro del área de recorte
    if x1 < xmin or y1 < ymin or x2 > xmax or y2 > ymax:
        print('RESIZING IMAGE...')
        # Calcular el nuevo tamaño de la imagen
        new_width = max(x2 - x1, y2 - y1)
        new_height = new_width
        
        # Ajustar los límites del área de recorte si es necesario
        xmin = max(0, min(x1, xmin))
        xmax = min(image.width, max(x2, xmax))
        ymin = max(0, min(y1, ymin))
        ymax = min(image.height, max(y2, ymax))
        
        # Calcular el centro de los cuadros delimitadores
        box_center_x = (x1 + x2) // 2
        box_center_y = (y1 + y2) // 2
        
        # Calcular el factor de escala para el zoom out
        scale_factor = min(area_size / new_width, area_size / new_height)
        
        # Redimensionar la imagen para el zoom out
        scaled_image = image.resize((int(image.width * scale_factor), int(image.height * scale_factor)))
        
        # Actualizar las coordenadas del área de recorte
        xmin = max(0, int(box_center_x * scale_factor - area_size // 2))
        ymin = max(0, int(box_center_y * scale_factor - area_size // 2))
        xmax = min(scaled_image.width, xmin + area_size)
        ymax = min(scaled_image.height, ymin + area_size)
        
        # Recortar la imagen según el área de recorte
        cropped_image = scaled_image.crop((xmin, ymin, xmax, ymax))
        
        # Recalcular las coordenadas de los cuadros delimitadores
        new_x1 = max(0, x1 * scale_factor - xmin)
        new_y1 = max(0, y1 * scale_factor - ymin)
        new_x2 = min(xmax - xmin, max(0, x2 * scale_factor - xmin))
        new_y2 = min(ymax - ymin, max(0, y2 * scale_factor - ymin))

    else:
        # Recortar la imagen según el área seleccionada
        cropped_image = image.crop((xmin, ymin, xmax, ymax))
        new_x1, new_y1, new_x2, new_y2 = x1 - xmin, y1 - ymin, x2 - xmin, y2 - ymin

    return cropped_image, (round(new_x1), round(new_y1), round(new_x2), round(new_y2)), scale_factor

def resize_dataset_item(xml_path, area_size = 320, autosave = False, show_image=True, show_boxes=False):
    image_path = get_image_path(xml_path)
    if not os.path.exists(image_path):
        print(f'{image_path} not exists.')
        print(f'XML Deleted: {xml_path}')
        os.remove(xml_path)
        return

    overlapping_object = find_overlapping_object(xml_path)
    if overlapping_object is None:
        return False

    boxes = overlapping_object['boxes']
    xmin, ymin, xmax, ymax = boxes
    x_center = xmin + (xmax - xmin) // 2
    y_center = ymin + (ymax - ymin) // 2

    # Recalculate and select area
    image, newboxes, scale = select_area(image_path, x_center, y_center, boxes, area_size)
    xml_boxes = recalculateBoxes(xml_path, overlapping_object, newboxes, scale=scale)

    if show_image:
        plt.imshow(image)
        plt.show()

    if autosave:
        print('File:', xml_path)
        print('Se actualizó la imagen y los boxes en archivos originales...')
        image.save(image_path)
        update_xml_coors(xml_path, xml_boxes)

    if not autosave and show_boxes:
        temp_image = draw_xml_boxes(image.copy(), xml_boxes)
        plt.imshow(temp_image)
        plt.show()

    if autosave and show_boxes:
        show_xml_targets(xml_path)
    return True

def clean_dataset(dataset_directory):
    xml_paths = glob(os.path.join(dataset_directory, '**', '*.xml'), recursive=True)
    deleted = []
    for xml_path in xml_paths:
        if not os.path.exists(get_image_path(xml_path)):
            os.remove(xml_path)
            deleted.append(xml_path)
    print(f'({len(deleted)}) files removed')
    print('Deleted List:', deleted)
    return deleted

def limit_samples_per_item(summary = [], limit = 1000):
    # Lista para almacenar los elementos procesados
    result = []
    
    # Itera sobre cada elemento en summary
    for item in summary:
        # Verifica si hay archivos disponibles para cumplir con el límite
        valid_files = [file for file in item['files'] if os.path.exists(get_image_path(file))]
        if len(valid_files) >= limit and len(valid_files) >= len(valid_files) * 2:
            # Inicializa una lista para almacenar los archivos seleccionados aleatoriamente
            random_files = []
                
            print('chossing for ', item['label'])
            
            # Intenta seleccionar aleatoriamente un archivo que exista y no esté en random_files
            while len(random_files) < limit:
                file = random.choice(valid_files)
                if file not in random_files:
                    random_files.append(file)
        elif len(valid_files) * 2 > len(valid_files):
            random_files = valid_files[:1000]
        else:
            # Si no hay suficientes archivos válidos, usa todos los archivos originales válidos
            random_files = valid_files
        
        # Agrega el elemento al resultado
        result.append({'label': item['label'], 'num': len(random_files), 'files': random_files})
    return result

from PIL import Image
import sys

def equalize_image_size(image_path):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            if width != height:
                if width > height:
                    y_offset = (width - height) // 2
                    new_img = Image.new("RGB", (width, width), color="black")
                    new_img.paste(img, (0, y_offset))
                    x_offset, y_offset = 0, y_offset
                else:
                    x_offset = (height - width) // 2
                    new_img = Image.new("RGB", (height, height), color="black")
                    new_img.paste(img, (x_offset, 0))
                    x_offset, y_offset = x_offset, 0
                new_img.save(image_path)
                return x_offset, y_offset
            else:
                return 0, 0
    except IOError as e:
        print(f"No se pudo abrir la imagen en la ruta especificada: {e}")

def equalize_image(xml_path):
    # Redimensionar la imagen
    x_offset, y_offset = equalize_image_size(get_image_path(xml_path))
    # newcoors = [{name: 'plate', 'boxes': (xmin, ymin, xmax, ymax)}]
    newcoors = []
    boxes = load_boxes_from_xml(xml_path)
    for item in boxes:
        name, boxs = item
        xmin, ymin, xmax, ymax = boxs

        # Sumamos x_offset y y_offset
        xmin += x_offset
        xmax += x_offset
        ymin += y_offset
        ymax += y_offset
        newcoors.append({'name': name, 'boxes': (xmin, ymin, xmax, ymax)})
    # Actualizamos los coors en el archivo xml
    update_xml_coors(xml_path, newcoors)
