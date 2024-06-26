import os
import time
import shutil
import tensorflow as tf
import subprocess
import numpy as np
import random
from glob import glob
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import re
import json
import cv2
import sys
import importlib.util
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def get_image_path(filepath):
    # voy a priorizar primero si existe otro archivo con el mismo nombre
    #if not os.path.exists(filepath_image):
    xmlpath = filepath
    xmlname = os.path.basename(xmlpath)
    folder = xmlpath.replace(xmlname, '')
    
    filepath_image = None
    search = os.path.join(folder, xmlname.replace('.xml', '.*'))
    coincidences = glob(search)
    image_files = [file for file in coincidences if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    #print('# COINCIDENCES FOR', filepath, 'search:', search, image_files)
    if len(image_files) > 0:
        return image_files[0]
   
    filename_image = ET.parse(xmlpath).getroot().find('filename').text
    filepath_image = os.path.join(folder, filename_image)

    return filepath_image

def update_xml_info(xml_file):
    not_found_files = []
    # Lo primero será actualizar encabezados (folder, filename, path)
    image_path = get_image_path(xml_file)
    if os.path.exists(image_path):
        file_name = os.path.basename(image_path)
        folder = image_path.replace(file_name, '')
        replace_key_item(xml_file, 'folder', folder)
        replace_key_item(xml_file, 'filename', file_name)
        replace_key_item(xml_file, 'path', image_path)

        # Ahora voy a actualizar los sizes de la imagen en el archivo xml
        width, height = get_image_size(image_path)
        replace_key_item(xml_file, 'size/width', str(width))
        replace_key_item(xml_file, 'size/height', str(height))
    else:
        print('NOT FOUND IMAGE:', image_path, 'for', xml_file)
        not_found_files.append({'image_path': image_path, 'xml_path': xml_file})
    
    return not_found_files

def update_dataset_headers(dataset_directory):
    xml_paths = glob(os.path.join(dataset_directory, '**', '*.xml'), recursive=True)
    print(f'({len(xml_paths)}) files found.')
    not_found = []
    for xml_path in xml_paths:
        print('Procesando: ', xml_path)
        not_found += update_xml_info(xml_path)

    # Removerá los archivos que aún tengan problemas para este punto crítico del entrenamiento:
    for item in not_found:
        image_path = item['image_path']
        xml_path = item['xml_path']
    
        print(xml_path, os.path.exists(xml_path))
        print(image_path, os.path.exists(image_path))
    
        if os.path.exists(xml_path):
            os.remove(xml_path)
            print('XML Removed: ', xml_path)
        if os.path.exists(image_path):
            os.remove(image_path)
            print('Image Removed: ', image_path)

def removeKey(file, removes = []):
    #print('processing: ', file)
    tree = ET.parse(file)
    root = tree.getroot()

    for element in root.findall('object'):
        name = element.find('name')
        if name is not None and name.text in removes:
            print(name.text, 'removed in', file)
            root.remove(element)

    # Convertir el árbol XML a una cadena formateada
    xml_string = ET.tostring(root, encoding='utf-8')
    dom = minidom.parseString(xml_string)
    xml_formatted = dom.toprettyxml(indent='  ')

    # Escribir el XML formateado en el archivo
    with open(file, 'w') as f:
        f.write(xml_formatted)

# Remove from xml images_path the removes list keys
def removeKeys(images_path, removes = []):
    print(f'# Remove keys {removes} in {images_path}')
    paths = glob(os.path.join(images_path, '**', '*.xml'), recursive=True)
    print(f'({len(paths)}) files founded')
    
    for file in paths:
        removeKey(file, removes)
    print(f'# ({len(paths)}) files proccesed')

def unzip(file, output = os.getcwd()):
    run_cmd(f'unzip {file} -x "__MACOSX/*" -d {output};')


def load_json_file(input):
    try:
        with open(input, 'r') as file:
            data = json.load(file)
            print("Se ha cargado el archivo '{}' como objeto Python:".format(input))
            print(data)
            return data
    except FileNotFoundError:
        print("El archivo '{}' no fue encontrado.".format(input))
    except Exception as e:
        print("Ocurrió un error:", e)


def write_object_to_json(object, output):
    try:
        # Abre el archivo en modo escritura
        with open(output, 'w') as file:
            # Escribe el objeto como JSON en el archivo
            json.dump(object, file)
            print("Se ha escrito el objeto Python como JSON en el archivo '{}'.".format(output))
    except Exception as e:
        print("Ocurrió un error:", e)

def run_cmd(cmd):
    print(cmd)
    subprocess.run(cmd, shell=True)

def replaceLabel(file, replaces = []):
    print('processing: ', file)
    tree = ET.parse(file)
    root = tree.getroot()

    for item_for_replace in replaces:
        for key, value in item_for_replace.items():
            for element in root.findall('object'):
                name = element.find('name')
                if name is not None and name.text == key:
                    # Replace
                    print('# Replace', name.text, value)
                    name.text = value

    # Convertir el árbol XML a una cadena formateada
    xml_string = ET.tostring(root, encoding='utf-8')
    dom = minidom.parseString(xml_string)
    xml_formatted = dom.toprettyxml(indent='  ')

    # Escribir el XML formateado en el archivo
    with open(file, 'w') as f:
        f.write(xml_formatted)

"""
images_paths = xml paths files
replaces = [{"PLATE_NUMBER": "plate"}]
"""
def replaceLabels(images_path, replaces = []):
    print('REPLACES IN', images_path)
    file_paths = os.path.join(images_path, '**', '*.xml')
    paths = glob(file_paths, recursive=True)
    print(f'# ({len(paths)}) FOR REPLACE LABEL...')
    for file in paths:
        replaceLabel(file, replaces)

def removeExtraObjects(file, labels_selection = []):
    tree = ET.parse(file)
    root = tree.getroot()

    for element in root.findall('object'):
        name = element.find('name')
        if name is not None and not name.text in labels_selection:
            print(name.text, 'removed')
            root.remove(element)

    # Convertir el árbol XML a una cadena formateada
    xml_string = ET.tostring(root, encoding='utf-8')
    dom = minidom.parseString(xml_string)
    xml_formatted = dom.toprettyxml(indent='  ')

    # Escribir el XML formateado en el archivo
    with open(file, 'w') as f:
        f.write(xml_formatted)

def rewriteLabelSelection(images_path, labels_selection = []):
    print('REWRITE IN',images_path)
    file_paths = os.path.join(images_path, '**', '*.xml')
    paths = glob(file_paths, recursive=True)
    print(f'# ({len(paths)}) FOR REWRITING LABEL SELECTION...')
    for filepath in paths:
        removeExtraObjects(filepath, labels_selection)

def replace_key_item(xml_file, key, new_filename):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename_element = root.find(key)
    if filename_element is not None:
        #print(f'ITEM {key} UPDATED:', filename_element.text, '=>', new_filename)
        filename_element.text = new_filename

    # Convertir el árbol XML a una cadena formateada
    xml_formatted = prettify_xml(root)

    # Escribir el XML formateado en el archivo
    with open(xml_file, 'w') as f:
        f.write(xml_formatted)

def remove_extra_targets(dataset_directory, limit=1000):
    dataset_paths = glob(os.path.join(dataset_directory, '**', '*.xml'), recursive=True)
    labels, summary = checkDatasetHealth(dataset_directory)
    result = limit_samples_per_item(summary, limit)

    for item in result:
        i = 0
        label = item['label']
        for xml_path in dataset_paths:
            if xml_path not in item['files']:
                # Llama a la función removeKey() con la ruta completa al archivo XML
                removeKey(os.path.join(xml_path), removes=[label])
            elif not os.path.exists(get_image_path(xml_path)):
                print('Removed File:', xml_path)
                os.remove(xml_path)
            else:
                i += 1
        print(f'{(label)} Valid Targets:', i)

    print('Now (after remove extra targets):')
    labels, summary = checkDatasetHealth(dataset_directory)


def printDatasetHealt(data, show_graphics=True):
    # Obtener etiquetas y valores
    labels = [item["label"] for item in data]
    values = [item["num"] for item in data]
    
    # Ordenar las etiquetas junto con sus valores correspondientes
    sorted_data = sorted(zip(labels, values), key=lambda x: x[1], reverse=True)
    labels, values = zip(*sorted_data)
    
    # Convertir los valores de "num" a enteros
    max_value = max(values)
    percentages = [value / max_value * 100 for value in values]
    
    # Crear colores aleatorios
    random.seed(0)  # Para reproducibilidad
    colors = ['#' + ''.join(random.choices('0123456789ABCDEF', k=6)) for _ in range(len(labels))]
    
    # Calcular la altura del gráfico en función de la cantidad de etiquetas
    altura_grafico = len(labels) * 0.5  # Ajusta este valor según tus necesidades

    if show_graphics:
        # Crear el gráfico de barras con la altura ajustada
        fig, ax = plt.subplots(figsize=(8, altura_grafico))  # Ajusta el ancho según sea necesario
        y = np.arange(len(labels))
        bar_height = 0.35
        
        bars = ax.barh(y, values, bar_height, color=colors, label='Valores')
        
        # Añadir etiquetas de porcentaje y cantidad de valores encontrados
        for i, (percentage, value) in enumerate(zip(percentages, values)):
            ax.text(value + 5, bars[i].get_y() + bars[i].get_height()/2., f'{value} - ({percentage:.0f}%)', va='center')
        
        # Añadir detalles al gráfico
        ax.set_xlabel(f'Valores ({max_value})')
        ax.set_title(f'Balance de Clases ({len(data)})')
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.legend()
        
        # Mostrar el gráfico
        plt.show()
    return labels

def checkDatasetHealth(dataset_path, show_graphics=True):
    print('# checking dataset healt:', dataset_path)
    numtargets = 0
    
    if not dataset_path or not os.path.exists(dataset_path):
        raise FileNotFoundError(f'El archivo {dataset_path} no existe.')
        return

    xml_paths = glob(os.path.join(dataset_path, '**', '*.xml'), recursive=True)
    labels_base = {}
    
    def getLabels(file):
        #print('processing: ', file)
        tree = ET.parse(file)
        root = tree.getroot()
        
        for element in root.findall('object'):
            name = element.find('name')
            if name is not None:
                nameValue = name.text
                labels_base[nameValue] = labels_base[nameValue] + [file] if nameValue in labels_base else [file]
                
    for xml_path in xml_paths:
        getLabels(xml_path)


    summary = [{"label": key, "num": len(value), "files": labels_base[key]} for key, value in labels_base.items()]
    total_targets = sum(item["num"] for item in summary)
    
    print(f'({len(xml_paths)}) files processed')
    print(f'({total_targets}) targets found')
    labels = printDatasetHealt(summary, show_graphics)
    return sorted(labels), summary

def remove_newlines_and_spaces(text):
    """Remove all newlines and extra spaces from the text."""
    return re.sub(r'\s+', ' ', text)

def prettify_xml(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, 'utf-8').decode()
    rough_string = remove_newlines_and_spaces(rough_string)
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def get_image_size(image_path):
    """Obtener el tamaño de la imagen (ancho y alto)."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return width, height
    except IOError:
        print("No se pudo abrir la imagen en la ruta:", image_path)
        return None

def write_new_xml(objects, output, width = 300, height = 300):
    # Crear el elemento raíz <annotation>
    annotation = ET.Element("annotation")
    
    # Añadir elementos hijos a <annotation>
    folder = ET.SubElement(annotation, "folder")
    folder.text = "./dataset/"
    filename = ET.SubElement(annotation, "filename")
    filename.text = "annotation_ocr_temp.jpg"
    path = ET.SubElement(annotation, "path")
    path.text = "./dataset/annotation_ocr_temp.jpg"
    source = ET.SubElement(annotation, "source")
    database = ET.SubElement(source, "database")
    database.text = "Unknown"
    size = ET.SubElement(annotation, "size")
    width = ET.SubElement(size, "width")
    width.text = str(width)
    height = ET.SubElement(size, "height")
    height.text = str(height)
    depth = ET.SubElement(size, "depth")
    depth.text = "3"
    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = "0"

    # Crea un nuevo objeto
    for obj in objects:
        object = ET.SubElement(annotation, "object")
        name = ET.SubElement(object, "name")
        name.text = obj['name']
        
        pose = ET.SubElement(object, "pose")
        pose.text = "Unspecified"
        truncated = ET.SubElement(object, "truncated")
        truncated.text = "1"
        difficult = ET.SubElement(object, "difficult")
        difficult.text = "0"

        boxes = obj['boxes']
        xminValue, yminValue, xmaxValue, ymaxValue = boxes
        # Crea el objeto bndbox para las coordenadas del objeto
        bndbox = ET.SubElement(object, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(xminValue)
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(yminValue)
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(xmaxValue)
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(ymaxValue)
    
    # Crear el árbol XML
    tree = ET.ElementTree(annotation)
    
    # Escribir el árbol XML en un archivo
    tree.write(output, encoding="utf-8", xml_declaration=True)

def current_milli_time():
    return round(time.time() * 1000)

# Voy a extraer solo las placas y le voy a mostrar a Yolo esto para que reconozca letras directamente en placas.
def getPlateCoords(parts):
    plateCoords = None
    for i in range(len(parts)):
        label, coords = parts[i]
        if label.lower() == 'plate_number' or label.lower() == 'plate' or label.lower() == 'plate_moto':
            plateCoords = coords
            return plateCoords

def updateCoords(parts, currentIndex):
    # coordenadas de placa
    coordsPlate = getPlateCoords(parts)
    xminPlate, yminPlate, xmaxPlate, ymaxPlate = coordsPlate

    # coordenadas de letra
    partName, coords = parts[currentIndex]
    
    xmin,ymin,xmax,ymax = coords
    xmin -= xminPlate
    ymin -= yminPlate
    xmax -= xminPlate
    ymax -= yminPlate
    return (xmin,ymin,xmax,ymax)

def pilToCv2(image):
  return np.array(image)[:,:,::-1].copy()

def getYoloBoxes(boxes, image_width, image_height, current_category=0):
  x,y,w,h = boxes
  # Finding midpoints
  x_centre = (x + (x+w))/2
  y_centre = (y + (y+h))/2

  # Normalization
  x_centre = x_centre / image_width
  y_centre = y_centre / image_height
  w = w / image_width
  h = h / image_height

  # Limiting upto fix number of decimal places
  x_centre = format(x_centre, '.6f')
  y_centre = format(y_centre, '.6f')
  w = format(w, '.6f')
  h = format(h, '.6f')

  return f"{current_category} {x_centre} {y_centre} {w} {h}"

def getPlateFileOut(image_path, parts):
    # cortar imagen con Image.crop
    image = Image.open(image_path)
    coords = getPlateCoords(parts)
    print('coors', coords)
    xmin,ymin,xmax,ymax = coords
    box = (xmin,ymin,xmax,ymax) #left,top,right,bottom
    im = image.crop(box)
    fileout = image_path
    im.save(fileout)
    return fileout

def getSummary(coords, labelname, filepath, parts):
    #xmin,ymin,xmax,ymax = getPlateCoords(parts)
    #width = xmax
    #height = ymax
    xmin,ymin,xmax,ymax = coords
    return {'file': filepath, 'name': labelname, 'boxes': (xmin,ymin,xmax,ymax)}

def getGroups(coordinates):
    plates = []
    current_plate = []
    for label, (xmin,ymin,xmax,ymax) in coordinates:
        if label == 'plate':
            if current_plate:
                plates.append(current_plate)
            current_plate = []
        current_plate.append((label, (xmin,ymin,xmax,ymax)))
    if current_plate:
        plates.append(current_plate)
    return plates

def processGroups(groups):
      # Si los grupos es mayor a dos, vamos a copiar los archivos...
    if len(groups) >= 2:
        results = []
        print('Duplicando archivos...')
        i = 1
        for group in groups:
            #obj = [{'name': 'plate', 'boxes': boxes}] # Estructura objeto para escribir xml nuevo...
            objects = []
            # Procesamos parts para armar el objects que requiere write_new_xml para escribir el nuevo xml...
            for box in group:
                name, boxes = box
                xmin,ymin,xmax,ymax = boxes
                obj = {'name': name, 'boxes': (xmin,ymin,xmax,ymax)}
                objects.append(obj)

            # Extraemos el formato del archivo
            img_format = os.path.basename(image_path).split('.')[-1]
            # Extraemos el nombre del archivo
            file_name = os.path.splitext(os.path.basename(filepath))[0]
            # Creamos un nuevo nombre para el archivo nuevo
            base_path = filepath.replace(os.path.basename(filepath), '')
            output_name = f'{file_name}_{i}'
            output_xml = os.path.join(base_path, f'{output_name}.xml')
            
            # Escribimos el nuevo xml
            write_new_xml(objects, output_xml)
            print('new xml file writed:', output_xml)
            
            # Escribimos el nuevo archivo de imagen
            output_img = os.path.join(base_path, f'{output_name}.{img_format}')
            shutil.copy(image_path, output_img)
            print('new image file writed:', output_img)
            
            # Ahora procesamos el item nuevamente...
            result = processItem({'filepath': output_xml, 'parts': group, 'hasPlateNumber': True}, single=True)
            results.append(result)
            i += 1
        return results

def processItem(item, single=False):
    filepath = item['filepath']
    image_path = get_image_path(item['filepath'])
    parts = item['parts']
    print('processItem:', item)
    #groups = getGroups(parts)
    #print('GROUPS:', groups)
    #return processGroups(groups)

    fileout = getPlateFileOut(image_path, parts)
    #fileout = get_image_path(filepath)

    newcoors = []
    for i in range(len(parts)):
        partName, coords = parts[i]
        xmin,ymin,xmax,ymax = coords
        #image = Image.open(fileout)
        
        # guardaré el archivo si es placa.
        if not partName.lower() in ['plate_number', 'plate_moto', 'plate']:
            newCoords = updateCoords(parts, i)
            result = getSummary(newCoords, partName, filepath, parts)
            newcoors.append(result)
        elif partName.lower() in ['plate_number', 'plate_moto', 'plate']:
            width = xmax - xmin
            height = ymax - ymin
            newcoors.append({'file': filepath, 'name': partName, 'boxes': (0,0,width,height)})

    if single:
        return {'filepath': filepath, 'newcoors': newcoors}

    return [{'filepath': filepath, 'newcoors': newcoors}]
