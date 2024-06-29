import re
import cv2
from bin.LPR import LPR
from threading import Thread
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Generator
from bin.VideoStream import VideoStream
# from bin.utils import get_grouped_lanes
from bin.DeviceParams import DeviceParams
from bin.ConfigResolver import ConfigResolver
from bin.UtilsResolver import UtilsResolver
from bin.Effects import Effects

def is_rtsp_url(url: str) -> bool:
    rtsp_regex = re.compile(r'^rtsp://.*')
    return bool(rtsp_regex.match(url))

frames = {}
videostreams = {}
devices = []
effects = {}

# Configuración inicial
resolution = '1280x720'
resW, resH = resolution.split("x")
imW, imH = int(resW), int(resH)
use_TPU = 'store_true'
show_outputs = False
min_conf = 0.3
lprModel = None

# Initialize LPR instance
settings = {
    "id:": "0",
    "label": "0, True:True, True:True",
    "plates_model": "lpr_models/nimbus",  # plates model
    "ocr_model": "lpr_models/aurora",  # ocr model
    "equalize_plates": False,
    "equalize_ocr": True,
}

app = FastAPI()
origins = [
    "http://localhost:8282"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['GET', 'POST'],
    allow_headers=['*'],
)

def init():
    print('Iniciando dispositivos...')
    
    # Cargar los devices
    global devices
    _configResolver = ConfigResolver()    
    cameras = _configResolver.get_cameras()
    if cameras['code'] > 0:
        list_cameras = cameras['data']['cameras']
        for camera in list_cameras:
            devices.append(camera['props'])
    
        # Ejecutar las funciones en los devices
    
        # 1. Ejecutar los streams:
        for device in devices:
            prepare_stream(str(device['index']))

# Crear un generador para los fotogramas
def generate_frames(camera_id) -> Generator[bytes, None, None]:
    global frames
    effectsInstance = Effects()
    if not camera_id in frames:
        raise IOError('No se encontró el streaming')
    
    while True:
        frame = frames[camera_id]
        
        if camera_id in effects:
            brightness = effects[camera_id]['brightness']
            contrast = effects[camera_id]['contrast']
            
            print('brightness:', brightness)
            print('contrast:', contrast)
            
            if brightness or contrast:
                frame = effectsInstance.ilumination(img=frame, brightness=brightness, contrast=contrast)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


def load_model():
    global lprModel

    if lprModel is not None:
        return
    
    # Cargamos el modelo.
    lprModel = LPR(
        plates_model=settings["plates_model"],
        ocr_model=settings["ocr_model"],
        equalize_plates=settings["equalize_plates"],
        equalize_ocr=settings["equalize_ocr"],
    )


def loop(camera_id):
    global frames
    global videostreams

    while True:
        # Aquí tengo que hacer la predicción...
        detections = lprModel.predict(
            image_path=frames[camera_id],
            show_image=show_outputs,
            show_boxes=True,
            min_conf=min_conf,
        )
        print("DETECTIONS: ", detections)


def capture_stream(camera_id):
    global frames
    while camera_id in videostreams:
        frames[camera_id] = videostreams[camera_id].read()
    print('capture_stream ended:', camera_id)


# Ruta para el streaming de video
@app.get("/video_feed/{camera_id}")
async def video_feed(camera_id: int):
    global videostreams
    if videostreams[camera_id] is None:
        return "stream unavailable"
    return StreamingResponse(
        generate_frames(camera_id), media_type="multipart/x-mixed-replace; boundary=frame"
    )


def prepare_stream(device_id: str):
    global videostreams
    isRTSP = is_rtsp_url(device_id) if isinstance(device_id, str) else False
    input = None
    if not isRTSP:
        input = int(device_id)
        
    if device_id not in videostreams:        
        videostreams[device_id] = VideoStream(input=input).start()
        Thread(target=capture_stream, args=(device_id,)).start()

        
@app.get('/stream_device/{device_id}')
def stream_device(device_id: str):
    # Initialize video stream
    prepare_stream(device_id)    
        
    return StreamingResponse(
        generate_frames(device_id), media_type="multipart/x-mixed-replace; boundary=frame"
    )


# Cuando se haya configurado la cámara, se iniciará el streaming de vídeo
@app.get("/run_stream/{camera_id}")
def run_stream(camera_id: str):
    # Initialize video stream
    global videostreams
    videostreams[camera_id] = VideoStream().start()
    Thread(target=capture_stream, args=(camera_id,)).start()
    return {"code": 1, "message": "Ok, Streaming started."}


@app.post('/stop_stream/{camera_id}')
def stop_stream(camera_id: str):
    global videostreams
    if camera_id in videostreams:        
        videostreams[camera_id].stop()
        del videostreams[camera_id]        
        return {"code": 1, "message": "Ok, Streaming stoped."}
    
    return {"code": -1, "message": "Invalid camera id."}


# Cuando la configuración esté lista, se invocará este método para reconocer las placas
@app.get("/run_inference/{camera_id}")
def run(camera_id: str):
    load_model()
    Thread(target=loop, args=(camera_id,)).start()
    return {"code": 1, "message": "Ok, started."}


@app.post('/save_camera')
def save_camera(device: DeviceParams):
    _configResolver = ConfigResolver()
    return _configResolver.save_camera(device)


@app.get("/get_cameras")
def get_cameras():
   _configResolver = ConfigResolver()
   return _configResolver.get_cameras()


@app.post('/set_areas')
def set_areas(device: DeviceParams):
    _configResolver = ConfigResolver()
    return _configResolver.set_areas(device)


@app.get('/get_devices')
def get_devices():
    global devices
    _utilsResolver = UtilsResolver()
    response = _utilsResolver.getDevices()
    # devices.extend(response['data'])
    for item in response['data']:
        found = False
        for device in devices:
            if item['index'] == device['index']:
                found = True
        if not found:
            devices.append(item)
        
    _configResolver = ConfigResolver()
    devices = _configResolver.filter_devices_unregistered(devices)
    response['data'] = devices
    return response

@app.post('/remove_device')
def remove_device(device: DeviceParams):
    global devices
    # Removemos la cámara del archivo de configuración
    _configResolver = ConfigResolver()    
    deleted_camera = _configResolver.remove_camera(device.id)
    print('deleted_camera:', deleted_camera)
    if deleted_camera is None:
        return {'code': -1, 'message': 'Esta cámara ya no existe'}
    
    # Removemos en los devices actuales
    filtered_devices = []
    for device in devices:
        if device['index'] != deleted_camera['props']['index']:
            filtered_devices.append(device)
    devices = filtered_devices
    
    # Detenemos los procesos existentes de stream
    stop_stream(str(deleted_camera['props']['index']))
    return {'code': 1, 'message': 'Se ha removido el dispositivo y se han detenido las tareas en ejecución'}

@app.post('/apply_filters')
def apply_filters(device: DeviceParams):
    _device = get_device(device.id)
    if _device['code'] < 0:
        return _device
    
    deviceIndex =_device['data']['props']['index']
    effects[str(deviceIndex)] = {
        'brightness': device.effects['brightness'] if 'brightness' in device.effects else None,
        'contrast': device.effects['contrast'] if 'contrast' in device.effects else None,
    }

@app.get('/get_device/{id}')
def get_device(id: str):
    _configResolver = ConfigResolver()
    cameras = _configResolver.get_cameras()
    if cameras['code'] > 0:
        cameras = cameras['data']['cameras']
        # Buscamos el dispositivo por id
        for camera in cameras:
            if camera['id'] == id:
                return {'code': 1, 'message': 'Se ha encontrado el dispositivo correctamente', 'data': camera}
    
    return {'code': -1, 'message': 'No se encontró el dispositivo'}

init()

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
