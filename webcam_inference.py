import os
import cv2
import json
import argparse
from bin.LPR import LPR
from threading import Thread
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from typing import Generator
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from typing import Generator
from bin.VideoStream import VideoStream
from bin.utils import get_grouped_lanes

frame = None
app = FastAPI()

# Crear un generador para los fotogramas
def generate_frames() -> Generator[bytes, None, None]:
    global frame
    while True:
        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--graph",
    help="Name of the .tflite file, if different than detect.tflite",
    default="detect.tflite",
)
parser.add_argument(
    "--labels",
    help="Name of the labelmap file, if different than labelmap.txt",
    default="labelmap.txt",
)
parser.add_argument(
    "--threshold",
    help="Minimum confidence threshold for displaying detected objects",
    default=0.5,
)
parser.add_argument(
    "--resolution",
    help="Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.",
    default="1280x720",
)
parser.add_argument(
    "--edgetpu",
    help="Use Coral Edge TPU Accelerator to speed up detection",
    action="store_true",
)

args = parser.parse_args()

GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split("x")
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu
show_outputs = False
min_conf = 0.3

# Initialize LPR instance
settings = {
    "id:": "0",
    "label": "0, True:True, True:True",
    "plates_model": "lpr_models/nimbus",  # plates model
    "ocr_model": "lpr_models/aurora",  # ocr model
    "equalize_plates": False,
    "equalize_ocr": True,
}

lprModel = LPR(
    plates_model=settings["plates_model"],
    ocr_model=settings["ocr_model"],
    equalize_plates=settings["equalize_plates"],
    equalize_ocr=settings["equalize_ocr"],
)

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

videostream = None
# time.sleep(1)

def loop():
    global frame
    while True:
        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()

        # Grab frame from video stream
        # frame1 = videostream.read()

        # Acquire frame and resize to expected shape [1xHxWx3]
        # frame = frame1.copy()

        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame_resized = cv2.resize(frame_rgb, (width, height))
        # input_data = np.expand_dims(frame_resized, axis=0)

        # Aquí tengo que hacer la predicción...
        detections = lprModel.predict(
            image_path=frame,
            show_image=show_outputs,
            show_boxes=True,
            min_conf=min_conf,
        )
        print("DETECTIONS: ", detections)

        # Draw framerate in corner of frame
        # cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2,cv2.LINE_AA)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow("Object detector", frame)

        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2 - t1) / freq
        frame_rate_calc = 1 / time1

        # Press 'q' to quit
        # if cv2.waitKey(1) == ord("q"):
        #    break

    # Clean up
    # cv2.destroyAllWindows()
    videostream.stop()

def capture_stream():
    global frame
    while True:
        frame = videostream.read()

# Ruta para el streaming de video
@app.get("/video_feed")
async def video_feed():
    global videostream
    if videostream is None:
        return "stream unavailable"
    return StreamingResponse(
        generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )

# Cuando se haya configurado la cámara, se iniciará el streaming de vídeo
@app.get("/run_stream")
def run_stream():
    # Initialize video stream
    global videostream
    videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
    Thread(target=capture_stream, args=()).start()
    return {"code": 1, "message": "Ok, Streaming started."}


# Cuando la configuración esté lista, se invocará este método para reconocer las placas
@app.get("/run_inference")
def run():
    Thread(target=loop, args=()).start()
    return {"code": 1, "message": "Ok, started."}

@app.post('/save_config')
def save_config(config):
    # Aquí guardaré la configuración de cámara, etc...
    '''
        {
            camera: {
                connection_type: "usb" | "rtsp",
                rtsp_url: string,
                device: number,
            }
        },
        lanes: []
    '''
    print(config)
    
@app.get("/get_config")
def get_config():
    # Tengo que verificar si el archivo de configuración existe
    file_config_path = os.path.join('./', 'config.json')
    if not os.path.exists(file_config_path):
        return {'code': -1, 'message': 'La configuración no está disponible.'}
    config_data = None
    with open('config.json', 'r') as f:
        config_data = json.load(f)

    if config_data is None:
        return {'code': -1, 'message': 'La configuración no está disponible.'}

    return {'code': 1, 'message': "Correcto", 'data': config_data}
    

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
