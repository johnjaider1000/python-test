# Descarga la versión 3.8 de python
FROM python:3.8

# Establecer el directorio de trabajo
WORKDIR /app

# Actualizar el sistema e instalar dependencias del sistema para OpenCV
RUN apt-get update && apt-get install -y \
  libgl1-mesa-glx \
  libglib2.0-0

# Instalar dependencias
RUN pip install fastapi uvicorn opencv-python-headless requests Pillow matplotlib

COPY . /app

# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["python", "webcam_inference.py"]
