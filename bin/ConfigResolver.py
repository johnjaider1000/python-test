import os
import uuid
import json
from bin.DeviceParams import DeviceParams

config_folder = os.path.join('./', '.config')

if not os.path.exists(config_folder):
    os.makedirs(config_folder)
    
file_config_path = os.path.join(config_folder, 'cameras.json')

class ConfigResolver:
    def save_camera(self, device: DeviceParams):
        # Generar un nuevo UUID para la cámara
        camera_id = str(uuid.uuid4()) if device.id is None else device.id
        
        # Crear el nuevo objeto de cámara
        new_camera = {
            'id': camera_id,
            'custom_code': device.custom_code,
            'connection_type': device.connection_type,
            'rtsp_url': device.rtsp_url,
            'device': device.device,
            'effects': device.effects,
            'lanes': device.lanes,
            'props': device.props,
        }

        # Leer el archivo config.json si existe para actualizarlo
        if os.path.exists(file_config_path):
            with open(file_config_path, 'r') as f:
                config_data = json.load(f)
                cameras = config_data.get('cameras', [])
                                
                # Detectar si la cámara ya existe
                cameraIndex = None
                for index, camera in enumerate(cameras):
                    if camera['props']['index'] == new_camera['props']['index'] and camera['id'] != new_camera['id']:
                        return {'code': -1, 'message': 'La cámara seleccionada ya se encuentra registrada'}
                    
                    if camera['id'] == new_camera['id']:
                        cameraIndex = index
                
                if cameraIndex is None:                    
                    cameras.append(new_camera)
                else:
                    cameras[cameraIndex] = new_camera
        else:
            # Si el archivo no existe, crear una estructura nueva
            cameras = [new_camera]

        # Actualizar o crear el diccionario de configuración
        self.write_cameras_file(cameras)       

        return {'code': 1, 'message': 'Se ha guardado la configuración', 'cameraId': camera_id}


    def write_cameras_file(self, cameras):
        config_data = {
            'cameras': cameras
        }

        # Escribir el diccionario actualizado en el archivo config.json
        with open(file_config_path, 'w') as f:
            json.dump(config_data, f, indent=1)


    def get_cameras(self):
         # Tengo que verificar si el archivo de configuración existe
        if not os.path.exists(file_config_path):
            return {'code': -1, 'message': 'La configuración no está disponible.'}
        
        config_data = None
        with open(file_config_path, 'r') as f:
            config_data = json.load(f)

        if config_data is None:
            return {'code': -1, 'message': 'La configuración no está disponible.'}

        return {'code': 1, 'message': "Correcto", 'data': config_data}
    
    
    def filter_devices_unregistered(self, devices):
        cameras = self.get_cameras()
        filtered_devices = []
        if cameras['code'] > 0:
            cameras = cameras['data']['cameras']
            # Filtrar los dispositivos que no se encuentren en cameras
            
            for device in devices:
                found = False
                
                for camera in cameras:
                    if device['index'] == camera['props']['index']:
                        found = True
                if not found:
                    filtered_devices.append(device)
            return filtered_devices
        else:
            return devices

    
    def set_areas(self, device: DeviceParams):
        if not os.path.exists(file_config_path):
            return {'code': -1, 'message': 'La configuración no está disponible.'}
        
        cameras = self.get_cameras()['data']
        cameras = cameras['cameras']
        
        # Buscar la cámara por id
        findedIndex = None
        for index,camera in enumerate(cameras):
            if camera['id'] == device.id:
                findedIndex = index
        
        # Actualizamos las áreas
        if findedIndex is None:
            return {'code': -1, 'message': 'Cámara no encontrada.'}
        
        cameras[findedIndex]['lanes'] = device.lanes
        
        # Escribimos el archivo
        self.write_cameras_file(cameras)
        return {'code': 1, 'message': "Se ha actualizado la configuración"}


    def remove_camera(self, id: str):
        cameras = self.get_cameras()['data']
        cameras = cameras['cameras']
        # Buscar la cámara por id
        filtered_cameras = []
        found_camera = None
        for camera in cameras:
            if camera['id'] != id:
                filtered_cameras.append(camera)
            else:
                found_camera = camera
        
        self.write_cameras_file(filtered_cameras)
        return found_camera


    def remove_config(self):
        if os.path.exists(file_config_path):
            os.unlink(file_config_path)
        return {'code': 1, 'message': "Se ha reseteado la configuración"}
