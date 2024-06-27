import os
import uuid
import json
from bin.AuthParams import AuthParams

file_auth_path = os.path.join('./', '.config', 'auth.json')

class AuthResolver:
    def add_user(auth: AuthParams):
        # Generar un nuevo UUID para la cámara
        new_user_id = str(uuid.uuid4())
        
        # Crear el nuevo objeto de cámara
        new_user = {
            'id': new_user_id,
            'username': auth.username,
            'password': auth.password,        
        }

        # Leer el archivo config.json si existe para actualizarlo
        if os.path.exists(file_auth_path):
            with open(file_auth_path, 'r') as f:
                config_data = json.load(f)
                cameras = config_data.get('users', [])
                cameras.append(new_user)
        else:
            # Si el archivo no existe, crear una estructura nueva
            cameras = [new_user]

        # Actualizar o crear el diccionario de configuración
        config_data = {
            'users': cameras
        }

        # Escribir el diccionario actualizado en el archivo config.json
        with open(file_auth_path, 'w') as f:
            json.dump(config_data, f, indent=4)

        return {'code': 1, 'message': 'Se ha guardado la configuración', 'userId': new_user_id}