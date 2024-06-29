import psycopg2
from psycopg2 import sql
from datetime import datetime
from pydantic import BaseModel

class HistoryEntity(BaseModel):
    plate: str
    device: str
    camera: str
    lane: str
    image_path: str
    boxes: str

class HistoryModel:
    def __init__(self, dbname = 'vibble_db', user = 'admin', password = '12345', host='localhost', port=5432):
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.conn = None
        self.connect()

    def connect(self):
        try:
            self.conn = psycopg2.connect(
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port
            )
            print("Conexión exitosa a la base de datos")
        except psycopg2.Error as e:
            print(f"Error al conectar a la base de datos: {e}")

    def insert(self, plate, device, camera, lane, image_path, boxes):
        try:
            cursor = self.conn.cursor()
            insert_query = sql.SQL("""
                INSERT INTO history.history (plate, device, camera, lane, date, image_path, boxes)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """)
            now = datetime.now().date()
            cursor.execute(insert_query, (plate, device, camera, lane, now, image_path, boxes))
            self.conn.commit()
            print("Inserción exitosa")
        except psycopg2.Error as e:
            self.conn.rollback()
            print(f"Error al insertar en la tabla: {e}")
        finally:
            cursor.close()

    def close(self):
        if self.conn is not None:
            self.conn.close()
            print("Conexión cerrada")
