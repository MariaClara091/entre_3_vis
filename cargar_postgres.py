import pandas as pd
import psycopg2
from sqlalchemy import create_engine

# Configuración de la base de datos
DB_CONFIG = {
    'dbname': 'base',      
    'user': 'postgres',      
    'password': 'vis2025..',  
    'host': 'localhost',
    'port': '5432'
}

def cargar_datos_postgres():
    # Cargar datos desde CSV
    df = pd.read_csv("DATASET_PROY_OFICIAL.csv", encoding="utf-8")
    
    # Crear conexión
    engine = create_engine(f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}")
    
    # Cargar datos a PostgreSQL
    df.to_sql('datos_uvb', engine, if_exists='replace', index=False)
    print("Datos cargados exitosamente a PostgreSQL")
    
    # Mostrar información de la tabla
    with engine.connect() as conn:
        result = pd.read_sql("SELECT COUNT(*) as total FROM datos_uvb", conn)
        count = result['total'].iloc[0]
        print(f"Total de registros cargados: {count}")

if __name__ == "__main__":
    cargar_datos_postgres()