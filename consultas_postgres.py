import psycopg2
import pandas as pd

DB_CONFIG = {
    'dbname': 'base',
    'user': 'postgres', 
    'password': 'vis2025..',
    'host': 'localhost',
    'port': '5432'
}

def ejemplo_consultas():
    conn = psycopg2.connect(**DB_CONFIG)
    
    # Consulta 1: Conteo básico
    query1 = "SELECT COUNT(*) as total_registros FROM datos_uvb;"
    
    # Consulta 2: Estadísticas de UVB
    query2 = """
    SELECT 
        AVG(ALLSKY_SFC_UVB) as uv_promedio,
        MIN(ALLSKY_SFC_UVB) as uv_minimo, 
        MAX(ALLSKY_SFC_UVB) as uv_maximo
    FROM datos_uvb 
    WHERE ALLSKY_SFC_UVB > 0;
    """
    
    # Consulta 3: Datos por mes
    query3 = """
    SELECT MO as mes, 
           AVG(ALLSKY_SFC_UVB) as uv_promedio_mes
    FROM datos_uvb 
    WHERE MO IS NOT NULL 
    GROUP BY MO 
    ORDER BY MO;
    """
    
    try:
        # Ejecutar consultas
        df1 = pd.read_sql(query1, conn)
        df2 = pd.read_sql(query2, conn) 
        df3 = pd.read_sql(query3, conn)
        
        print("RESULTADOS DE CONSULTAS EJEMPLO:")
        print("\n1. Total de registros:")
        print(df1)
        
        print("\n2. Estadísticas UVB:")
        print(df2)
        
        print("\n3. UVB promedio por mes:")
        print(df3)
        
    finally:
        conn.close()

if __name__ == "__main__":
    ejemplo_consultas()