# Dashboard - Análisis de Radiación UVB en Chía, Cundinamarca

## Descripción del Proyecto
Dashboard interactivo para el análisis de datos meteorológicos y radiación UVB en Chía, Cundinamarca. Este proyecto implementa un sistema completo de visualización, análisis estadístico y modelado predictivo.

## Objetivos
- **Analizar** la variabilidad temporal de la radiación UVB
- **Identificar** patrones estacionales y relaciones entre variables meteorológicas
- **Desarrollar** un modelo predictivo para clasificación de niveles de UVB
- **Implementar** un dashboard interactivo para visualización de resultados

## Tecnologías Utilizadas
- **Backend:** Python, Pandas, Scikit-learn, XGBoost
- **Frontend:** Dash, Bootstrap Components
- **Base de Datos:** PostgreSQL
- **Contenedores:** Docker, Docker Compose
- **Visualización:** Matplotlib, Seaborn, Plotly

## Estructura del Proyecto
entregable3/
├── app_proyect_1.py # Dashboard principal
├── cargar_postgres.py # Script de carga a PostgreSQL
├── consultas_postgres.py # Consultas de ejemplo
├── requirements.txt # Dependencias del proyecto
├── Dockerfile # Configuración Docker
├── docker-compose.yml # Orquestación de contenedores
├── Procfile # Configuración despliegue
├── assets/ # Recursos estáticos (imágenes)
│ ├── logo.png
│ ├── imagen1.jpeg
│ └── imagen2.jpeg
└── DATASET_PROY_OFICIAL.csv # Dataset original

text

## Instalación y Configuración

### Prerrequisitos
- Python 3.9+
- PostgreSQL 13+
- Docker y Docker Compose

### Instalación Local
# 1. Clonar o descargar el proyecto
git clone <repositorio>
cd entregable3

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar PostgreSQL local
# Crear base de datos y usuario según configuración en cargar_postgres.py

# 5. Cargar datos
python cargar_postgres.py

# 6. Ejecutar dashboard
python app_proyect_1.py
Instalación con Docker
bash
# Construir y ejecutar con Docker Compose
docker-compose up --build

# El dashboard estará disponible en: http://localhost:8050

Estructura Metodológica
1. Contexto
Análisis de radiación UVB en Chía, Cundinamarca utilizando datos meteorológicos de la NASA POWER.

2. Planteamiento del Problema
Problema de Investigación:
¿Cuáles son los patrones temporales de la radiación UVB en Chía y cómo se relaciona con la precipitación y la presión superficial?

Hipótesis:

H1: La radiación UVB presenta picos durante los meses secos

H2: Existe correlación negativa entre precipitación y radiación UVB

H3: La presión superficial y la radiación UVB están asociadas de forma indirecta

3. Objetivos y Justificación
Objetivo General:
Analizar la variabilidad temporal y las relaciones entre radiación UVB y variables atmosféricas.

Objetivos Específicos:

Caracterizar la distribución y variabilidad temporal de la radiación UVB

Evaluar la relación entre radiación UVB y precipitación

Analizar la asociación entre presión superficial y radiación UVB

Identificar patrones estacionales en el comportamiento de la radiación UVB

Justificación:
Comprender patrones de radiación UVB es importante para salud pública, agricultura y planificación urbana.

4. Marco Teórico
Radiación UVB: Radiación ultravioleta B (280-315 nm) parcialmente absorbida por la capa de ozono.

Variables Meteorológicas:

Presión superficial: Relacionada con sistemas meteorológicos y nubosidad

Precipitación: Factor atenuante de radiación UVB

Radiación de onda larga: Indicativa de procesos térmicos atmosféricos

5. Metodología
a. Definición del Problema
Tipo: Análisis exploratorio y modelado predictivo

Variable objetivo: Clasificación de niveles de UVB (High/Low)

Variables predictoras: Radiación onda larga, precipitación, presión superficial, variables temporales

b. Preparación de Datos
Limpieza: Eliminación de valores imposibles (<=0)

Transformación: Codificación de variables categóricas, escalado

División: 80% entrenamiento, 20% prueba con estratificación

c. Selección del Modelo
Algoritmo: XGBoost Classifier

Justificación: Alto rendimiento en problemas de clasificación, manejo de relaciones no lineales

Hyperparámetros optimizados: n_estimators, max_depth, learning_rate, subsample

d. Entrenamiento y Evaluación
Validación: Cross-validation estratificado (5 folds)

Métricas: Accuracy, F1-macro, Balanced Accuracy, AUC-ROC

Optimización: GridSearchCV para selección de hiperparámetros

### Resultados y Análisis
### Análisis Exploratorio (EDA)
Distribuciones asimétricas en variables meteorológicas

Correlación negativa moderada UVB-Precipitación (r = -0.48)

Patrón estacional claro con máximos en meses secos

Validación de Hipótesis
H1: CONFIRMADA - Picos en meses secos (Ene-Feb: 0.78-0.85 W/m²)

H2: CONFIRMADA - Correlación Pearson = -0.48 (p < 0.001)

H3: PARCIALMENTE CONFIRMADA - Correlación débil (r = 0.12)

Modelo Predictivo
Accuracy: 0.85

F1-Macro: 0.84

AUC-ROC: 0.92

Balanced Accuracy: 0.85

### Uso del Dashboard
Pestañas Disponibles
Introducción: Contexto y objetivos del proyecto

Metodología: Descripción del proceso analítico

Análisis Exploratorio: Distribuciones y correlaciones básicas

Análisis Temporal: Series temporales y análisis estacional

Correlaciones: Matrices de correlación extendidas

Validación: Resultados de pruebas de hipótesis

Modelado: Métricas y resultados del modelo XGBoost

Conclusiones: Hallazgos y recomendaciones

Características Interactivas
Filtros por meses y rango de UVB

Selector de variables para análisis

Panel de KPIs en tiempo real

Visualizaciones dinámicas

### Referencia Bibliográfica
Fuente de Datos:
bibtex
NASA POWER Data Access Viewer (2025). Prediction Of Worldwide Energy Resources.
Disponible en: https://power.larc.nasa.gov/data-access-viewer/
Consultado: Noviembre 2025
Plataforma: NASA POWER Data Access Viewer (DAV)
URL: https://power.larc.nasa.gov/data-access-viewer/
Módulo: Meteorología - Sustainable Buildings
Localización: Chía, Cundinamarca, Colombia
Dataset: DATASET_PROY_OFICIAL.csv - registros horarios 2025

Solución de Problemas
Error de Conexión PostgreSQL
bash
# Verificar que PostgreSQL esté ejecutándose
sudo systemctl status postgresql

# Crear usuario y base de datos manualmente
psql -U postgres -c "CREATE USER usuario_uvb WITH PASSWORD 'password123';"
psql -U postgres -c "CREATE DATABASE proyecto_uvb;"
Problemas con Docker
bash
# Limpiar contenedores previos
docker-compose down -v

# Reconstruir desde cero
docker-compose up --build


### Autores: María Clara Ávila y Mateo José Giraldo

Curso: Visualización de Datos
Institución: [Nombre de tu Institución]
Fecha: Noviembre 2025
Acceso al Dashboard: http://localhost:8050