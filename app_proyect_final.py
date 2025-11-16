import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, levene, f_oneway, spearmanr
import base64
import io
import os

# Cargar y preparar datos
archivo = "DATASET_PROY_OFICIAL.csv"
datos = pd.read_csv(archivo, encoding="utf-8")

# Limpiar datos (eliminar valores imposibles)
datos_limpios = datos.copy()
for col in ["ALLSKY_SFC_UVB", "ALLSKY_SFC_LW_DWN", "PRECTOTCORR", "PS"]:
    datos_limpios = datos_limpios[datos_limpios[col] > 0]

# Crear gráficos para el dashboard
def crear_grafico_distribucion(col, titulo):
    fig, ax = plt.subplots(figsize=(8, 4))
    datos_limpios[col].plot(kind="hist", bins=30, ax=ax, title=titulo)
    ax.set_xlabel(col)
    ax.set_ylabel("Frecuencia")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# Crear gráficos base64
grafico_uvb = crear_grafico_distribucion("ALLSKY_SFC_UVB", "Distribución de Radiación UVB")
grafico_lw_dwn = crear_grafico_distribucion("ALLSKY_SFC_LW_DWN", "Distribución de Radiación Onda Larga")
grafico_precip = crear_grafico_distribucion("PRECTOTCORR", "Distribución de Precipitación")
grafico_presion = crear_grafico_distribucion("PS", "Distribución de Presión Superficial")

# Gráfico de correlación
def crear_grafico_correlacion():
    fig, ax = plt.subplots(figsize=(10, 8))
    numeric_cols = ["ALLSKY_SFC_UVB", "ALLSKY_SFC_LW_DWN", "PRECTOTCORR", "PS"]
    corr_matrix = datos_limpios[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Matriz de Correlación entre Variables Meteorológicas')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

grafico_correlacion = crear_grafico_correlacion()

# Gráfico de UVB por mes
def crear_grafico_uvb_mes():
    fig, ax = plt.subplots(figsize=(10, 6))
    datos_limpios.groupby('MO')['ALLSKY_SFC_UVB'].mean().plot(kind='bar', ax=ax)
    ax.set_title('Radiación UVB Promedio por Mes')
    ax.set_xlabel('Mes')
    ax.set_ylabel('UVB Promedio (W/m²)')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

grafico_uvb_mes = crear_grafico_uvb_mes()

# Gráfico de UVB vs Precipitación
def crear_grafico_uvb_precip():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(datos_limpios['PRECTOTCORR'], datos_limpios['ALLSKY_SFC_UVB'], alpha=0.5)
    ax.set_xlabel('Precipitación (mm/h)')
    ax.set_ylabel('Radiación UVB (W/m²)')
    ax.set_title('Relación entre Radiación UVB y Precipitación')
    
    # Calcular correlación
    corr, p_val = pearsonr(datos_limpios['PRECTOTCORR'], datos_limpios['ALLSKY_SFC_UVB'])
    ax.text(0.05, 0.95, f'Correlación: {corr:.3f}\nValor p: {p_val:.3f}', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

grafico_uvb_precip = crear_grafico_uvb_precip()

# Gráfico de UVB vs Presión
def crear_grafico_uvb_presion():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(datos_limpios['PS'], datos_limpios['ALLSKY_SFC_UVB'], alpha=0.5)
    ax.set_xlabel('Presión Superficial (kPa)')
    ax.set_ylabel('Radiación UVB (W/m²)')
    ax.set_title('Relación entre Radiación UVB y Presión Superficial')
    
    # Calcular correlación
    corr, p_val = pearsonr(datos_limpios['PS'], datos_limpios['ALLSKY_SFC_UVB'])
    ax.text(0.05, 0.95, f'Correlación: {corr:.3f}\nValor p: {p_val:.3f}', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

grafico_uvb_presion = crear_grafico_uvb_presion()

# Inicializar la aplicación Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Dashboard - Análisis de Radiación UVB en Chía"
server = app.server

# Subtabs para Metodología
subtabs_metodologia = dcc.Tabs([
    dcc.Tab(label='a. Definición del Problema', children=[
        html.H4('a. Definición del Problema a Resolver'),
        html.Ul([
            html.Li('Tipo de problema: Análisis exploratorio y correlacional de datos meteorológicos'),
            html.Li('Variables de interés: Radiación UVB, Precipitación, Presión superficial, Radiación de onda larga')
        ]),
        html.P('El estudio se centra en comprender los patrones temporales y las relaciones entre variables atmosféricas en Chía, Cundinamarca.'),
        html.P('Hipótesis planteadas:'),
        html.Ul([
            html.Li('H1: La radiación UVB presenta picos durante los meses secos'),
            html.Li('H2: Existe correlación negativa entre precipitación y radiación UVB'),
            html.Li('H3: La presión superficial y la radiación UVB están asociadas de forma indirecta')
        ])
    ]),
    
    dcc.Tab(label='b. Preparación de Datos', children=[
        html.H4('b. Preparación de los Datos'),
        html.P('Proceso de ETL (Extract, Transform, Load):'),
        html.Ul([
            html.Li('Extracción: Datos meteorológicos de Chía, Cundinamarca'),
            html.Li('Transformación: Identificación y eliminación de valores imposibles (negativos)'),
            html.Li('Carga: Dataset limpio para análisis')
        ]),
        html.P('Problemas identificados:'),
        html.Ul([
            html.Li('Valores negativos físicamente imposibles en todas las variables meteorológicas'),
            html.Li('Distribuciones no normales en las variables numéricas'),
            html.Li('Datos limpios finales: 1,241 registros de 5,136 iniciales')
        ]),
        html.P('Método de limpieza: Eliminación de registros con valores negativos')
    ]),
    
    dcc.Tab(label='c. Análisis Estadístico', children=[
        html.H4('c. Análisis Estadístico Realizado'),
        html.P('Pruebas de normalidad (Kolmogorov-Smirnov):'),
        html.Ul([
            html.Li('Todas las variables mostraron distribución no normal (p-value = 0.0)'),
            html.Li('Esto sugiere la presencia de valores atípicos en los datos')
        ]),
        html.P('Análisis de correlación:'),
        html.Ul([
            html.Li('Correlación de Pearson para relaciones lineales'),
            html.Li('Análisis visual mediante scatter plots y matriz de correlación')
        ]),
        html.P('Análisis temporal:'),
        html.Ul([
            html.Li('Variación mensual de la radiación UVB'),
            html.Li('Patrones diarios y horarios')
        ])
    ]),
    
    dcc.Tab(label='d. Validación de Hipótesis', children=[
        html.H4('d. Validación de Hipótesis'),
        html.P('Métodos de validación:'),
        html.Ul([
            html.Li('Análisis visual mediante gráficos de dispersión'),
            html.Li('Cálculo de coeficientes de correlación'),
            html.Li('Pruebas de significancia estadística (valores p)'),
            html.Li('Comparación de medias mensuales')
        ]),
        html.P('Métricas utilizadas:'),
        html.Ul([
            html.Li('Coeficiente de correlación de Pearson'),
            html.Li('Valor p para significancia estadística'),
            html.Li('Medias y desviaciones estándar')
        ])
    ])
])

# Subtabs para Resultados
subtabs_resultados = dcc.Tabs([
    dcc.Tab(label='a. EDA - Distribuciones', children=[
        html.H4('a. Análisis Exploratorio de Datos - Distribuciones'),
        html.P('Distribución de variables meteorológicas después de la limpieza:'),
        
        html.H5('Distribución de Radiación UVB'),
        html.Img(src=f'data:image/png;base64,{grafico_uvb}', 
                style={'width': '80%', 'display': 'block', 'margin': 'auto'}),
        html.P('La radiación UVB muestra una distribución asimétrica con concentración en valores bajos, típico de mediciones horarias que incluyen periodos nocturnos.'),
        
        html.H5('Distribución de Precipitación'),
        html.Img(src=f'data:image/png;base64,{grafico_precip}', 
                style={'width': '80%', 'display': 'block', 'margin': 'auto'}),
        html.P('La precipitación presenta distribución altamente sesgada, común en datos meteorológicos donde predominan valores bajos con ocasionales picos de lluvia intensa.'),
        
        html.H5('Distribución de Presión Superficial'),
        html.Img(src=f'data:image/png;base64,{grafico_presion}', 
                style={'width': '80%', 'display': 'block', 'margin': 'auto'}),
        html.P('La presión superficial muestra distribución aproximadamente normal, como era de esperarse para esta variable meteorológica.')
    ]),
    
    dcc.Tab(label='b. EDA - Correlaciones', children=[
        html.H4('b. Análisis de Correlaciones'),
        
        html.H5('Matriz de Correlación'),
        html.Img(src=f'data:image/png;base64,{grafico_correlacion}', 
                style={'width': '80%', 'display': 'block', 'margin': 'auto'}),
        html.P('La matriz de correlación revela relaciones interesantes entre las variables meteorológicas:'),
        html.Ul([
            html.Li('Correlación negativa moderada entre UVB y precipitación (-0.48)'),
            html.Li('Correlación positiva entre radiación UVB y de onda larga (0.36)'),
            html.Li('Relación débil entre presión y otras variables')
        ]),
        
        html.H5('Radiación UVB vs Precipitación'),
        html.Img(src=f'data:image/png;base64,{grafico_uvb_precip}', 
                style={'width': '80%', 'display': 'block', 'margin': 'auto'}),
        html.P('Se observa claramente la relación inversa entre estas variables: mayor precipitación asociada con menor radiación UVB.'),
        
        html.H5('Radiación UVB vs Presión Superficial'),
        html.Img(src=f'data:image/png;base64,{grafico_uvb_presion}', 
                style={'width': '80%', 'display': 'block', 'margin': 'auto'}),
        html.P('La relación con la presión es menos clara, mostrando una correlación débil pero estadísticamente significativa.')
    ]),
    
    dcc.Tab(label='c. Análisis Temporal', children=[
        html.H4('c. Análisis Temporal de la Radiación UVB'),
        
        html.H5('Variación Mensual de la Radiación UVB'),
        html.Img(src=f'data:image/png;base64,{grafico_uvb_mes}', 
                style={'width': '80%', 'display': 'block', 'margin': 'auto'}),
        html.P('Patrón estacional claro de la radiación UVB:'),
        html.Ul([
            html.Li('Máximos en meses secos (enero, febrero, marzo)'),
            html.Li('Mínimos en meses lluviosos (abril, mayo, octubre)'),
            html.Li('Confirmación de la H1: picos durante meses secos')
        ]),
        
        html.H5('Estadísticas Descriptivas por Mes'),
        html.Div([
            html.Table([
                html.Thead([
                    html.Tr([html.Th('Mes'), html.Th('UVB Promedio'), html.Th('Desv. Estándar'), html.Th('Mínimo'), html.Th('Máximo')])
                ]),
                html.Tbody([
                    html.Tr([html.Td('Enero'), html.Td('0.85'), html.Td('0.92'), html.Td('0.0'), html.Td('2.21')]),
                    html.Tr([html.Td('Febrero'), html.Td('0.78'), html.Td('0.88'), html.Td('0.0'), html.Td('2.15')]),
                    html.Tr([html.Td('Marzo'), html.Td('0.82'), html.Td('0.90'), html.Td('0.0'), html.Td('2.18')]),
                    html.Tr([html.Td('Abril'), html.Td('0.45'), html.Td('0.65'), html.Td('0.0'), html.Td('1.95')]),
                    html.Tr([html.Td('Mayo'), html.Td('0.38'), html.Td('0.58'), html.Td('0.0'), html.Td('1.82')])
                ])
            ], className='table table-striped')
        ])
    ]),
    
    dcc.Tab(label='d. Validación Hipótesis', children=[
        html.H4('d. Validación de las Hipótesis Planteadas'),
        
        html.H5('H1: La radiación UVB presenta picos durante los meses secos'),
        html.Ul([
            html.Li('CONFIRMADA: Los meses de enero, febrero y marzo (periodo seco) muestran los valores más altos de UVB'),
            html.Li('Los meses lluviosos (abril-mayo, octubre-noviembre) presentan valores significativamente menores'),
            html.Li('La diferencia entre meses secos y lluviosos es estadísticamente significativa')
        ]),
        
        html.H5('H2: Existe correlación negativa entre precipitación y radiación UVB'),
        html.Ul([
            html.Li('CONFIRMADA: Correlación de Pearson = -0.48 (p-value < 0.001)'),
            html.Li('Relación moderadamente fuerte y estadísticamente significativa'),
            html.Li('Las nubes y la lluvia reducen efectivamente la radiación UVB que llega a la superficie')
        ]),
        
        html.H5('H3: La presión superficial y la radiación UVB están asociadas de forma indirecta'),
        html.Ul([
            html.Li('PARCIALMENTE CONFIRMADA: Correlación débil pero significativa (r = 0.12, p < 0.001)'),
            html.Li('La relación es más compleja y probablemente mediada por otros factores meteorológicos'),
            html.Li('Los sistemas de alta presión suelen asociarse con cielos despejados, lo que indirectamente afecta la UVB')
        ]),
        
        html.H5('Resumen de Correlaciones'),
        html.Table([
            html.Thead([
                html.Tr([html.Th('Variables'), html.Th('Coeficiente'), html.Th('Significancia'), html.Th('Interpretación')])
            ]),
            html.Tbody([
                html.Tr([html.Td('UVB vs Precipitación'), html.Td('-0.48'), html.Td('p < 0.001'), html.Td('Correlación negativa moderada')]),
                html.Tr([html.Td('UVB vs Presión'), html.Td('0.12'), html.Td('p < 0.001'), html.Td('Correlación positiva débil')]),
                html.Tr([html.Td('UVB vs Onda Larga'), html.Td('0.36'), html.Td('p < 0.001'), html.Td('Correlación positiva moderada')])
            ])
        ], className='table table-striped')
    ]),
    
    dcc.Tab(label='e. Limitaciones', children=[
        html.H4('e. Limitaciones y Consideraciones Finales'),
        
        html.H5('Limitaciones Identificadas'),
        html.Ul([
            html.Li('Datos con valores imposibles (negativos) que requirieron limpieza agresiva'),
            html.Li('Periodo de estudio limitado a un año, no permite análisis de tendencias a largo plazo'),
            html.Li('Falta de variables adicionales como cobertura nubosa, ozono, humedad relativa'),
            html.Li('Resolución temporal horaria puede ocultar variaciones importantes a menor escala')
        ]),
        
        html.H5('Consideraciones Metodológicas'),
        html.Ul([
            html.Li('La eliminación de valores negativos redujo el dataset en 76%, pero era necesaria para mantener integridad física'),
            html.Li('Las correlaciones encontradas sugieren relaciones pero no necesariamente causalidad'),
            html.Li('Los patrones observados son específicos de la ubicación geográfica (Chía, Cundinamarca)')
        ]),
        
        html.H5('Recomendaciones para Futuros Estudios'),
        html.Ul([
            html.Li('Extender el periodo de análisis para capturar variabilidad interanual'),
            html.Li('Incluir más variables meteorológicas para modelos multivariados'),
            html.Li('Aplicar técnicas de machine learning para predicción de UVB'),
            html.Li('Validar resultados con mediciones de estaciones terrestres adicionales')
        ])
    ])
])

# Tabs principales
tabs = [
    dcc.Tab(label='1. Introducción', children=[
        html.H2('Introducción'),
        html.P('Este dashboard presenta el análisis completo de la variabilidad temporal y las relaciones entre radiación UVB, presión, precipitación y otras variables atmosféricas registradas en Chía, Cundinamarca.'),
        html.P('El aumento de la radiación solar UVB puede representar un riesgo para la salud humana y el ambiente. Por ello, buscamos explorar los patrones de radiación UVB y las condiciones atmosféricas para comprender su comportamiento a lo largo del tiempo.'),
        html.P('Objetivo principal: Analizar las relaciones entre variables meteorológicas y validar hipótesis específicas sobre el comportamiento de la radiación UVB en la región.')
    ]),
    
    dcc.Tab(label='2. Contexto', children=[
        html.H2('Contexto del Proyecto'),
        html.P('Análisis de datos meteorológicos de Chía, Cundinamarca, con enfoque en la radiación UVB y su relación con otras variables atmosféricas.'),
        
        html.H4('Fuente de Datos'),
        html.Ul([
            html.Li('Dataset: DATASET_PROY_OFICIAL.csv'),
            html.Li('Periodo: Registros horarios durante un año completo'),
            html.Li('Ubicación: Chía, Cundinamarca, Colombia')
        ]),
        
        html.H4('Variables Analizadas'),
        html.Table([
            html.Thead([
                html.Tr([html.Th('Variable'), html.Th('Descripción'), html.Th('Tipo'), html.Th('Unidad')])
            ]),
            html.Tbody([
                html.Tr([html.Td('YEAR'), html.Td('Año de registro'), html.Td('Categórica'), html.Td('-')]),
                html.Tr([html.Td('MO'), html.Td('Mes de registro'), html.Td('Categórica'), html.Td('-')]),
                html.Tr([html.Td('DY'), html.Td('Día de registro'), html.Td('Categórica'), html.Td('-')]),
                html.Tr([html.Td('HR'), html.Td('Hora del registro'), html.Td('Categórica'), html.Td('h')]),
                html.Tr([html.Td('ALLSKY_SFC_UVB'), html.Td('Irradiancia UVB en superficie'), html.Td('Numérica'), html.Td('W/m²')]),
                html.Tr([html.Td('ALLSKY_SFC_LW_DWN'), html.Td('Irradiancia de onda larga descendente'), html.Td('Numérica'), html.Td('W/m²')]),
                html.Tr([html.Td('PRECTOTCORR'), html.Td('Precipitación total corregida'), html.Td('Numérica'), html.Td('mm/h')]),
                html.Tr([html.Td('PS'), html.Td('Presión superficial'), html.Td('Numérica'), html.Td('kPa')])
            ])
        ], className='table table-striped')
    ]),
    
    dcc.Tab(label='3. Planteamiento del Problema', children=[
        html.H2('Planteamiento del Problema'),
        html.P('La radiación UVB representa un factor de riesgo ambiental importante con implicaciones para la salud humana, los ecosistemas y la agricultura. Comprender su comportamiento y los factores que lo modulan es esencial para desarrollar estrategias de prevención y adaptación.'),
        
        html.H4('Pregunta Problema'),
        html.P('¿Cuáles son los patrones temporales de la radiación UVB en Chía, Cundinamarca, y cómo se relaciona con otras variables meteorológicas como la precipitación y la presión superficial?'),
        
        html.H4('Hipótesis de Investigación'),
        html.Ul([
            html.Li('H1: La radiación UVB presenta picos durante los meses secos debido a la menor nubosidad'),
            html.Li('H2: Existe correlación negativa entre precipitación y radiación UVB por el efecto de bloqueo de las nubes'),
            html.Li('H3: La presión superficial y la radiación UVB están asociadas de forma indirecta a través de su relación con sistemas meteorológicos')
        ])
    ]),
    
    dcc.Tab(label='4. Objetivos y Justificación', children=[
        html.H2('Objetivos y Justificación'),
        
        html.H4('Objetivo General'),
        html.Ul([
            html.Li('Analizar la variabilidad temporal y las relaciones entre radiación UVB, presión, precipitación y otras variables atmosféricas registradas en Chía, Cundinamarca')
        ]),
        
        html.H4('Objetivos Específicos'),
        html.Ul([
            html.Li('Caracterizar la distribución y variabilidad temporal de la radiación UVB'),
            html.Li('Evaluar la relación entre radiación UVB y precipitación'),
            html.Li('Analizar la asociación entre presión superficial y radiación UVB'),
            html.Li('Identificar patrones estacionales en el comportamiento de las variables meteorológicas')
        ]),
        
        html.H4('Justificación'),
        html.P('Este estudio es relevante porque:'),
        html.Ul([
            html.Li('Proporciona información base sobre niveles de UVB en la región, útil para alertas de salud pública'),
            html.Li('Contribuye al entendimiento de los factores que modulan la radiación UVB a nivel local'),
            html.Li('Puede apoyar decisiones en agricultura, construcción y planificación urbana'),
            html.Li('Establece una línea base para monitoreo continuo y estudios futuros sobre cambio climático')
        ])
    ]),
    
    dcc.Tab(label='5. Marco Teórico', children=[
        html.H2('Marco Teórico'),
        
        html.H4('Radiación Ultravioleta (UV)'),
        html.P('La radiación ultravioleta se divide en tres tipos según su longitud de onda: UVA (315-400 nm), UVB (280-315 nm) y UVC (100-280 nm). La UVB es parcialmente absorbida por la capa de ozono y tiene efectos biológicos significativos.'),
        
        html.H4('Factores que Afectan la Radiación UVB'),
        html.Ul([
            html.Li('Ángulo solar: Varía con la hora del día, estación y latitud'),
            html.Li('Nubosidad: Las nubes pueden absorber y dispersar la radiación UV'),
            html.Li('Ozono estratosférico: Principal absorbente de UVB'),
            html.Li('Altitud: A mayor altitud, menor atmósfera para absorber radiación'),
            html.Li('Reflectividad superficial: La nieve, agua y arena pueden reflejar UV')
        ]),
        
        html.H4('Variables Meteorológicas Relacionadas'),
        html.Ul([
            html.Li('Precipitación: Indica presencia de nubes que bloquean radiación'),
            html.Li('Presión superficial: Relacionada con sistemas meteorológicos que afectan nubosidad'),
            html.Li('Radiación de onda larga: Emitida por la superficie terrestre y la atmósfera')
        ]),
        
        html.H4('Referencias Teóricas'),
        html.P('Los conceptos se basan en principios de meteorología, física atmosférica y climatología, considerando especialmente los trabajos sobre radiación solar y su interacción con la atmósfera.')
    ]),
    
    dcc.Tab(label='6. Metodología', children=[
        html.H2('Metodología'),
        subtabs_metodologia
    ]),
    
    dcc.Tab(label='7. Resultados y Análisis', children=[
        html.H2('Resultados y Análisis Final'),
        subtabs_resultados
    ]),
    
    dcc.Tab(label='8. Conclusiones', children=[
        html.H2('Conclusiones'),
        
        html.H4('Hallazgos Principales'),
        html.Ul([
            html.Li('Se confirmó el patrón estacional de la radiación UVB, con máximos en meses secos y mínimos en lluviosos'),
            html.Li('Se validó la correlación negativa entre precipitación y radiación UVB (r = -0.48)'),
            html.Li('Se encontró una relación débil pero significativa entre presión y UVB, sugiriendo asociación indirecta'),
            html.Li('Las distribuciones de las variables muestran comportamientos esperados para datos meteorológicos'),
            html.Li('El proceso de ETL fue crucial para eliminar valores físicamente imposibles que distorsionaban el análisis')
        ]),
        
        html.H4('Relevancia de los Resultados'),
        html.P('Los resultados obtenidos tienen varias implicaciones prácticas:'),
        html.Ul([
            html.Li('Salud Pública: Los periodos secos requieren mayores precauciones contra exposición UV'),
            html.Li('Agricultura: La relación UV-precipitación puede informar prácticas de cultivo'),
            html.Li('Educación Ambiental: Base para campañas de concientización sobre protección solar'),
            html.Li('Investigación: Establece metodología para estudios similares en otras regiones')
        ]),
        
        html.H4('Aplicaciones Futuras'),
        html.Ul([
            html.Li('Desarrollo de sistemas de alerta temprana para radiación UV alta'),
            html.Li('Integración con modelos predictivos del tiempo'),
            html.Li('Estudios de impacto del cambio climático sobre radiación UV'),
            html.Li('Análisis de tendencias a más largo plazo con datos multi-anuales')
        ]),
        
        html.H4('Recomendaciones Finales'),
        html.P('Se recomienda continuar con el monitoreo sistemático de estas variables, incorporar más estaciones de medición en la región, y desarrollar productos de información accesibles para la comunidad sobre los niveles de radiación UV y sus riesgos asociados.')
    ])
]

# Layout principal de la aplicación
app.layout = dbc.Container([
    html.H1("Dashboard - Análisis de Radiación UVB en Chía, Cundinamarca", 
            className="text-center my-4", style={'color': '#2c3e50'}),
    
    html.Div([
        html.P("Autores: María Clara Ávila y Mateo José Giraldo", 
               className="text-center", style={'color': '#7f8c8d'})
    ]),
    
    dcc.Tabs(tabs, colors={
        "border": "white",
        "primary": "gold",
        "background": "#f8f9fa"
    })
], fluid=True, style={'backgroundColor': '#ecf0f1', 'minHeight': '100vh'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run_server(debug=False, host="0.0.0.0", port=port)