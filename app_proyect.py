import os
import math
import pandas as pd
import numpy as np

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State

import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# -----------------------------------------------------------------------------
# Configuración inicial de la app (tomada del app_proyect.py original)
# -----------------------------------------------------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Dashboard del Proyecto Final"
server = app.server

# -----------------------------------------------------------------------------
# Ruta fija al dataset (mantuvimos la ruta absoluta que pidió el usuario)
# -----------------------------------------------------------------------------
DATA_PATH = "C:/Users/Administrador/Downloads/DATASET_PROY_OFICIAL.csv"

# -----------------------------------------------------------------------------
# Funciones auxiliares para cargar datos, generar gráficos interactivos y métricas
# -----------------------------------------------------------------------------
def load_data(path=DATA_PATH):
    """
    Carga el dataset desde la ruta fija proporcionada por el usuario.
    Devuelve un DataFrame o None si no se puede cargar.
    """
    try:
        df = pd.read_csv(path, encoding="utf-8")
        return df
    except Exception as e:
        print(f"Error cargando el archivo: {e}")
        return None

def summary_table(df):
    """
    Genera una tabla de estadísticas descriptivas (transpuesta) para mostrar en Dash.
    """
    desc = df.describe(include='all').transpose()
    desc.reset_index(inplace=True)
    desc.rename(columns={"index": "variable"}, inplace=True)
    return desc

def missing_values_table(df):
    """
    Tabla de valores faltantes por columna.
    """
    miss = df.isna().sum().reset_index()
    miss.columns = ["variable", "missing_count"]
    miss["missing_pct"] = (df.isna().sum().values / len(df)) * 100
    return miss

def detect_problem_type(df, target_column=None):
    """
    Intento heurístico de detectar si el problema es clasificación o regresión.
    Si no hay target, devuelve None.
    """
    if target_column is None or target_column not in df.columns:
        return None
    ser = df[target_column].dropna()
    # Si es numérico con muchas valores únicos -> regresión
    if pd.api.types.is_numeric_dtype(ser):
        nunique = ser.nunique()
        if nunique > 20:
            return "regression"
        else:
            # pocos valores únicos: posible clasificación si enteros/categoricos
            return "classification" if nunique <= 20 else "regression"
    else:
        return "classification"

def compute_model_metrics(df, target_col, features, problem_type="regression"):
    """
    Entrena un modelo simple de RandomForest (regresión o clasificación) y devuelve
    métricas y predicciones para mostrar en la pestaña de resultados.
    Nota: esto es un auxiliar para reproducir resultados tipo 'última celda' si el notebook original
    no incluye un modelo serializado. Si el notebook trae columnas de predicción (p.ej 'y_pred'),
    preferimos usar esas columnas en la interfaz principal.
    """
    X = df[features].copy()
    y = df[target_col].copy()

    # Manejar categóricos simples
    X_enc = X.copy()
    for col in X_enc.select_dtypes(include=['object', 'category']).columns:
        X_enc[col] = LabelEncoder().fit_transform(X_enc[col].astype(str))

    # Si y es categórico, encode
    is_classification = problem_type == "classification"
    if is_classification and not pd.api.types.is_numeric_dtype(y):
        y = LabelEncoder().fit_transform(y.astype(str))

    # Drop rows with NA in features or target
    data = pd.concat([X_enc, y.reset_index(drop=True)], axis=1).dropna()
    if data.shape[0] < 10:
        return None  # no hay datos suficientes

    X_clean = data.iloc[:, :-1]
    y_clean = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42)

    if is_classification:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {}
    if is_classification:
        metrics['accuracy'] = accuracy_score(y_test, preds)
    else:
        metrics['rmse'] = math.sqrt(mean_squared_error(y_test, preds))
        metrics['mae'] = mean_absolute_error(y_test, preds)
        metrics['r2'] = r2_score(y_test, preds)

    results_df = X_test.copy()
    results_df['y_true'] = y_test.values
    results_df['y_pred'] = preds

    return {
        "model": model,
        "metrics": metrics,
        "results_df": results_df
    }

# -----------------------------------------------------------------------------
# Cargar datos (al iniciar la app)
# -----------------------------------------------------------------------------
df = load_data(DATA_PATH)

# -----------------------------------------------------------------------------
# Componentes de la interface: sub-tabs (metodologia y resultados) del app_proyect.py base
# -----------------------------------------------------------------------------
subtabs_metodologia = dcc.Tabs([
    dcc.Tab(label='a. Definición del Problema', children=[
        html.H4('a. Definición del Problema a Resolver'),
        html.Ul([
            html.Li('Tipo de problema: clasificación / regresión / agrupamiento / series de tiempo'),
            html.Li('Variable objetivo o de interés: Selección interactiva abajo'),
        ]),
        html.H5("Selección de variable objetivo (target)"),
        dcc.Dropdown(
            id='metodo-target-dropdown',
            options=[{"label": c, "value": c} for c in (df.columns.tolist() if df is not None else [])],
            placeholder="Selecciona la variable objetivo (si aplica)"
        ),
        html.Div(id='metodo-target-info', className='my-2'),
    ]),
    dcc.Tab(label='b. Preparación de Datos', children=[
        html.H4('b. Preparación de los Datos'),
        html.P("Aquí se resumen los pasos de limpieza y transformación. El panel interactivo permite ejecutar transformaciones básicas."),
        html.Ul([
            html.Li('Limpieza y transformación de datos (ej. tratamiento de NA, encoding)'),
            html.Li('División del dataset en entrenamiento y prueba o validación cruzada')
        ]),
        html.H5("Operaciones rápidas sobre los datos"),
        dbc.Row([
            dbc.Col(dbc.Button("Eliminar filas con NA", id="btn-dropna", color="danger", className="me-2"), width="auto"),
            dbc.Col(dbc.Button("Rellenar NA con mediana (numéricas)", id="btn-fillna-median", color="secondary"), width="auto"),
            dbc.Col(html.Div(id="prep-action-output"), width=True),
        ], className="my-2")
    ]),
    dcc.Tab(label='c. Selección del Modelo', children=[
        html.H4('c. Selección del Modelo o Algoritmo'),
        html.P("Aquí se indican modelos candidatos; hay un entrenamiento de ejemplo con RandomForest si eliges target y features."),
        html.Ul([
            html.Li('Modelo(s) seleccionados: RandomForest (ejemplo)'),
            html.Li('Justificación de la elección: robustez y buen rendimiento por defecto'),
            html.Li('Ecuación o representación matemática si aplica: (modelo de caja negra)')
        ]),
        html.H5("Entrenamiento rápido (demo)"),
        html.P("Selecciona target y features para ejecutar un entrenamiento de ejemplo:"),
        dcc.Dropdown(id="select-target", options=[{"label": c, "value": c} for c in (df.columns.tolist() if df is not None else [])],
                     placeholder="Selecciona la variable objetivo"),
        dcc.Dropdown(id="select-features", options=[{"label": c, "value": c} for c in (df.columns.tolist() if df is not None else [])],
                     multi=True, placeholder="Selecciona columnas predictoras"),
        dbc.Button("Entrenar modelo de ejemplo", id="btn-train-example", color="primary", className="my-2"),
        html.Div(id="train-output")
    ]),
    dcc.Tab(label='d. Evaluación del Modelo', children=[
        html.H4('d. Entrenamiento y Evaluación del Modelo'),
        html.Ul([
            html.Li('Proceso de entrenamiento: mostrado en el panel "Selección del Modelo"'),
            html.Li('Métricas de evaluación: RMSE, MAE, Accuracy, etc. (según tipo de problema)'),
            html.Li('Validación utilizada: hold-out 80/20 (ejemplo)')
        ]),
    ])
])

subtabs_resultados = dcc.Tabs([
    dcc.Tab(label='a. EDA', children=[
        html.H4('a. Análisis Exploratorio de Datos (EDA)'),
        html.P('Panel interactivo para explorar tablas, distribuciones y correlaciones.'),
        dbc.Row([
            dbc.Col([
                html.H5("Tabla de datos (primeras filas)"),
                dash_table.DataTable(
                    id='table-head',
                    columns=[{"name": c, "id": c} for c in (df.columns.tolist() if df is not None else [])],
                    data=(df.head(200).to_dict('records') if df is not None else []),
                    page_size=10,
                    style_table={'overflowX': 'auto'},
                )
            ], width=12)
        ]),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                html.Label("Variable para histograma"),
                dcc.Dropdown(id='hist-var', options=[{"label": c, "value": c} for c in (df.columns.tolist() if df is not None else [])], placeholder="Selecciona variable"),
                dcc.Graph(id='hist-graph'),
            ], md=6),
            dbc.Col([
                html.Label("X para scatter (selecciona dos)"),
                dcc.Dropdown(id='scatter-x', options=[{"label": c, "value": c} for c in (df.columns.tolist() if df is not None else [])], placeholder="X"),
                dcc.Dropdown(id='scatter-y', options=[{"label": c, "value": c} for c in (df.columns.tolist() if df is not None else [])], placeholder="Y"),
                dcc.Graph(id='scatter-graph'),
            ], md=6),
        ]),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                html.H6("Estadísticas descriptivas"),
                dash_table.DataTable(id='desc-table',
                                     columns=[{"name": i, "id": i} for i in ["variable", "count", "mean", "std", "min", "25%", "50%", "75%", "max"]],
                                     data=(summary_table(df).to_dict('records') if df is not None else []),
                                     page_size=10)
            ], md=6),
            dbc.Col([
                html.H6("Valores faltantes"),
                dash_table.DataTable(id='missing-table',
                                     columns=[{"name": i, "id": i} for i in ["variable", "missing_count", "missing_pct"]],
                                     data=(missing_values_table(df).to_dict('records') if df is not None else []),
                                     page_size=10)
            ], md=6)
        ]),
        html.Hr(),
        html.H6("Mapa de correlación (heatmap)"),
        dcc.Graph(id='corr-heatmap'),
    ]),
    dcc.Tab(label='b. EDA 2', children=[
        html.H4('b. EDA 2 - Análisis adicional'),
        html.P('Análisis exploratorios complementarios: segmentaciones, boxplots, histogramas comparativos o mapas si aplica.'),
        html.Label("Variable para boxplot (por categoría)"),
        dcc.Dropdown(id='box-cat', options=[{"label": c, "value": c} for c in (df.columns.tolist() if df is not None else [])], placeholder="Categoría"),
        dcc.Dropdown(id='box-num', options=[{"label": c, "value": c} for c in (df.columns.tolist() if df is not None else [])], placeholder="Numérica"),
        dcc.Graph(id='box-graph')
    ]),
    dcc.Tab(label='c. Visualización del Modelo', children=[
        html.H4('c. Visualización de Resultados del Modelo'),
        html.P('Aquí se mostrarán las métricas de evaluación del modelo y comparaciones valores reales vs predichos.'),
        html.H6("Si en su dataset existen columnas 'y_true' y 'y_pred' o 'target'/'pred', el sistema las detectará y mostrará métricas."),
        dcc.Dropdown(id='maybe-target', options=[{"label": c, "value": c} for c in (df.columns.tolist() if df is not None else [])],
                     placeholder="(Opcional) Selecciona la columna target real si existe"),
        dbc.Button("Mostrar resultados del modelo (usar columnas existentes si las hay, sino entrenar demo)", id="btn-show-model", className="my-2"),
        html.Div(id='model-output'),
        dcc.Graph(id='model-scatter')
    ]),
    dcc.Tab(label='d. Indicadores del Modelo', children=[
        html.H4('d. Indicadores de Evaluación del Modelo'),
        html.Ul([
            html.Li(' Tabla de errores: RMSE, MAE, MSE, etc.'),
            html.Li(' Interpretación de los valores para comparar modelos')
        ]),
        html.Div(id='model-metrics-table')
    ]),
    dcc.Tab(label='e. Limitaciones', children=[
        html.H4('e. Limitaciones y Consideraciones Finales'),
        html.Ul([
            html.Li('Restricciones del análisis'),
            html.Li('Posibles mejoras futuras')
        ])
    ])
])

# -----------------------------------------------------------------------------
# Pestañas principales (manteniendo estructura original)
# -----------------------------------------------------------------------------
tabs = [
    dcc.Tab(label='1. Introducción', children=[
        html.H2('Introducción'),
        html.P('Aquí se presenta una visión general del contexto de la problemática, el análisis realizado y los hallazgos encontrados.'),
        html.P('De manera resumida, indicar lo que se pretende lograr con el proyecto'),
        html.Hr(),
        html.H5("Contenido del notebook (Introducción)"),
        dcc.Textarea(id='intro-textarea', value='(El texto de la introducción debe copiarse aquí desde Visualiza_python1.ipynb)', style={'width': '100%', 'height': 200})
    ]),
    dcc.Tab(label='2. Contexto', children=[
        html.H2('Contexto'),
        html.P('Descripción breve del contexto del proyecto.'),
        html.Ul([
            html.Li('Fuente de los datos: Nombre de la fuente'),
            html.Li('Variables de interés: listar variables-operacionalización')
        ]),
        html.Hr(),
        dcc.Textarea(id='context-textarea', value='(Contenido de contexto extraído del notebook)', style={'width': '100%', 'height': 200})
    ]),
    dcc.Tab(label='3. Planteamiento del Problema', children=[
        html.H2('Planteamiento del Problema'),
        dcc.Textarea(id='planteamiento-textarea', value='(Redacta aquí el planteamiento del problema presente en Visualiza_python1.ipynb)', style={'width': '100%', 'height': 200})
    ]),
    dcc.Tab(label='4. Objetivos y Justificación', children=[
        html.H2('Objetivos y Justificación'),
        html.H4('Objetivo General'),
        dcc.Textarea(id='obj-general', value='(Objetivo general del proyecto)', style={'width': '100%', 'height': 100}),
        html.H4('Objetivos Específicos'),
        dcc.Textarea(id='obj-especificos', value='- Objetivo 1\n- Objetivo 2\n- Objetivo 3', style={'width': '100%', 'height': 120}),
        html.H4('Justificación'),
        dcc.Textarea(id='justificacion', value='(Justificación extraída del notebook)', style={'width': '100%', 'height': 120})
    ]),
    dcc.Tab(label='5. Marco Teórico', children=[
        html.H2('Marco Teórico'),
        dcc.Textarea(id='marco-textarea', value='(Resumen de conceptos teóricos clave)', style={'width': '100%', 'height': 300})
    ]),
    dcc.Tab(label='6. Metodología', children=[
        html.H2('Metodología'),
        subtabs_metodologia
    ]),
    dcc.Tab(label='7. Resultados y Análisis Final', children=[
        html.H2('Resultados y Análisis Final'),
        subtabs_resultados
    ]),
    dcc.Tab(label='8. Conclusiones', children=[
        html.H2('Conclusiones'),
        html.P('(Se dejará vacía según indicación del usuario)')
    ])
]

# -----------------------------------------------------------------------------
# Layout principal del app
# -----------------------------------------------------------------------------
app.layout = dbc.Container([
    html.H1("Dashboard del Proyecto Final", className="text-center my-4"),
    dcc.Tabs(tabs)
], fluid=True)

# -----------------------------------------------------------------------------
# Callbacks para interactividad EDA y Modelado
# -----------------------------------------------------------------------------
@app.callback(
    Output('hist-graph', 'figure'),
    Input('hist-var', 'value')
)
def update_histogram(var):
    if df is None or var is None:
        return go.Figure()
    try:
        fig = px.histogram(df, x=var, nbins=40, title=f'Histograma de {var}')
        return fig
    except Exception:
        return go.Figure()

@app.callback(
    Output('scatter-graph', 'figure'),
    Input('scatter-x', 'value'),
    Input('scatter-y', 'value')
)
def update_scatter(x, y):
    if df is None or x is None or y is None:
        return go.Figure()
    try:
        fig = px.scatter(df, x=x, y=y, title=f'Scatter: {x} vs {y}', trendline="ols")
        return fig
    except Exception:
        return go.Figure()

@app.callback(
    Output('desc-table', 'data'),
    Output('desc-table', 'columns'),
    Input('table-head', 'data')  # trigger once loaded
)
def update_desc_table(_):
    if df is None:
        return [], []
    desc = summary_table(df)
    # Keep only typical stats columns for display
    cols = [{"name": i, "id": i} for i in desc.columns]
    return desc.to_dict('records'), cols

@app.callback(
    Output('missing-table', 'data'),
    Input('table-head', 'data')
)
def update_missing_table(_):
    if df is None:
        return []
    miss = missing_values_table(df)
    return miss.to_dict('records')

@app.callback(
    Output('corr-heatmap', 'figure'),
    Input('table-head', 'data')
)
def update_corr(_):
    if df is None:
        return go.Figure()
    # Calculamos correlación solo para numéricas
    try:
        num = df.select_dtypes(include=np.number)
        if num.shape[1] == 0:
            return go.Figure()
        corr = num.corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", title="Matriz de correlación (numéricas)")
        return fig
    except Exception:
        return go.Figure()

@app.callback(
    Output('box-graph', 'figure'),
    Input('box-cat', 'value'),
    Input('box-num', 'value')
)
def update_box(cat, num):
    if df is None or cat is None or num is None:
        return go.Figure()
    try:
        fig = px.box(df, x=cat, y=num, points="all", title=f'Boxplot de {num} por {cat}')
        return fig
    except Exception:
        return go.Figure()

# Preparación de datos - botones
@app.callback(
    Output('prep-action-output', 'children'),
    Input('btn-dropna', 'n_clicks'),
    Input('btn-fillna-median', 'n_clicks'),
    State('prep-action-output', 'children'),
    prevent_initial_call=True
)
def prep_actions(n_dropna, n_fill, current):
    ctx = dash.callback_context
    global df
    if not ctx.triggered:
        return ""
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if df is None:
        return "No hay dataset cargado."
    if button_id == "btn-dropna":
        df = df.dropna()
        return "Filas con NA eliminadas."
    elif button_id == "btn-fillna-median":
        num_cols = df.select_dtypes(include=np.number).columns
        for c in num_cols:
            median = df[c].median()
            df[c] = df[c].fillna(median)
        return "NAs numéricos rellenados con mediana."
    return ""

# Entrenamiento de ejemplo (desde pestaña metodología)
@app.callback(
    Output('train-output', 'children'),
    Input('btn-train-example', 'n_clicks'),
    State('select-target', 'value'),
    State('select-features', 'value'),
    prevent_initial_call=True
)
def train_example(n_clicks, target, features):
    if df is None:
        return html.Div("No hay dataset cargado en la ruta especificada.")
    if target is None or not features:
        return html.Div("Selecciona target y al menos una feature.")
    # Detectar tipo problema
    prob_type = detect_problem_type(df, target)
    res = compute_model_metrics(df, target, features, problem_type=prob_type or "regression")
    if res is None:
        return html.Div("No se pudo entrenar el modelo con los datos seleccionados (filtrado por NaN o pocos datos).")
    metrics = res["metrics"]
    metrics_list = [html.Li(f"{k}: {v}") for k, v in metrics.items()]
    return html.Div([
        html.H6("Métricas (ejemplo)"),
        html.Ul(metrics_list),
        html.P("Se entrenó un RandomForest de demostración. Revisa la pestaña 'Resultados' para ver comparaciones.")
    ])

# Mostrar resultados del modelo (usar columnas existentes o entrenar demo)
@app.callback(
    Output('model-output', 'children'),
    Output('model-scatter', 'figure'),
    Output('model-metrics-table', 'children'),
    Input('btn-show-model', 'n_clicks'),
    State('maybe-target', 'value'),
    prevent_initial_call=True
)
def show_model_results(n_clicks, maybe_target):
    if df is None:
        return html.Div("No hay dataset cargado."), go.Figure(), ""
    # Priorizar columnas existentes: si hay 'y_true' y 'y_pred' o 'y' y 'y_hat' etc.
    possible_pairs = [
        ('y_true', 'y_pred'),
        ('target', 'predicted'),
        ('y', 'y_pred'),
        ('y_true', 'predicted'),
        ('observed', 'predicted'),
    ]
    for true_col, pred_col in possible_pairs:
        if true_col in df.columns and pred_col in df.columns:
            y_true = df[true_col].dropna()
            paired = df[[true_col, pred_col]].dropna()
            fig = px.scatter(paired, x=true_col, y=pred_col, title="Valores reales vs predichos (from columns)")
            # métricas
            if pd.api.types.is_numeric_dtype(paired[true_col]):
                rmse = math.sqrt(mean_squared_error(paired[true_col], paired[pred_col]))
                mae = mean_absolute_error(paired[true_col], paired[pred_col])
                metrics_table = dbc.Table([
                    html.Thead(html.Tr([html.Th("Métrica"), html.Th("Valor")])),
                    html.Tbody([
                        html.Tr([html.Td("RMSE"), html.Td(f"{rmse:.4f}")]),
                        html.Tr([html.Td("MAE"), html.Td(f"{mae:.4f}")]),
                    ])
                ], bordered=True)
            else:
                metrics_table = html.P("No numeric target: no se calculan RMSE/MAE.")
            return html.Div([
                html.H6("Se detectaron columnas de verdad y predicción en el dataset:"),
                html.Ul([html.Li(f"Truth: {true_col}"), html.Li(f"Predicted: {pred_col}")])
            ]), fig, metrics_table

    # Si el usuario seleccionó una columna target, intentar usarla y entrenar demo con primeras 5 num features
    if maybe_target and maybe_target in df.columns:
        target = maybe_target
        # seleccionar features numéricas distintas al target
        candidate_feats = [c for c in df.select_dtypes(include=np.number).columns.tolist() if c != target]
        if len(candidate_feats) == 0:
            return html.Div("No se encontraron features numéricas para entrenar un modelo de ejemplo."), go.Figure(), ""
        features = candidate_feats[:5]
        prob_type = detect_problem_type(df, target)
        res = compute_model_metrics(df, target, features, problem_type=prob_type or "regression")
        if res is None:
            return html.Div("No se pudo entrenar modelo demo con los datos seleccionados."), go.Figure(), ""
        results_df = res["results_df"]
        fig = px.scatter(results_df, x='y_true', y='y_pred', title="Valores reales vs predichos (modelo demo)")
        # metrics
        m = res["metrics"]
        rows = []
        if prob_type == "classification":
            rows.append(html.Tr([html.Td("Accuracy"), html.Td(f"{m.get('accuracy', np.nan):.4f}")]))
        else:
            rows.extend([
                html.Tr([html.Td("RMSE"), html.Td(f"{m.get('rmse', np.nan):.4f}")]),
                html.Tr([html.Td("MAE"), html.Td(f"{m.get('mae', np.nan):.4f}")]),
                html.Tr([html.Td("R2"), html.Td(f"{m.get('r2', np.nan):.4f}")]),
            ])
        metrics_table = dbc.Table([
            html.Thead(html.Tr([html.Th("Métrica"), html.Th("Valor")])),
            html.Tbody(rows)
        ], bordered=True)
        return html.Div([
            html.H6("Resultados del modelo demo:"),
            html.P(f"Target: {target}. Features usadas: {', '.join(features)}.")
        ]), fig, metrics_table

    # Si no hay columnas de predicción ni target seleccionado -> intentar detectar si hay columnas sovrepred
    # También ofrecer entrenar demo con primer num column como target (heurístico)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) >= 2:
        target = numeric_cols[0]
        features = numeric_cols[1:6] if len(numeric_cols) > 1 else []
        res = compute_model_metrics(df, target, features, problem_type="regression")
        if res is None:
            return html.Div("No fue posible entrenar un modelo demo automáticamente."), go.Figure(), ""
        results_df = res["results_df"]
        fig = px.scatter(results_df, x='y_true', y='y_pred', title=f"Valores reales vs predichos (modelo demo, target={target})")
        m = res["metrics"]
        metrics_table = dbc.Table([
            html.Thead(html.Tr([html.Th("Métrica"), html.Th("Valor")])),
            html.Tbody([
                html.Tr([html.Td("RMSE"), html.Td(f"{m.get('rmse', np.nan):.4f}")]),
                html.Tr([html.Td("MAE"), html.Td(f"{m.get('mae', np.nan):.4f}")]),
                html.Tr([html.Td("R2"), html.Td(f"{m.get('r2', np.nan):.4f}")]),
            ])
        ], bordered=True)
        return html.Div([
            html.H6("Se entrenó un modelo demo automáticamente usando columnas numéricas."),
            html.P(f"Target automático: {target}. Features: {', '.join(features)}")
        ]), fig, metrics_table

    return html.Div("No se detectaron columnas de predicción en el dataset y no hay suficientes columnas numéricas para entrenar un modelo de ejemplo."), go.Figure(), ""

# -----------------------------------------------------------------------------
# Run server
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    # Nota: debug=False para despliegue, según app_proyect.py original
    app.run_server(debug=False, host="0.0.0.0", port=port)
