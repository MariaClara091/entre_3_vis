# app_proyect_final.py
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

# ----------------------------
# CONFIG
# ----------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Dashboard del Proyecto Final"
server = app.server

DATA_PATH = "C:/Users/Administrador/Downloads/DATASET_PROY_OFICIAL.csv"

# ----------------------------
# UTILITIES
# ----------------------------
def load_data(path=DATA_PATH):
    try:
        df = pd.read_csv(path, encoding="utf-8")
        return df
    except Exception as e:
        print(f"Error cargando el archivo: {e}")
        return None

def summary_table(df):
    desc = df.describe(include='all').transpose()
    desc.reset_index(inplace=True)
    desc.rename(columns={"index": "variable"}, inplace=True)
    return desc

def missing_values_table(df):
    miss = df.isna().sum().reset_index()
    miss.columns = ["variable", "missing_count"]
    miss["missing_pct"] = (df.isna().sum().values / len(df)) * 100
    return miss

def detect_problem_type(df, target_column=None):
    if target_column is None or target_column not in df.columns:
        return None
    ser = df[target_column].dropna()
    if pd.api.types.is_numeric_dtype(ser):
        nunique = ser.nunique()
        if nunique > 20:
            return "regression"
        else:
            return "classification" if nunique <= 20 else "regression"
    else:
        return "classification"

def compute_model_metrics(df, target_col, features, problem_type="regression"):
    X = df[features].copy()
    y = df[target_col].copy()
    X_enc = X.copy()
    for col in X_enc.select_dtypes(include=['object', 'category']).columns:
        X_enc[col] = LabelEncoder().fit_transform(X_enc[col].astype(str))
    is_classification = problem_type == "classification"
    if is_classification and not pd.api.types.is_numeric_dtype(y):
        y = LabelEncoder().fit_transform(y.astype(str))
    data = pd.concat([X_enc, y.reset_index(drop=True)], axis=1).dropna()
    if data.shape[0] < 10:
        return None
    X_clean = data.iloc[:, :-1]
    y_clean = data.iloc[:, -1]
    # If classification and classes are imbalanced, stratify; otherwise no stratify.
    strat = y_clean if (is_classification and y_clean.nunique() > 1) else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.2, random_state=42, stratify=strat
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.2, random_state=42
        )
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
    return {"model": model, "metrics": metrics, "results_df": results_df}

# ----------------------------
# LOAD
# ----------------------------
df = load_data(DATA_PATH)

# Create safe copies for structure
if df is not None:
    # ensure YEAR, MO, DY, HR exist as strings/categoricals if present
    for c in ["YEAR", "MO", "DY", "HR"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
else:
    numeric_cols = []

# ----------------------------
# TEXT CONTENT (from notebook)
# ----------------------------
AUTHORS_TEXT = "Autores: María Clara Ávila y Mateo José Giraldo"

INTRO_TEXT = """
**Contexto de nuestro proyecto**  
El aumento de la radiación solar UVB puede representar un riesgo para la salud humana y el ambiente. Por ello, buscamos explorar los patrones de radiación UVB y las condiciones atmosféricas registradas en Chía, Cundinamarca para comprender su comportamiento a lo largo del tiempo.
"""

OBJECTIVE_TEXT = """
**Objetivo general:**  
Analizar la variabilidad temporal y las relaciones entre radiación UVB, presión, precipitación y otras variables atmosféricas.

**Hipótesis:**  
- H1: La radiación UVB presenta picos durante los meses secos.  
- H2: Existe correlación negativa entre precipitación y radiación UVB.  
- H3: La presión superficial y la radiación UVB están asociadas de forma indirecta.
"""

VARIABLES_MD = """
Tenemos las siguientes variables:

| Variable | Descripción | Tipo | Unidad |
|-----------|--------------|------|--------|
| YEAR | Año de registro | Categórica | - |
| MO | Mes de registro | Categórica | - |
| DY | Día de registro | Categórica | - |
| HR | Hora del registro | Categórica | h |
| ALLSKY_SFC_UVB | Irradiancia UVB en superficie bajo cielo total | Numérica | W/m² |
| ALLSKY_SFC_LW_DWN | Irradiancia de onda larga descendente | Numérica | W/m² |
| PRECTOTCORR | Precipitación total corregida | Numérica | mm/h |
| PS | Presión superficial | Numérica | kPa |
"""

ETL_TEXT = """
# ETL (Extract, Transform, Load)
En esta etapa del proyecto revisamos la estructura, limpieza y trazabilidad de los datos detectando valores faltantes, duplicados y explicamos decisiones de imputación o eliminación.
Se han identificado valores negativos físicamente imposibles en variables meteorológicas y se realizaron imputaciones basadas en IQR donde aplica.
"""

ANALYSIS_DESC_TEXT = """
# Análisis descriptivo
Después de la imputación de atípicos, las distribuciones muestran valores físicamente posibles y patrones coherentes con su asimetría esperada.
"""

RELATIONAL_TEXT = """
# Análisis Relacional
- Variación mensual de la radiación UVB: máximo alrededor de enero (primera temporada seca).
- Relación precipitación vs UVB: correlación negativa moderada (soporta H2).
- Relación presión vs UVB: correlación muy débil y positiva (r ≈ 0.079); H3 no apoyada.
- Hora del día vs UVB: correlación positiva moderada (mayor radiación en horas centrales).
- Día del año vs precipitación: correlación negativa fuerte, sugiere estacionalidad de lluvias.
"""

RESULTS_INTERPRETATION = """
# Estadísticas por mes y conclusiones parciales
- Enero muestra la mayor radiación UVB promedio (pico en temporada seca).
- La mediana de UVB igual a 0 todos los meses indica muchas horas de oscuridad o nublado.
- H1 y H2 corroboradas por los patrones mensuales; H3 es rechazada por ausencia de patrón claro.
"""

EVIDENCE_TEXT = """
# Evidencia analítica y visualización
Pruebas ANOVA y Levene se usaron para comparar niveles de UVB entre meses; Levene mostró heterogeneidad de varianzas. 
Correlaciones muestran una relación negativa débil pero significativa entre precipitación y UVB (H2), y una relación positiva débil entre presión y UVB (contradice H3).
También se observa correlación moderada entre UVB y radiación de onda larga.
"""

STRATFY_EXPLANATION = """
# Nota sobre particionado estratificado
La línea `train_test_split(..., stratify=y)` garantiza que la distribución de clases se mantenga idéntica entre train y test. 
Es buena práctica en clasificación con clases desbalanceadas para obtener evaluaciones más representativas.
"""

# ----------------------------
# LAYOUT: Tabs with content and interactive EDA
# ----------------------------
subtabs_metodologia = dcc.Tabs([
    dcc.Tab(label='a. Definición del Problema', children=[
        html.H4('a. Definición del Problema a Resolver'),
        html.P(OBJECTIVE_TEXT),
        html.P("Variable objetivo o de interés: seleccionar abajo para interactividad."),
        dcc.Dropdown(
            id='metodo-target-dropdown',
            options=[{"label": c, "value": c} for c in (df.columns.tolist() if df is not None else [])],
            placeholder="Selecciona la variable objetivo (si aplica)"
        ),
        html.Div(id='metodo-target-info', className='my-2'),
    ]),
    dcc.Tab(label='b. Preparación de Datos', children=[
        html.H4('b. Preparación de los Datos'),
        html.P(ETL_TEXT),
        html.H5("Operaciones rápidas sobre los datos"),
        dbc.Row([
            dbc.Col(dbc.Button("Eliminar filas con NA", id="btn-dropna", color="danger", className="me-2"), width="auto"),
            dbc.Col(dbc.Button("Rellenar NA con mediana (numéricas)", id="btn-fillna-median", color="secondary"), width="auto"),
            dbc.Col(html.Div(id="prep-action-output"), width=True),
        ], className="my-2")
    ]),
    dcc.Tab(label='c. Selección del Modelo', children=[
        html.H4('c. Selección del Modelo o Algoritmo'),
        html.P("Entrenamiento de ejemplo con RandomForest (demo)."),
        dcc.Dropdown(id="select-target", options=[{"label": c, "value": c} for c in (df.columns.tolist() if df is not None else [])],
                     placeholder="Selecciona la variable objetivo"),
        dcc.Dropdown(id="select-features", options=[{"label": c, "value": c} for c in (df.columns.tolist() if df is not None else [])],
                     multi=True, placeholder="Selecciona columnas predictoras"),
        dbc.Button("Entrenar modelo de ejemplo", id="btn-train-example", color="primary", className="my-2"),
        html.Div(id="train-output"),
        html.Hr(),
        dcc.Markdown(STRATFY_EXPLANATION)
    ]),
    dcc.Tab(label='d. Evaluación del Modelo', children=[
        html.H4('d. Entrenamiento y Evaluación del Modelo'),
        html.Ul([
            html.Li('Proceso de entrenamiento: mostrado en el panel "Selección del Modelo"'),
            html.Li('Métricas de evaluación: RMSE, MAE, Accuracy, etc.'),
            html.Li('Validación utilizada: hold-out 80/20 (ejemplo)')
        ]),
    ])
])

subtabs_resultados = dcc.Tabs([
    dcc.Tab(label='a. EDA', children=[
        html.H4('a. Análisis Exploratorio de Datos (EDA)'),
        html.P(AUTHORS_TEXT),
        dcc.Markdown(INTRO_TEXT),
        html.Hr(),
        # Filters
        dbc.Row([
            dbc.Col([
                html.Label("Filtrar: Año"),
                dcc.Dropdown(
                    id='filter-year',
                    options=[{"label": y, "value": y} for y in (sorted(df['YEAR'].unique()) if df is not None and 'YEAR' in df.columns else [])],
                    placeholder="Todos",
                    multi=False
                )
            ], md=3),
            dbc.Col([
                html.Label("Filtrar: Mes"),
                dcc.Dropdown(
                    id='filter-month',
                    options=[{"label": m, "value": m} for m in (sorted(df['MO'].unique(), key=lambda x: int(x)) if df is not None and 'MO' in df.columns else [])],
                    placeholder="Todos",
                    multi=False
                )
            ], md=3),
            dbc.Col([
                html.Label("Variable para histograma"),
                dcc.Dropdown(id='hist-var', options=[{"label": c, "value": c} for c in numeric_cols], placeholder="Selecciona variable"),
            ], md=6),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(id='hist-graph'), md=6),
            dbc.Col(dcc.Graph(id='corr-heatmap'), md=6),
        ]),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                html.Label("X para scatter"),
                dcc.Dropdown(id='scatter-x', options=[{"label": c, "value": c} for c in numeric_cols], placeholder="X"),
            ], md=6),
            dbc.Col([
                html.Label("Y para scatter"),
                dcc.Dropdown(id='scatter-y', options=[{"label": c, "value": c} for c in numeric_cols], placeholder="Y"),
            ], md=6),
        ]),
        dcc.Graph(id='scatter-graph'),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                html.H6("Estadísticas descriptivas"),
                dash_table.DataTable(id='desc-table', page_size=10)
            ], md=6),
            dbc.Col([
                html.H6("Valores faltantes"),
                dash_table.DataTable(id='missing-table', page_size=10)
            ], md=6)
        ])
    ]),
    dcc.Tab(label='b. EDA 2', children=[
        html.H4('b. EDA 2 - Análisis adicional'),
        dcc.Markdown(VARIABLES_MD),
        html.Hr(),
        html.Label("Variable para boxplot (por categoría)"),
        dcc.Dropdown(id='box-cat', options=[{"label": c, "value": c} for c in (df.columns.tolist() if df is not None else [])], placeholder="Categoría"),
        dcc.Dropdown(id='box-num', options=[{"label": c, "value": c} for c in numeric_cols], placeholder="Numérica"),
        dcc.Graph(id='box-graph')
    ]),
    dcc.Tab(label='c. Visualización del Modelo', children=[
        html.H4('c. Visualización de Resultados del Modelo'),
        html.P('Aquí se mostrarán las métricas de evaluación del modelo y comparaciones valores reales vs predichos.'),
        dcc.Dropdown(id='maybe-target', options=[{"label": c, "value": c} for c in (df.columns.tolist() if df is not None else [])],
                     placeholder="(Opcional) Selecciona la columna target real si existe"),
        dbc.Button("Mostrar resultados del modelo (usar columnas existentes si las hay, sino entrenar demo)", id="btn-show-model", className="my-2"),
        html.Div(id='model-output'),
        dcc.Graph(id='model-scatter')
    ]),
    dcc.Tab(label='d. Indicadores del Modelo', children=[
        html.H4('d. Indicadores de Evaluación del Modelo'),
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

tabs = [
    dcc.Tab(label='1. Introducción', children=[
        html.H2("Introducción"),
        dcc.Markdown(INTRO_TEXT),
        html.Hr(),
        dcc.Markdown(VARIABLES_MD)
    ]),
    dcc.Tab(label='2. Contexto', children=[
        html.H2("Contexto"),
        dcc.Markdown(INTRO_TEXT)
    ]),
    dcc.Tab(label='3. Planteamiento del Problema', children=[
        html.H2("Planteamiento del Problema"),
        dcc.Markdown(OBJECTIVE_TEXT)
    ]),
    dcc.Tab(label='4. Objetivos y Justificación', children=[
        html.H2("Objetivos y Justificación"),
        dcc.Markdown(OBJECTIVE_TEXT)
    ]),
    dcc.Tab(label='5. Marco Teórico', children=[
        html.H2("Marco Teórico"),
        dcc.Markdown("La radiación ultravioleta B (UVB) es... (resumen del marco teórico proporcionado en el notebook).")
    ]),
    dcc.Tab(label='6. Metodología', children=[
        html.H2("Metodología"),
        dcc.Markdown(ETL_TEXT),
        subtabs_metodologia
    ]),
    dcc.Tab(label='7. Resultados y Análisis Final', children=[
        html.H2("Resultados y Análisis Final"),
        dcc.Markdown(ANALYSIS_DESC_TEXT),
        dcc.Markdown(RELATIONAL_TEXT),
        dcc.Markdown(RESULTS_INTERPRETATION),
        html.Hr(),
        dcc.Markdown(EVIDENCE_TEXT)
    ]),
    dcc.Tab(label='8. Conclusiones', children=[
        html.H2("Conclusiones"),
        html.P("(Se dejará vacía según indicación del usuario)")
    ])
]

app.layout = dbc.Container([
    html.H1("Dashboard del Proyecto Final", className="text-center my-4"),
    html.Hr(),
    dcc.Tabs(tabs)
], fluid=True)

# ----------------------------
# CALLBACKS
# ----------------------------
def filter_df(df, year, month):
    if df is None:
        return None
    dff = df.copy()
    if year:
        if 'YEAR' in dff.columns:
            dff = dff[dff['YEAR'] == str(year)]
    if month:
        if 'MO' in dff.columns:
            dff = dff[dff['MO'] == str(month)]
    return dff

@app.callback(
    Output('hist-graph', 'figure'),
    Input('hist-var', 'value'),
    Input('filter-year', 'value'),
    Input('filter-month', 'value')
)
def update_histogram(var, year, month):
    if df is None or var is None:
        return go.Figure()
    dff = filter_df(df, year, month)
    try:
        fig = px.histogram(dff, x=var, nbins=40, title=f'Histograma de {var}')
        return fig
    except Exception:
        return go.Figure()

@app.callback(
    Output('corr-heatmap', 'figure'),
    Input('filter-year', 'value'),
    Input('filter-month', 'value')
)
def update_corr(year, month):
    if df is None:
        return go.Figure()
    dff = filter_df(df, year, month)
    try:
        num = dff.select_dtypes(include=np.number)
        if num.shape[1] == 0:
            return go.Figure()
        corr = num.corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", title="Matriz de correlación (numéricas)")
        return fig
    except Exception:
        return go.Figure()

@app.callback(
    Output('scatter-graph', 'figure'),
    Input('scatter-x', 'value'),
    Input('scatter-y', 'value'),
    Input('filter-year', 'value'),
    Input('filter-month', 'value')
)
def update_scatter(x, y, year, month):
    if df is None or x is None or y is None:
        return go.Figure()
    dff = filter_df(df, year, month)
    try:
        fig = px.scatter(dff, x=x, y=y, title=f'Scatter: {x} vs {y}', trendline="ols")
        return fig
    except Exception:
        return go.Figure()

@app.callback(
    Output('box-graph', 'figure'),
    Input('box-cat', 'value'),
    Input('box-num', 'value'),
    Input('filter-year', 'value'),
    Input('filter-month', 'value')
)
def update_box(cat, num, year, month):
    if df is None or cat is None or num is None:
        return go.Figure()
    dff = filter_df(df, year, month)
    try:
        fig = px.box(dff, x=cat, y=num, points="all", title=f'Boxplot de {num} por {cat}')
        return fig
    except Exception:
        return go.Figure()

@app.callback(
    Output('desc-table', 'data'),
    Output('desc-table', 'columns'),
    Input('filter-year', 'value'),
    Input('filter-month', 'value')
)
def update_desc_table(year, month):
    if df is None:
        return [], []
    dff = filter_df(df, year, month)
    desc = summary_table(dff)
    cols = [{"name": i, "id": i} for i in desc.columns]
    return desc.to_dict('records'), cols

@app.callback(
    Output('missing-table', 'data'),
    Input('filter-year', 'value'),
    Input('filter-month', 'value')
)
def update_missing_table(year, month):
    if df is None:
        return []
    dff = filter_df(df, year, month)
    miss = missing_values_table(dff)
    return miss.to_dict('records')

# Preparación data buttons
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

# Train demo
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

# Show model results
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
    possible_pairs = [
        ('y_true', 'y_pred'),
        ('target', 'predicted'),
        ('y', 'y_pred'),
        ('y_true', 'predicted'),
        ('observed', 'predicted'),
    ]
    for true_col, pred_col in possible_pairs:
        if true_col in df.columns and pred_col in df.columns:
            paired = df[[true_col, pred_col]].dropna()
            fig = px.scatter(paired, x=true_col, y=pred_col, title="Valores reales vs predichos (from columns)")
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

    if maybe_target and maybe_target in df.columns:
        target = maybe_target
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

    numeric_cols_local = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols_local) >= 2:
        target = numeric_cols_local[0]
        features = numeric_cols_local[1:6] if len(numeric_cols_local) > 1 else []
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

# ----------------------------
# RUN
# ----------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run_server(debug=False, host="0.0.0.0", port=port)