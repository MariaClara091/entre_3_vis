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
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from scipy import stats
import statsmodels.api as sm

# ----------------------------
# CONFIG
# ----------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Dashboard del Proyecto Final"
server = app.server

DATA_FILE = "DATASET_PROY_OFICIAL.csv"  # must be in same folder as this .py

# ----------------------------
# HELPERS / ETL
# ----------------------------
def load_data(path=DATA_FILE):
    if not os.path.exists(path):
        print(f"[load_data] No existe el archivo: {path}")
        return None
    try:
        df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    except Exception as e:
        print(f"[load_data] Error leyendo CSV: {e}")
        return None
    df.columns = [c.strip() for c in df.columns]
    for c in ["YEAR", "MO", "DY", "HR"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df

def impute_outliers_iqr(df, cols=None):
    if df is None:
        return df
    dff = df.copy()
    if cols is None:
        cols = dff.select_dtypes(include=[np.number]).columns.tolist()
    for c in cols:
        ser = dff[c]
        if ser.dropna().shape[0] < 10:
            continue
        q1 = ser.quantile(0.25)
        q3 = ser.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        dff[c] = ser.clip(lower=lower, upper=upper)
    return dff

def safe_stratify_split(X, y, test_size=0.2, random_state=42):
    """
    Try stratified split when possible (classification), otherwise fall back.
    """
    stratify = None
    try:
        # If y is categorical with small number of classes -> stratify
        if pd.api.types.is_integer_dtype(y) or pd.api.types.is_object_dtype(y) or pd.api.types.is_categorical_dtype(y):
            if y.nunique() > 1 and y.nunique() <= 50:
                stratify = y
    except Exception:
        stratify = None
    try:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
    except Exception:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

def infer_problem_type(y):
    """
    Heuristic: si y es numérico y muchos valores únicos -> regresión,
    si pocos valores únicos -> clasificación.
    """
    if y is None:
        return None
    if pd.api.types.is_numeric_dtype(y):
        if y.nunique() <= 20:
            return "classification"
        else:
            return "regression"
    else:
        return "classification"

def encode_features(X):
    """
    Simple label-encoding for object/categorical columns (in-place copy).
    """
    X_enc = X.copy()
    for col in X_enc.select_dtypes(include=['object', 'category']).columns:
        try:
            X_enc[col] = LabelEncoder().fit_transform(X_enc[col].astype(str))
        except Exception:
            X_enc[col] = X_enc[col].astype('category').cat.codes
    return X_enc

def train_and_evaluate(df_local, target_col, feature_cols):
    """
    Trains RF (classification or regression) and returns (model, metrics_dict, results_df)
    results_df contains X_test features + y_true and y_pred.
    Returns None if insufficient data.
    """
    if df_local is None:
        return None
    if target_col not in df_local.columns:
        return None
    X = df_local[feature_cols].copy()
    y = df_local[target_col].copy()

    # drop NA rows
    data = pd.concat([X, y.reset_index(drop=True)], axis=1).dropna()
    if data.shape[0] < 10:
        return None

    X_clean = data.iloc[:, :-1]
    y_clean = data.iloc[:, -1]

    prob_type = infer_problem_type(y_clean)
    X_enc = encode_features(X_clean)

    X_train, X_test, y_train, y_test = safe_stratify_split(X_enc, y_clean)

    if prob_type == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {}
    if prob_type == "classification":
        try:
            metrics['accuracy'] = accuracy_score(y_test, preds)
            metrics['precision'] = precision_score(y_test, preds, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_test, preds, average='weighted', zero_division=0)
            metrics['f1'] = f1_score(y_test, preds, average='weighted', zero_division=0)
        except Exception:
            metrics['accuracy'] = accuracy_score(y_test, preds)
    else:
        metrics['rmse'] = math.sqrt(mean_squared_error(y_test, preds))
        metrics['mae'] = mean_absolute_error(y_test, preds)
        metrics['r2'] = r2_score(y_test, preds)

    results_df = X_test.copy()
    results_df['y_true'] = y_test.values
    results_df['y_pred'] = preds

    return {"model": model, "metrics": metrics, "results_df": results_df, "problem_type": prob_type}

# ----------------------------
# LOAD data
# ----------------------------
df = load_data(DATA_FILE)
numeric_cols = df.select_dtypes(include=np.number).columns.tolist() if df is not None else []

# ----------------------------
# TEXT CONTENT (from notebook)
# ----------------------------
AUTHORS = "Autores: María Clara Ávila y Mateo José Giraldo"

INTRO_TEXT = (
    "Contexto de nuestro proyecto\n\n"
    "El aumento de la radiación solar UVB puede representar un riesgo para la salud humana y el ambiente. "
    "Por ello, buscamos explorar los patrones de radiación UVB y las condiciones atmosféricas registradas en Chía, Cundinamarca para comprender su comportamiento a lo largo del tiempo."
)

OBJECTIVE_TEXT = (
    "**Objetivo general:**\n\n"
    "Analizar la variabilidad temporal y las relaciones entre radiación UVB, presión, precipitación y otras variables atmosféricas.\n\n"
    "**Objetivos específicos y justificación aplicada:**\n\n"
    "- Identificar patrones estacionales de radiación UVB para informar campañas de salud pública (protección solar).\n"
    "- Evaluar la relación entre precipitación y radiación UVB para apoyar estudios agroclimáticos y manejo de cultivos.\n"
    "- Proveer evidencia cuantitativa para modelos predictivos de radiación usando variables meteorológicas, útil para investigación académica y monitoreo ambiental.\n\n"
    "La justificación combina interés sanitario, ambiental y académico: comprender patrones de UVB en una localidad permite diseñar alertas y recomendaciones, además de aportar datos para investigaciones posteriores."
)

VARIABLES_MD = (
    "Tenemos las siguientes variables:\n\n"
    "| Variable | Descripción | Tipo | Unidad |\n"
    "|---|---|---|---|\n"
    "| YEAR | Año de registro | Categórica | - |\n"
    "| MO | Mes de registro | Categórica | - |\n"
    "| DY | Día de registro | Categórica | - |\n"
    "| HR | Hora del registro | Categórica | h |\n"
    "| ALLSKY_SFC_UVB | Irradiancia UVB en superficie bajo cielo total | Numérica | W/m² |\n"
    "| ALLSKY_SFC_LW_DWN | Irradiancia de onda larga descendente | Numérica | W/m² |\n"
    "| PRECTOTCORR | Precipitación total corregida | Numérica | mm/h |\n"
    "| PS | Presión superficial | Numérica | kPa |\n"
)

ETL_TEXT = (
    "ETL (Extract, Transform, Load)\n\n"
    "En esta etapa revisamos la estructura, limpieza y trazabilidad de los datos detectando valores faltantes, "
    "duplicados y explicamos decisiones de imputación o eliminación. Se identificaron valores negativos físicamente imposibles; "
    "se procedió a imputar atípicos mediante IQR donde aplica y a convertir tipos según necesidad."
)

ANALYSIS_DESC = (
    "Análisis descriptivo\n\n"
    "Después de la imputación de atípicos, las distribuciones muestran valores físicamente posibles y patrones coherentes con su asimetría esperada."
)

RELATIONAL_TEXT = (
    "Análisis Relacional\n\n"
    "- Variación mensual de la radiación UVB: máximo alrededor de enero (primera temporada seca).\n"
    "- Relación precipitación vs UVB: correlación negativa moderada (soporta H2).\n"
    "- Relación presión vs UVB: correlación muy débil y positiva (r ≈ 0.079); H3 no apoyada.\n"
    "- Hora del día vs UVB: correlación positiva moderada (mayor radiación en horas centrales).\n"
    "- Día del año vs precipitación: correlación negativa fuerte, sugiere estacionalidad."
)

RESULTS_INTERPRETATION = (
    "Estadísticas por mes y conclusiones parciales\n\n"
    "- Enero muestra la mayor radiación UVB promedio (pico en temporada seca).\n"
    "- La mediana de UVB igual a 0 todos los meses indica muchas horas de oscuridad o nublado.\n"
    "- H1 y H2 corroboradas por los patrones mensuales; H3 es rechazada por ausencia de patrón claro.\n"
)

EVIDENCE_TEXT = (
    "Evidencia analítica y visualización\n\n"
    "Pruebas ANOVA y Levene se usaron para comparar niveles de UVB entre meses; Levene mostró heterogeneidad de varianzas. "
    "Correlaciones muestran una relación negativa débil pero significativa entre precipitación y UVB (H2), y una relación positiva débil entre presión y UVB (contradice H3). "
    "También se observa correlación moderada entre UVB y radiación de onda larga."
)

STRATIFY_EXPLANATION = (
    "¡Perfecto! Ahora entiendo por qué las proporciones son exactamente iguales (0.556).\n\n"
    "La razón es esta línea:\n\n"
    "```python\n"
    "X_train, X_test, y_train, y_test = train_test_split(\n"
    "    X, y, test_size=0.2, random_state=42, stratify=y  # ← ¡ESTA ES LA CLAVE!\n"
    ")\n"
    "```\n\n"
    "¿Por qué `stratify=y` produce proporciones idénticas?\n\n"
    "1) Propósito: garantiza que la distribución de clases sea idéntica en train y test.\n"
    "2) Cómo funciona: si y tiene 55.6% clase 0, 44.4% clase 1, ambos conjuntos mantendrán esas proporciones.\n"
    "3) Ventajas: evaluación más justa, representatividad, evita sesgos en problemas desbalanceados.\n\n"
    "En tu caso de radiación UVB: HighUVB=1, LowUVB=0 — la división estratificada asegura representatividad en ambos conjuntos. ¡Es una buena práctica!"
)

# ----------------------------
# LAYOUT BUILD
# ----------------------------
subtabs_metodologia = dcc.Tabs([
    dcc.Tab(label='a. Definición del Problema', children=[
        html.H4('a. Definición del Problema a Resolver'),
        dcc.Markdown(OBJECTIVE_TEXT)
    ]),
    dcc.Tab(label='b. Preparación de Datos', children=[
        html.H4('b. Preparación de los Datos'),
        dcc.Markdown(ETL_TEXT),
        html.H5("Operaciones rápidas"),
        dbc.Row([
            dbc.Col(dbc.Button("Eliminar filas con NA", id="btn-dropna", color="danger", className="me-2"), width="auto"),
            dbc.Col(dbc.Button("Rellenar NA con mediana", id="btn-fillna-median", color="secondary"), width="auto"),
            dbc.Col(dbc.Button("Imputar atípicos (IQR)", id="btn-impute-iqr", color="warning"), width="auto"),
            dbc.Col(html.Div(id="prep-action-output"), width=True),
        ], className="my-2")
    ]),
    dcc.Tab(label='c. Selección del Modelo', children=[
        html.H4('c. Selección del Modelo o Algoritmo'),
        html.P("Entrenamiento demo (RandomForest). Selecciona target y features numéricas."),
        dcc.Dropdown(id="select-target", options=[{"label": c, "value": c} for c in (df.columns.tolist() if df is not None else [])],
                     placeholder="Selecciona la variable objetivo"),
        dcc.Dropdown(id="select-features", options=[{"label": c, "value": c} for c in numeric_cols],
                     multi=True, placeholder="Selecciona columnas predictoras (numéricas preferidas)"),
        dbc.Button("Entrenar modelo de ejemplo", id="btn-train-example", color="primary", className="my-2"),
        html.Div(id="train-output"),
        html.Hr(),
        dcc.Markdown(STRATIFY_EXPLANATION)
    ]),
    dcc.Tab(label='d. Evaluación del Modelo', children=[
        html.H4('d. Entrenamiento y Evaluación del Modelo'),
        html.Ul([
            html.Li('Proceso de entrenamiento: mostrado en "Selección del Modelo"'),
            html.Li('Métricas: RMSE, MAE, R2 (regresión) o Accuracy (clasificación)'),
            html.Li('Validación: Hold-out 80/20, con opción de stratify en clasificación')
        ]),
    ])
])

subtabs_resultados = dcc.Tabs([
    dcc.Tab(label='a. EDA', children=[
        html.H4('a. Análisis Exploratorio de Datos (EDA)'),
        html.P(AUTHORS),
        dcc.Markdown(INTRO_TEXT),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                html.Label("Filtrar Año"),
                dcc.Dropdown(id='filter-year', options=[{"label": y, "value": y} for y in sorted(df['YEAR'].unique())] if df is not None and 'YEAR' in df.columns else [], placeholder="Todos")
            ], md=3),
            dbc.Col([
                html.Label("Filtrar Mes"),
                dcc.Dropdown(id='filter-month', options=[{"label": m, "value": m} for m in sorted(df['MO'].unique(), key=lambda x: int(x))] if df is not None and 'MO' in df.columns else [], placeholder="Todos")
            ], md=3),
            dbc.Col([
                html.Label("Variable (histograma)"),
                dcc.Dropdown(id='hist-var', options=[{"label": c, "value": c} for c in numeric_cols], placeholder="Selecciona variable")
            ], md=6),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(id='hist-graph'), md=6),
            dbc.Col(dcc.Graph(id='corr-heatmap'), md=6),
        ]),
        html.Hr(),
        dbc.Row([
            dbc.Col([html.Label("X para scatter"), dcc.Dropdown(id='scatter-x', options=[{"label": c, "value": c} for c in numeric_cols])], md=6),
            dbc.Col([html.Label("Y para scatter"), dcc.Dropdown(id='scatter-y', options=[{"label": c, "value": c} for c in numeric_cols])], md=6),
        ]),
        dcc.Graph(id='scatter-graph'),
        html.Hr(),
        dbc.Row([
            dbc.Col([html.H6("Estadísticas descriptivas"), dash_table.DataTable(id='desc-table', page_size=10)], md=6),
            dbc.Col([html.H6("Valores faltantes"), dash_table.DataTable(id='missing-table', page_size=10)], md=6)
        ])
    ]),
    dcc.Tab(label='b. EDA 2', children=[
        html.H4('b. EDA 2 - Análisis adicional'),
        dcc.Markdown(VARIABLES_MD),
        html.Hr(),
        html.Label("Boxplot: categoría (e.g., HR, MO)"),
        dcc.Dropdown(id='box-cat', options=[{"label": c, "value": c} for c in (df.columns.tolist() if df is not None else [])], placeholder="Categoría"),
        dcc.Dropdown(id='box-num', options=[{"label": c, "value": c} for c in numeric_cols], placeholder="Numérica"),
        dcc.Graph(id='box-graph')
    ]),
    dcc.Tab(label='c. Visualización del Modelo', children=[
        html.H4('c. Visualización de Resultados del Modelo'),
        html.P('Si tu CSV ya contiene columnas de verdad (y_true) y predicción (y_pred), se mostrarán. Si no, entrena demo.'),
        dcc.Dropdown(id='maybe-target', options=[{"label": c, "value": c} for c in (df.columns.tolist() if df is not None else [])], placeholder="(Opcional) Selecciona target real"),
        dbc.Button("Mostrar resultados del modelo (usar columnas si existen / sino entrenar demo)", id="btn-show-model", className="my-2"),
        html.Div(id='model-output'),
        dcc.Graph(id='model-scatter')
    ]),
    dcc.Tab(label='d. Indicadores del Modelo', children=[
        html.H4('d. Indicadores de Evaluación del Modelo'),
        html.Div(id='model-metrics-table')
    ]),
    dcc.Tab(label='e. Limitaciones', children=[
        html.H4('e. Limitaciones y Consideraciones Finales'),
        html.Ul([html.Li('Restricciones del análisis'), html.Li('Posibles mejoras futuras')])
    ])
])

tabs = [
    dcc.Tab(label='1. Introducción', children=[html.H2("Introducción"), dcc.Markdown(INTRO_TEXT), html.Hr(), dcc.Markdown(VARIABLES_MD)]),
    dcc.Tab(label='2. Contexto', children=[html.H2("Contexto"), dcc.Markdown(INTRO_TEXT)]),
    dcc.Tab(label='3. Planteamiento del Problema', children=[html.H2("Planteamiento del Problema"), dcc.Markdown(OBJECTIVE_TEXT)]),
    dcc.Tab(label='4. Objetivos y Justificación', children=[html.H2("Objetivos y Justificación"), dcc.Markdown(OBJECTIVE_TEXT)]),
    dcc.Tab(label='5. Marco Teórico', children=[html.H2("Marco Teórico"), dcc.Markdown("La radiación ultravioleta B (UVB) es relevante para salud y ambiente...")]),
    dcc.Tab(label='6. Metodología', children=[html.H2("Metodología"), dcc.Markdown(ETL_TEXT), subtabs_metodologia]),
    dcc.Tab(label='7. Resultados y Análisis Final', children=[
        html.H2("Resultados y Análisis Final"),
        dcc.Markdown(ANALYSIS_DESC),
        dcc.Markdown(RELATIONAL_TEXT),
        dcc.Markdown(RESULTS_INTERPRETATION),
        html.Hr(),
        dcc.Markdown(EVIDENCE_TEXT),
        html.Hr(),
        dcc.Markdown(STRATIFY_EXPLANATION)
    ]),
    dcc.Tab(label='8. Conclusiones', children=[html.H2("Conclusiones"), html.P("(Se deja vacía)")])
]

app.layout = dbc.Container([html.H1("Dashboard del Proyecto Final", className="text-center my-4"), html.Hr(), dcc.Tabs(tabs)], fluid=True)

# ----------------------------
# CALLBACKS
# ----------------------------
def filter_df(df_local, year, month):
    if df_local is None:
        return None
    dff = df_local.copy()
    if year and 'YEAR' in dff.columns:
        dff = dff[dff['YEAR'] == str(year)]
    if month and 'MO' in dff.columns:
        dff = dff[dff['MO'] == str(month)]
    return dff

@app.callback(
    Output('hist-graph', 'figure'),
    Input('hist-var', 'value'),
    Input('filter-year', 'value'),
    Input('filter-month', 'value'),
)
def update_histogram(var, year, month):
    if df is None or var is None:
        return go.Figure()
    dff = filter_df(df, year, month)
    if dff is None or dff.shape[0] == 0:
        return go.Figure()
    try:
        fig = px.histogram(dff, x=var, nbins=40, title=f'Histograma de {var}')
        return fig
    except Exception as e:
        print("update_histogram error:", e)
        return go.Figure()

@app.callback(
    Output('corr-heatmap', 'figure'),
    Input('filter-year', 'value'),
    Input('filter-month', 'value'),
)
def update_corr(year, month):
    if df is None:
        return go.Figure()
    dff = filter_df(df, year, month)
    if dff is None:
        return go.Figure()
    num = dff.select_dtypes(include=np.number)
    if num.shape[1] == 0:
        return go.Figure()
    try:
        corr = num.corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", title="Matriz de correlación (numéricas)")
        return fig
    except Exception as e:
        print("update_corr error:", e)
        return go.Figure()

@app.callback(
    Output('scatter-graph', 'figure'),
    Input('scatter-x', 'value'),
    Input('scatter-y', 'value'),
    Input('filter-year', 'value'),
    Input('filter-month', 'value'),
)
def update_scatter(x, y, year, month):
    if df is None or x is None or y is None:
        return go.Figure()
    dff = filter_df(df, year, month)
    if dff is None or dff.shape[0] == 0:
        return go.Figure()
    try:
        fig = px.scatter(dff, x=x, y=y, trendline="ols", title=f'Scatter: {x} vs {y}')
        return fig
    except Exception as e:
        print("update_scatter error:", e)
        return go.Figure()

@app.callback(
    Output('box-graph', 'figure'),
    Input('box-cat', 'value'),
    Input('box-num', 'value'),
    Input('filter-year', 'value'),
    Input('filter-month', 'value'),
)
def update_box(cat, num, year, month):
    if df is None or cat is None or num is None:
        return go.Figure()
    dff = filter_df(df, year, month)
    if dff is None or dff.shape[0] == 0:
        return go.Figure()
    try:
        fig = px.box(dff, x=cat, y=num, points="all", title=f'Boxplot de {num} por {cat}')
        return fig
    except Exception as e:
        print("update_box error:", e)
        return go.Figure()

@app.callback(
    Output('desc-table', 'data'),
    Output('desc-table', 'columns'),
    Input('filter-year', 'value'),
    Input('filter-month', 'value'),
)
def update_desc_table(year, month):
    if df is None:
        return [], []
    dff = filter_df(df, year, month)
    try:
        desc = dff.describe(include='all').transpose().reset_index().rename(columns={"index": "variable"})
    except Exception as e:
        print("update_desc_table error:", e)
        return [], []
    if desc.empty:
        return [], []
    cols = [{"name": i, "id": i} for i in desc.columns]
    return desc.to_dict('records'), cols

@app.callback(
    Output('missing-table', 'data'),
    Input('filter-year', 'value'),
    Input('filter-month', 'value'),
)
def update_missing_table(year, month):
    if df is None:
        return []
    dff = filter_df(df, year, month)
    miss = dff.isna().sum().reset_index()
    miss.columns = ["variable", "missing_count"]
    miss["missing_pct"] = (miss["missing_count"] / max(1, len(dff))) * 100
    return miss.to_dict('records')

# Prep actions
@app.callback(
    Output('prep-action-output', 'children'),
    Input('btn-dropna', 'n_clicks'),
    Input('btn-fillna-median', 'n_clicks'),
    Input('btn-impute-iqr', 'n_clicks'),
    prevent_initial_call=True
)
def prep_actions(n_dropna, n_fill_median, n_iqr):
    ctx = dash.callback_context
    global df, numeric_cols
    if not ctx.triggered:
        return ""
    action = ctx.triggered[0]['prop_id'].split('.')[0]
    if df is None:
        return "No hay dataset cargado."
    if action == "btn-dropna":
        df = df.dropna()
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        return "Filas con NA eliminadas (global)."
    elif action == "btn-fillna-median":
        num_cols = df.select_dtypes(include=np.number).columns
        for c in num_cols:
            df[c] = df[c].fillna(df[c].median())
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        return "NAs numéricos rellenados con mediana."
    elif action == "btn-impute-iqr":
        phys_cols = [c for c in ["ALLSKY_SFC_UVB", "ALLSKY_SFC_LW_DWN", "PRECTOTCORR", "PS"] if c in df.columns]
        for c in phys_cols:
            # cap extreme negatives -> set NaN if absurdly negative
            try:
                q01 = df[c].quantile(0.01)
                stdc = df[c].std(skipna=True)
                df.loc[df[c] < (q01 - 10*stdc), c] = np.nan
            except Exception:
                pass
        df = impute_outliers_iqr(df, cols=phys_cols)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        return "Imputación de atípicos (IQR) aplicada en columnas físicas."
    return ""

# Train demo callback (replaces compute_model_metrics usage)
@app.callback(
    Output('train-output', 'children'),
    Input('btn-train-example', 'n_clicks'),
    State('select-target', 'value'),
    State('select-features', 'value'),
    prevent_initial_call=True
)
def train_example(n_clicks, target, features):
    if df is None:
        return html.Div("No hay dataset cargado.")
    if not target or not features:
        return html.Div("Selecciona target y al menos una feature.")
    res = train_and_evaluate(df, target, features)
    if res is None:
        return html.Div("No se pudo entrenar el modelo con los datos seleccionados (filtrado por NaN o pocos datos).")
    metrics = res["metrics"]
    prob_type = res.get("problem_type", "regression")
    rows = []
    for k, v in metrics.items():
        try:
            rows.append(html.Li(f"{k}: {v:.4f}"))
        except Exception:
            rows.append(html.Li(f"{k}: {v}"))
    return html.Div([html.H6("Métricas (ejemplo)"), html.Ul(rows), html.P(f"Tipo problema detectado: {prob_type}")])

# Show model results (uses direct training if needed)
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
    # If dataset already has predictions
    pairs = [('y_true', 'y_pred'), ('target', 'predicted'), ('y', 'y_pred'), ('observed', 'predicted')]
    for tcol, pcol in pairs:
        if tcol in df.columns and pcol in df.columns:
            paired = df[[tcol, pcol]].dropna()
            fig = px.scatter(paired, x=tcol, y=pcol, title="Valores reales vs predichos (from columns)")
            if pd.api.types.is_numeric_dtype(paired[tcol]):
                rmse = math.sqrt(mean_squared_error(paired[tcol], paired[pcol]))
                mae = mean_absolute_error(paired[tcol], paired[pcol])
                table = dbc.Table([
                    html.Thead(html.Tr([html.Th("Métrica"), html.Th("Valor")])),
                    html.Tbody([html.Tr([html.Td("RMSE"), html.Td(f"{rmse:.4f}")]),
                                html.Tr([html.Td("MAE"), html.Td(f"{mae:.4f}")])])
                ], bordered=True)
            else:
                table = html.P("No numeric target: no se calculan RMSE/MAE.")
            return html.Div([html.H6("Se detectaron columnas de verdad y predicción en el dataset.")]), fig, table
    # else if user provided maybe_target, train demo
    if maybe_target and maybe_target in df.columns:
        candidate_feats = [c for c in df.select_dtypes(include=np.number).columns.tolist() if c != maybe_target]
        if len(candidate_feats) == 0:
            return html.Div("No se encontraron features numéricas para entrenar un modelo de ejemplo."), go.Figure(), ""
        features = candidate_feats[:5]
        res = train_and_evaluate(df, maybe_target, features)
        if res is None:
            return html.Div("No se pudo entrenar modelo demo con los datos seleccionados."), go.Figure(), ""
        results_df = res["results_df"]
        fig = px.scatter(results_df, x='y_true', y='y_pred', title="Valores reales vs predichos (modelo demo)")
        m = res["metrics"]
        rows = []
        for k, v in m.items():
            try:
                rows.append(html.Tr([html.Td(k), html.Td(f"{v:.4f}")]))
            except Exception:
                rows.append(html.Tr([html.Td(k), html.Td(str(v))]))
        metrics_table = dbc.Table([html.Thead(html.Tr([html.Th("Métrica"), html.Th("Valor")])), html.Tbody(rows)], bordered=True)
        return html.Div([html.H6("Resultados del modelo demo:"), html.P(f"Target: {maybe_target}. Features usadas: {', '.join(features)}.")]), fig, metrics_table

    # fallback: automatic demo using numeric columns
    numeric_local = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_local) >= 2:
        target = numeric_local[0]
        features = numeric_local[1:6]
        res = train_and_evaluate(df, target, features)
        if res is None:
            return html.Div("No fue posible entrenar un modelo demo automáticamente."), go.Figure(), ""
        results_df = res["results_df"]
        fig = px.scatter(results_df, x='y_true', y='y_pred', title=f"Valores reales vs predichos (modelo demo, target={target})")
        m = res["metrics"]
        rows = []
        for k, v in m.items():
            try:
                rows.append(html.Tr([html.Td(k), html.Td(f"{v:.4f}")]))
            except Exception:
                rows.append(html.Tr([html.Td(k), html.Td(str(v))]))
        metrics_table = dbc.Table([html.Thead(html.Tr([html.Th("Métrica"), html.Th("Valor")])), html.Tbody(rows)], bordered=True)
        return html.Div([html.H6("Se entrenó un modelo demo automáticamente usando columnas numéricas.")]), fig, metrics_table

    return html.Div("No se detectaron columnas de predicción ni suficientes columnas numéricas."), go.Figure(), ""

# ANOVA / Levene computations for model-metrics table
@app.callback(
    Output('model-metrics-table', 'children'),
    Input('filter-year', 'value'),
    Input('filter-month', 'value'),
)
def compute_stats(year, month):
    if df is None:
        return ""
    dff = filter_df(df, year, month)
    if dff is None or dff.shape[0] < 10:
        return html.P("No hay datos suficientes para pruebas estadísticas.")
    if 'ALLSKY_SFC_UVB' in dff.columns and 'MO' in dff.columns:
        try:
            groups = [g.dropna() for _, g in dff.groupby('MO')['ALLSKY_SFC_UVB']]
            groups_valid = [g for g in groups if g.shape[0] > 1]
            if len(groups_valid) >= 2:
                f, p = stats.f_oneway(*groups_valid)
                lev_stat, lev_p = stats.levene(*groups_valid)
                return html.Div([html.P(f"ANOVA F={f:.4f}, p={p:.4e}"), html.P(f"Levene stat={lev_stat:.4f}, p={lev_p:.4e}")])
            else:
                return html.P("ANOVA/Levene no aplicable: no hay suficientes grupos con muestras.")
        except Exception as e:
            return html.P(f"Error en pruebas estadísticas: {e}")
    return html.P("ANOVA no aplicable (faltan columnas)")

# ----------------------------
# RUN
# ----------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run_server(debug=False, host="0.0.0.0", port=port)