# ============================================================
# app_proyect_final.py
# Dashboard completo (Dash + Bootstrap) para el proyecto:
# "Análisis de Radiación UVB en Chía, Cundinamarca"
#
# Este archivo integra:
# - Todo el EDA (gráficas, tablas, textos interpretativos)
# - Pruebas estadísticas (normalidad, Levene, ANOVA, correlaciones)
# - Modelo XGBoost con pipeline y validación (tal como en el notebook)
# - Visualizaciones (matplotlib -> base64) embebidas en Dash
#
# NOTAS:
# - Colocar DATASET_PROY_OFICIAL.csv en el mismo folder que este .py
# - No se incluye sección de Conclusiones (tal como solicitaste)
# - Requiere: dash, dash_bootstrap_components, pandas, numpy, matplotlib,
#   seaborn, scikit-learn, xgboost, scipy
# ============================================================

import os
import io
import base64
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, f1_score, balanced_accuracy_score, roc_curve, auc
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import xgboost as xgb
from scipy.stats import shapiro, levene, f_oneway, pearsonr

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["font.size"] = 11

# ---------------------------
# 1) CARGAR DATOS
# ---------------------------
CSV = "DATASET_PROY_OFICIAL.csv"
if not os.path.exists(CSV):
    raise FileNotFoundError(f"El dataset '{CSV}' no fue encontrado en el directorio actual.")

datos = pd.read_csv(CSV, encoding="utf-8")
# Crear copia para limpieza/uso
df = datos.copy()

# ---------------------------
# 2) LIMPIEZA BÁSICA
# ---------------------------
# Eliminar registros con valores físicamente imposibles (<= 0) en variables físicas clave
variables_fisicas = ["ALLSKY_SFC_UVB", "ALLSKY_SFC_LW_DWN", "PRECTOTCORR", "PS"]
for v in variables_fisicas:
    if v in df.columns:
        df = df[df[v] > 0]

df = df.reset_index(drop=True)

# Asegurar que columnas de fecha/hora existen (si no, intenta inferir)
for col in ["YEAR", "MO", "DY", "HR"]:
    if col not in df.columns:
        df[col] = np.nan

# ---------------------------
# 3) UTILIDADES: convertir figura a base64
# ---------------------------
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_b64

# ---------------------------
# 4) GRÁFICAS (usamos casi el mismo código del notebook)
# ---------------------------
def grafico_hist(col, title, bins=30):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.histplot(df[col].dropna(), bins=bins, kde=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(col)
    ax.set_ylabel("Frecuencia")
    return fig_to_base64(fig)

def grafico_corr_matrix(cols):
    fig, ax = plt.subplots(figsize=(8,6))
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Matriz de correlación")
    return fig_to_base64(fig)

def grafico_uvb_mes():
    # Agrupar por mes (asumiendo MO numérico 1-12)
    if "MO" not in df.columns:
        return None
    series = df.groupby("MO")["ALLSKY_SFC_UVB"].mean().sort_index()
    fig, ax = plt.subplots(figsize=(10,5))
    series.plot(kind="bar", ax=ax)
    ax.set_title("Radiación UVB Promedio por Mes")
    ax.set_xlabel("Mes")
    ax.set_ylabel("UVB Promedio (W/m²)")
    return fig_to_base64(fig)

def grafico_scatter(x_col, y_col, title):
    fig, ax = plt.subplots(figsize=(9,5))
    ax.scatter(df[x_col], df[y_col], alpha=0.5, s=18)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    # calcular Pearson si hay suficientes valores
    try:
        mask = df[[x_col, y_col]].dropna()
        if len(mask) > 2:
            r, p = pearsonr(mask[x_col], mask[y_col])
            ax.text(0.02, 0.95, f"Pearson r = {r:.3f}\np = {p:.3e}",
                    transform=ax.transAxes, verticalalignment="top",
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    except Exception:
        pass
    return fig_to_base64(fig)

# Generar principales imágenes (base64)
imgs = {}
# histograms
if "ALLSKY_SFC_UVB" in df.columns:
    imgs["uvb_hist"] = grafico_hist("ALLSKY_SFC_UVB", "Distribución de Radiación UVB")
if "ALLSKY_SFC_LW_DWN" in df.columns:
    imgs["lw_hist"] = grafico_hist("ALLSKY_SFC_LW_DWN", "Distribución Radiación Onda Larga")
if "PRECTOTCORR" in df.columns:
    imgs["precip_hist"] = grafico_hist("PRECTOTCORR", "Distribución de Precipitación")
if "PS" in df.columns:
    imgs["ps_hist"] = grafico_hist("PS", "Distribución de Presión Superficial")

# correlation matrix
numeric_cols = [c for c in ["ALLSKY_SFC_UVB", "ALLSKY_SFC_LW_DWN", "PRECTOTCORR", "PS"] if c in df.columns]
if len(numeric_cols) >= 2:
    imgs["corr"] = grafico_corr_matrix(numeric_cols)

# uvb by month
if "MO" in df.columns and "ALLSKY_SFC_UVB" in df.columns:
    imgs["uvb_mes"] = grafico_uvb_mes()

# scatterplots
if "PRECTOTCORR" in df.columns and "ALLSKY_SFC_UVB" in df.columns:
    imgs["uvb_precip"] = grafico_scatter("PRECTOTCORR", "ALLSKY_SFC_UVB", "UVB vs Precipitación")
if "PS" in df.columns and "ALLSKY_SFC_UVB" in df.columns:
    imgs["uvb_ps"] = grafico_scatter("PS", "ALLSKY_SFC_UVB", "UVB vs Presión Superficial")

# ---------------------------
# 5) ESTADÍSTICAS Y PRUEBAS
# ---------------------------
stats_results = {}

# Normalidad (Shapiro) por variable (si número de muestras adecuado)
for col in numeric_cols:
    try:
        sample = df[col].dropna()
        # Shapiro requiere <= 5000 observations; si más, muestreamos 5000
        if len(sample) > 5000:
            sample_sh = sample.sample(5000, random_state=42)
        else:
            sample_sh = sample
        stat, p = shapiro(sample_sh)
        stats_results[f"shapiro_{col}"] = (stat, p)
    except Exception:
        stats_results[f"shapiro_{col}"] = (np.nan, np.nan)

# Homogeneidad (Levene) entre meses para UVB (si hay MO)
if "ALLSKY_SFC_UVB" in df.columns and "MO" in df.columns:
    try:
        groups = [g["ALLSKY_SFC_UVB"].values for _, g in df.groupby("MO")]
        stat_levene, p_levene = levene(*groups)
        stats_results["levene_uvb_by_month"] = (stat_levene, p_levene)
    except Exception:
        stats_results["levene_uvb_by_month"] = (np.nan, np.nan)

# ANOVA UVB por mes
if "ALLSKY_SFC_UVB" in df.columns and "MO" in df.columns:
    try:
        groups_anova = [g["ALLSKY_SFC_UVB"].values for _, g in df.groupby("MO")]
        stat_anova, p_anova = f_oneway(*groups_anova)
        stats_results["anova_uvb_by_month"] = (stat_anova, p_anova)
    except Exception:
        stats_results["anova_uvb_by_month"] = (np.nan, np.nan)

# Correlaciones (pearson) guardadas
corr_table = []
if "ALLSKY_SFC_UVB" in df.columns:
    for v in ["PRECTOTCORR", "PS", "ALLSKY_SFC_LW_DWN"]:
        if v in df.columns:
            try:
                mask = df[["ALLSKY_SFC_UVB", v]].dropna()
                r, p = pearsonr(mask["ALLSKY_SFC_UVB"], mask[v])
                corr_table.append((v, r, p))
            except Exception:
                corr_table.append((v, np.nan, np.nan))

# ---------------------------
# 6) MODELO XGBOOST (pipeline tal como en el notebook)
# ---------------------------
# Crear la variable objetivo "HighUVB" con umbral de la mediana (igual que en notebook)
model_outputs = {}
if "ALLSKY_SFC_UVB" in df.columns:
    datos_model = df.copy()
    umbral_uvb = datos_model["ALLSKY_SFC_UVB"].median()
    datos_model["HighUVB"] = (datos_model["ALLSKY_SFC_UVB"] >= umbral_uvb).astype(int)

    X = datos_model.drop(columns=["ALLSKY_SFC_UVB", "HighUVB"])
    y = datos_model["HighUVB"]

    # Selección simple de features (tal y como en notebook original)
    # Si faltan columnas numéricas/categóricas en X, se ajusta la lista
    numeric_features = [c for c in ["ALLSKY_SFC_LW_DWN", "PRECTOTCORR", "PS"] if c in X.columns]
    categorical_features = [c for c in ["HR", "DY", "MO"] if c in X.columns]

    # División estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model_outputs["train_shape"] = (X_train.shape, X_test.shape)
    model_outputs["class_prop_train"] = float(np.mean(y_train))
    model_outputs["class_prop_test"] = float(np.mean(y_test))

    # Preprocesador
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]) if numeric_features else None

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]) if categorical_features else None

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers) if transformers else None

    # scale_pos_weight
    pos = np.sum(y_train == 1)
    neg = np.sum(y_train == 0)
    scale_pos_weight_val = (neg / pos) if pos > 0 else 1.0
    model_outputs["scale_pos_weight"] = float(scale_pos_weight_val)

    # Pipeline XGBoost
    xgb_clf = xgb.XGBClassifier(
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        use_label_encoder=False,
        scale_pos_weight=scale_pos_weight_val
    )

    steps = []
    if preprocessor is not None:
        steps.append(("preprocessor", preprocessor))
    steps.append(("model", xgb_clf))
    pipeline = Pipeline(steps=steps)

    # Grid para búsqueda (similar al notebook)
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [3, 6],
        "model__learning_rate": [0.05, 0.1],
        "model__subsample": [0.8, 1.0]
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="f1_macro",
        n_jobs=-1,
        verbose=0,
        refit=True,
        return_train_score=False
    )

    # Entrenar (puede tardar dependiendo del tamaño del dataset)
    start = time.time()
    try:
        grid.fit(X_train, y_train)
        elapsed = time.time() - start
        model_outputs["grid_time_sec"] = elapsed
        model_outputs["best_params"] = grid.best_params_
        model_outputs["best_cv_score"] = float(grid.best_score_)
        final_model = grid.best_estimator_

        # Predicción sobre test
        y_pred = final_model.predict(X_test)
        y_proba = final_model.predict_proba(X_test)[:, 1] if hasattr(final_model, "predict_proba") else None

        # Métricas
        acc = accuracy_score(y_test, y_pred)
        f1m = f1_score(y_test, y_pred, average="macro")
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        auc_score = float(roc_auc_score(y_test, y_proba)) if y_proba is not None else np.nan

        model_outputs["metrics"] = {
            "accuracy": float(acc),
            "f1_macro": float(f1m),
            "balanced_accuracy": float(bal_acc),
            "auc": auc_score
        }

        model_outputs["classification_report"] = classification_report(y_test, y_pred, output_dict=True)
        # Matriz de confusión imagen
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Verdadero")
        ax.set_title("Matriz de Confusión - XGBoost")
        model_outputs["cm_img"] = fig_to_base64(fig)

        # ROC curve image
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            fig2, ax2 = plt.subplots(figsize=(6,5))
            ax2.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
            ax2.plot([0,1],[0,1], linestyle="--", color="gray")
            ax2.set_xlabel("False Positive Rate")
            ax2.set_ylabel("True Positive Rate")
            ax2.set_title("ROC - XGBoost")
            ax2.legend(loc="lower right")
            model_outputs["roc_img"] = fig_to_base64(fig2)
    except Exception as e:
        model_outputs["error"] = str(e)

# ---------------------------
# 7) DASH APP: LAYOUT (texto, tablas, imágenes, model outputs)
# ---------------------------
external_styles = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_styles)
server = app.server
app.title = "Dashboard Proyecto Final - UVB Chía"

# Helper: build HTML table for month stats (we will compute simple summary)
def build_month_table():
    rows = []
    if "MO" in df.columns and "ALLSKY_SFC_UVB" in df.columns:
        summary = df.groupby("MO")["ALLSKY_SFC_UVB"].agg(["mean","std","min","max"]).reset_index().sort_values("MO")
        # If there are many months, limit to 12
        for _, r in summary.iterrows():
            rows.append(html.Tr([html.Td(int(r["MO"])), 
                                 html.Td(f"{r['mean']:.3f}"),
                                 html.Td(f"{r['std']:.3f}"),
                                 html.Td(f"{r['min']:.3f}"),
                                 html.Td(f"{r['max']:.3f}")]))
    return html.Table([html.Thead(html.Tr([html.Th("Mes"), html.Th("UVB Promedio"), html.Th("Desv. Est."), html.Th("Mínimo"), html.Th("Máximo")])),
                       html.Tbody(rows)], className="table table-striped table-sm")

# Build correlation summary table (from corr_table)
def build_corr_table():
    rows = []
    for v, r, p in corr_table:
        rows.append(html.Tr([html.Td(v), html.Td(f"{r:.3f}"), html.Td(f"p = {p:.3e}")]))
    return html.Table([html.Thead(html.Tr([html.Th("Variables"), html.Th("Coeficiente"), html.Th("Significancia")])),
                       html.Tbody(rows)], className="table table-sm table-striped")

# Layout content pieces (texts taken/rewritten from your notebook)
intro_card = dbc.Card(
    dbc.CardBody([
        html.H2("Introducción"),
        html.P("Este dashboard presenta el análisis de la variabilidad temporal y las relaciones entre radiación UVB, presión, precipitación y otras variables atmosféricas registradas en Chía, Cundinamarca."),
        html.P("Objetivo: Analizar relaciones entre variables meteorológicas y validar hipótesis sobre el comportamiento de la radiación UVB.")
    ])
)

context_card = dbc.Card(
    dbc.CardBody([
        html.H3("Contexto"),
        html.P("Dataset: DATASET_PROY_OFICIAL.csv — registros horarios durante un año (ubicación: Chía, Cundinamarca)."),
        html.P("Variables clave: ALLSKY_SFC_UVB (W/m²), ALLSKY_SFC_LW_DWN (W/m²), PRECTOTCORR (mm/h), PS (kPa).")
    ])
)

problema_card = dbc.Card(
    dbc.CardBody([
        html.H3("Planteamiento del problema"),
        html.P("¿Cuáles son los patrones temporales de la radiación UVB en Chía y cómo se relaciona con la precipitación y la presión superficial?")
    ])
)

objetivos_card = dbc.Card(
    dbc.CardBody([
        html.H3("Objetivos y Justificación"),
        html.H5("Objetivo general"),
        html.P("Analizar la variabilidad temporal y las relaciones entre radiación UVB y variables atmosféricas."),
        html.H5("Objetivos específicos"),
        html.Ul([html.Li("Caracterizar la distribución y variabilidad temporal de la radiación UVB"),
                 html.Li("Evaluar la relación entre radiación UVB y precipitación"),
                 html.Li("Analizar la asociación entre presión superficial y radiación UVB")]),
        html.H5("Justificación"),
        html.P("Comprender patrones de radiación UVB es importante para salud pública, agricultura y planificación urbana.")
    ])
)

marco_card = dbc.Card(
    dbc.CardBody([
        html.H3("Marco Teórico (resumen académico)"),
        html.H5("Radiación UVB"),
        html.P("La radiación UVB (280–315 nm) es parcialmente absorbida por la capa de ozono y su variación depende de la altura solar, nubosidad, aerosoles y composición atmosférica."),
        html.H5("Presión Superficial"),
        html.P("La presión superficial se relaciona con los sistemas meteorológicos y la presencia de nubosidad, afectando la transmisión de radiación."),
        html.H5("Radiación de Onda Larga"),
        html.P("La radiación LW es emitida por la superficie y la atmósfera y es indicativa de procesos térmicos atmosféricos."),
    ])
)

metodologia_card = dbc.Card(
    dbc.CardBody([
        html.H3("Metodología"),
        html.P("Se realizó un ETL básico, EDA, pruebas estadísticas y se construyó un modelo de clasificación (XGBoost) para distinguir niveles de UVB (High vs Low) usando la mediana como umbral."),
        html.Ul([
            html.Li("Limpieza: eliminación de valores físicamente imposibles (<=0)."),
            html.Li("EDA: histogramas, scatterplots, matriz de correlación, análisis mensual."),
            html.Li("Pruebas: Shapiro (normalidad), Levene (homogeneidad), ANOVA por mes."),
            html.Li("Modelado: pipeline con ColumnTransformer y XGBoost (GridSearchCV con CV estratificado).")
        ])
    ])
)

# Resultados: EDA, correlaciones, tablas, model outputs
eda_card = dbc.Card(
    dbc.CardBody([
        html.H3("Resultados y Análisis"),

        html.H4("Análisis Exploratorio - Distribuciones"),
        html.Div([
            html.Img(src="data:image/png;base64,{}".format(imgs.get("uvb_hist","")), style={"width":"45%", "margin":"8px"}),
            html.Img(src="data:image/png;base64,{}".format(imgs.get("lw_hist","")), style={"width":"45%", "margin":"8px"})
        ]),
        html.Div([
            html.Img(src="data:image/png;base64,{}".format(imgs.get("precip_hist","")), style={"width":"45%", "margin":"8px"}),
            html.Img(src="data:image/png;base64,{}".format(imgs.get("ps_hist","")), style={"width":"45%", "margin":"8px"})
        ]),

        html.H4("Correlaciones"),
        html.Div([
            html.Img(src="data:image/png;base64,{}".format(imgs.get("corr","")), style={"width":"60%", "margin":"8px"}),
        ]),
        html.Div([
            html.H5("Resumen de correlaciones clave"),
            build_corr_table()
        ]),

        html.H4("Análisis Temporal"),
        html.Div([
            html.Img(src="data:image/png;base64,{}".format(imgs.get("uvb_mes","")), style={"width":"70%", "margin":"8px"})
        ]),
        html.H5("Estadísticas por Mes"),
        build_month_table(),

        html.H4("UVB vs Precipitación"),
        html.Img(src="data:image/png;base64,{}".format(imgs.get("uvb_precip","")), style={"width":"60%"}),
        html.H4("UVB vs Presión Superficial"),
        html.Img(src="data:image/png;base64,{}".format(imgs.get("uvb_ps","")), style={"width":"60%"}),
    ])
)

validacion_card = dbc.Card(
    dbc.CardBody([
        html.H3("Validación de Hipótesis y pruebas estadísticas"),
        html.Ul([
            html.Li("H1: Picos de UVB en meses secos — ver gráfico mensual."),
            html.Li("H2: Correlación negativa entre precipitación y UVB — ver coeficiente en la tabla de correlaciones."),
            html.Li("H3: Presión y UVB: correlación débil pero significativa en los datos.")
        ]),
        html.H5("Resumen pruebas estadísticas (Shapiro – normalidad)"),
        html.Div([html.Pre(str(stats_results))], style={"whiteSpace":"pre-wrap", "fontSize":12}),
        html.H5("ANOVA y Levene para UVB por mes"),
        html.Div([
            html.P(f"Levene (varianzas entre meses) estat={stats_results.get('levene_uvb_by_month',('n/a','n/a'))}"),
            html.P(f"ANOVA (UVB por mes) estat={stats_results.get('anova_uvb_by_month',('n/a','n/a'))}")
        ])
    ])
)

# Modelo card (si se entrenó)
model_card_children = [
    html.H3("Modelado — Clasificación UVB (XGBoost)"),
    html.P("Se definió 'HighUVB' usando la mediana de ALLSKY_SFC_UVB y se entrenó un pipeline con validación cruzada estratificada."),
    html.P(f"Datos usados (train/test): {model_outputs.get('train_shape','No entrenado')}"),
    html.P(f"Proporción clase (train/test): {model_outputs.get('class_prop_train','-'):.3f} / {model_outputs.get('class_prop_test','-'):.3f}" if "class_prop_train" in model_outputs else "")
]

if "metrics" in model_outputs:
    metrics = model_outputs["metrics"]
    model_card_children += [
        html.H5("Métricas en Test"),
        html.Ul([
            html.Li(f"Accuracy: {metrics.get('accuracy'):.4f}"),
            html.Li(f"Balanced Accuracy: {metrics.get('balanced_accuracy'):.4f}"),
            html.Li(f"F1-macro: {metrics.get('f1_macro'):.4f}"),
            html.Li(f"AUC: {metrics.get('auc'):.4f}")
        ]),
        html.H5("Reporte de clasificación (por clase)"),
        html.Pre(str(model_outputs["classification_report"])),
    ]
    if "cm_img" in model_outputs:
        model_card_children.append(html.H5("Matriz de Confusión"))
        model_card_children.append(html.Img(src="data:image/png;base64,{}".format(model_outputs["cm_img"]), style={"width":"50%"}))
    if "roc_img" in model_outputs:
        model_card_children.append(html.H5("Curva ROC"))
        model_card_children.append(html.Img(src="data:image/png;base64,{}".format(model_outputs["roc_img"]), style={"width":"50%"}))
else:
    # If model didn't run or had error
    if "error" in model_outputs:
        model_card_children.append(html.Div([html.H5("Error al entrenar el modelo:"), html.Pre(model_outputs["error"])]))
    else:
        model_card_children.append(html.P("No se entrenó modelo XGBoost (falta información o ocurrió un error)."))

model_card = dbc.Card(dbc.CardBody(model_card_children))

# Limitations card
limit_card = dbc.Card(
    dbc.CardBody([
        html.H3("Limitaciones y Consideraciones"),
        html.Ul([
            html.Li("Datos con valores imposibles que requirieron limpieza agresiva."),
            html.Li("Periodo limitado a un año; no permite análisis interanual."),
            html.Li("Falta de variables como ozono y cobertura nubosa directa."),
            html.Li("Resolución horaria puede ocultar variaciones a escala sub-horaria.")
        ])
    ])
)

# Layout final con tabs y subtabs (estructura solicitada)
app.layout = dbc.Container([
    html.H1("Dashboard - Análisis de Radiación UVB en Chía, Cundinamarca", className="text-center my-3"),
    html.P("Autores: María Clara Ávila y Mateo José Giraldo", className="text-center text-muted"),

    dcc.Tabs([
        dcc.Tab(label="1. Introducción", children=[dbc.Container([intro_card, html.Br(), context_card], fluid=True)]),
        dcc.Tab(label="2. Contexto", children=[dbc.Container([context_card, html.Br(), problemas := problema_card], fluid=True)]),
        dcc.Tab(label="3. Planteamiento del Problema", children=[dbc.Container([problema_card], fluid=True)]),
        dcc.Tab(label="4. Objetivos y Justificación", children=[dbc.Container([objetivos_card], fluid=True)]),
        dcc.Tab(label="5. Marco Teórico", children=[dbc.Container([marco_card], fluid=True)]),
        dcc.Tab(label="6. Metodología", children=[dbc.Container([metodologia_card], fluid=True)]),
        dcc.Tab(label="7. Resultados y Análisis", children=[dbc.Container([eda_card, html.Br(), validacion_card, html.Br(), model_card, html.Br(), limit_card], fluid=True)]),
        # Tab 8 Conclusiones intentionally omitted per user request
    ])
], fluid=True)

# ---------------------------
# 8) SERVER RUN
# ---------------------------
if __name__ == "__main__":
    # Run with debug=False in production; for development you can set debug=True
    app.run_server(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))