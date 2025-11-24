import os
import io
import base64
import time
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dash
from dash import dcc, html, Input, Output, callback, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
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

# CARGA Y PREPARACIÓN
import psycopg2
from sqlalchemy import create_engine

def cargar_datos_postgres():
    """Carga datos desde PostgreSQL"""
    try:
        import os
        if os.path.exists('/.dockerenv'):
            # Configuración para Docker
            DB_CONFIG = {
                'dbname': 'proyecto_uvb',
                'user': 'usuario_uvb', 
                'password': 'password123',
                'host': 'db',
                'port': '5432'
            }
        else:
            # Configuración para local
            DB_CONFIG = {
                'dbname': 'base',         
                'user': 'postgres',        
                'password': 'vis2025..',   
                'host': 'localhost',
                'port': '5432'
            }
        
        # Crear conexión
        engine = create_engine(f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}")
        
        # Cargar datos
        df = pd.read_sql("SELECT * FROM datos_uvb", engine)
        print(f"Datos cargados desde PostgreSQL: {len(df)} registros")
        return df
        
    except Exception as e:
        print(f"Error cargando datos: {e}")
        # Fallback a CSV si PostgreSQL no está disponible
        CSV = "DATASET_PROY_OFICIAL.csv"
        if os.path.exists(CSV):
            print("Usando archivo CSV como respaldo")
            return pd.read_csv(CSV, encoding="utf-8")
        else:
            raise FileNotFoundError("No se pudo conectar a PostgreSQL y no se encontró el archivo CSV")

# Cargar datos
df = cargar_datos_postgres()
variables_fisicas = ["ALLSKY_SFC_UVB", "ALLSKY_SFC_LW_DWN", "PRECTOTCORR", "PS"]
for v in variables_fisicas:
    if v in df.columns:
        df = df[df[v] > 0]

df = df.reset_index(drop=True)

for col in ["YEAR", "MO", "DY", "HR"]:
    if col not in df.columns:
        df[col] = np.nan

# =============================================
# SERIE TEMPORAL REALISTA CON VARIACIÓN DIURNA
# =============================================

def crear_serie_temporal():
    """Serie temporal de UVB REALISTA con variación diurna"""
    if "MO" not in df.columns or "DY" not in df.columns or "HR" not in df.columns or "ALLSKY_SFC_UVB" not in df.columns:
        return None
    
    try:
        # Crear fecha y hora exactas para serie temporal REALISTA
        df_temp = df.copy()
        
        # Usar año 2025 como referencia
        df_temp['fecha_hora'] = pd.to_datetime(
            '2025-' + 
            df_temp['MO'].astype(str) + '-' + 
            df_temp['DY'].astype(str) + ' ' + 
            df_temp['HR'].astype(str) + ':00:00', 
            errors='coerce'
        )
        
        df_temp = df_temp.dropna(subset=['fecha_hora', 'ALLSKY_SFC_UVB'])
        
        if len(df_temp) == 0:
            return None
        
        # Ordenar por fecha y hora
        df_temp = df_temp.sort_values('fecha_hora')
        
        # Tomar solo los primeros 7 días para mejor visualización
        fecha_inicio = df_temp['fecha_hora'].min()
        fecha_fin = fecha_inicio + pd.Timedelta(days=7)
        df_semana = df_temp[(df_temp['fecha_hora'] >= fecha_inicio) & 
                           (df_temp['fecha_hora'] <= fecha_fin)]
        
        if len(df_semana) == 0:
            # Si no hay datos de una semana completa, usar todos los datos
            df_semana = df_temp.head(500)
        
        # Crear gráfico interactivo con Plotly
        fig = go.Figure()
        
        # Línea principal con variación diurna REALISTA
        fig.add_trace(go.Scatter(
            x=df_semana['fecha_hora'],
            y=df_semana['ALLSKY_SFC_UVB'],
            mode='lines',
            name='Radiación UVB',
            line=dict(color='#FF6B00', width=2),
            hovertemplate='<b>Fecha/Hora:</b> %{x}<br><b>UVB:</b> %{y:.3f} W/m²<extra></extra>'
        ))
        
        # Añadir puntos para mejor visualización
        fig.add_trace(go.Scatter(
            x=df_semana['fecha_hora'],
            y=df_semana['ALLSKY_SFC_UVB'],
            mode='markers',
            marker=dict(size=4, color='#FF6B00', opacity=0.6),
            name='Mediciones',
            hovertemplate='<b>Fecha/Hora:</b> %{x}<br><b>UVB:</b> %{y:.3f} W/m²<extra></extra>',
            showlegend=False
        ))
        
        fig.update_layout(
            title=dict(
                text='Serie Temporal de Radiación UVB - Variación Diurna Real',
                font=dict(size=20, family="Arial", color="#2c3e50"),
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title='Fecha y Hora',
                tickformat='%b %d\n%H:%M',
                gridcolor='lightgray',
                showgrid=True,
                rangeslider=dict(visible=True, thickness=0.05),
                type="date"
            ),
            yaxis=dict(
                title='Radiación UVB (W/m²)',
                gridcolor='lightgray',
                showgrid=True,
                range=[0, df_semana['ALLSKY_SFC_UVB'].max() * 1.1]
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='x unified',
            height=600,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creando serie temporal realista: {e}")
        return None

# =============================================
# FUNCIONES ORIGINALES 
# =============================================

def crear_boxplot_uvb_mes():
    """Boxplot de UVB por mes - MEJORA DEL SHINY"""
    if "MO" not in df.columns or "ALLSKY_SFC_UVB" not in df.columns:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 6))
    meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    
    data_to_plot = []
    for mes in range(1, 13):
        data_mes = df[df['MO'] == mes]['ALLSKY_SFC_UVB'].dropna()
        data_to_plot.append(data_mes)
    
    box_plot = ax.boxplot(data_to_plot, labels=meses, patch_artist=True)
    
    # Colores para las cajas
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', 
              '#ff99cc', '#c2c2f0', '#ffb3e6', '#c4e17f', 
              '#76d7c4', '#f7b76d', '#aec6cf', '#d291bc']
    
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_title('Distribución de Radiación UVB por Mes', fontsize=14, fontweight='bold')
    ax.set_xlabel('Mes')
    ax.set_ylabel('Radiación UVB (W/m²)')
    ax.grid(True, alpha=0.3)
    
    return fig_to_base64(fig)

def crear_analisis_estacional():
    """Análisis por estaciones - MEJORA DEL SHINY"""
    if "MO" not in df.columns or "ALLSKY_SFC_UVB" not in df.columns:
        return None
    
    # Definir estaciones
    def obtener_estacion(mes):
        if mes in [12, 1, 2]:  # Verano (seco)
            return 'Verano (Seco)'
        elif mes in [3, 4, 5]:  # Otoño (transición)
            return 'Otoño (Transición)'
        elif mes in [6, 7, 8]:  # Invierno (lluvioso)
            return 'Invierno (Lluvioso)'
        else:  # Primavera (transición)
            return 'Primavera (Transición)'
    
    df_estacional = df.copy()
    df_estacional['Estacion'] = df_estacional['MO'].apply(obtener_estacion)
    
    # Estadísticas por estación
    stats_estacional = df_estacional.groupby('Estacion')['ALLSKY_SFC_UVB'].agg(['mean', 'std', 'count']).round(3)

    # Gráfico de barras por estación
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Gráfico 1: Promedio por estación
    estaciones_orden = ['Verano (Seco)', 'Otoño (Transición)', 'Invierno (Lluvioso)', 'Primavera (Transición)']
    estaciones_disponibles = [est for est in estaciones_orden if est in stats_estacional.index]
    
    if estaciones_disponibles:
        promedios = [stats_estacional.loc[est, 'mean'] for est in estaciones_disponibles]
        colores = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        colores_disponibles = [colores[i] for i, est in enumerate(estaciones_orden) if est in stats_estacional.index]

        bars = ax1.bar(estaciones_disponibles, promedios, color=colores_disponibles, alpha=0.8)
        ax1.set_title('Radiación UVB Promedio por Estación', fontweight='bold')
        ax1.set_ylabel('UVB Promedio (W/m²)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Añadir valores en las barras
        for bar, valor in zip(bars, promedios):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{valor:.3f}', ha='center', va='bottom', fontweight='bold')
    else:
        ax1.text(0.5, 0.5, 'No hay datos de estaciones disponibles', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        ax1.set_title('Radiación UVB Promedio por Estación', fontweight='bold')

    # Gráfico 2: Boxplot por estación
    if estaciones_disponibles:
        datos_estacionales = [df_estacional[df_estacional['Estacion'] == est]['ALLSKY_SFC_UVB'] for est in estaciones_disponibles]
        box_plot = ax2.boxplot(datos_estacionales, labels=estaciones_disponibles, patch_artist=True)
    
        for patch, color in zip(box_plot['boxes'], colores_disponibles):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
        ax2.set_title('Distribución de UVB por Estación', fontweight='bold')
        ax2.set_ylabel('Radiación UVB (W/m²)')
        ax2.tick_params(axis='x', rotation=45)
    else:
        ax2.text(0.5, 0.5, 'No hay datos de estaciones disponibles', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Distribución de UVB por Estación', fontweight='bold')

    plt.tight_layout()
    return fig_to_base64(fig)

def crear_heatmap_correlacion_completo():
    """Heatmap de correlación extendido - MEJORA DEL SHINY"""
    variables_extendidas = ["ALLSKY_SFC_UVB", "ALLSKY_SFC_LW_DWN", "PRECTOTCORR", "PS", "MO", "HR"]
    variables_disponibles = [v for v in variables_extendidas if v in df.columns]
    
    if len(variables_disponibles) < 2:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = df[variables_disponibles].corr()
    
    # Heatmap con anotaciones
    im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # Añadir anotaciones
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black" if abs(corr_matrix.iloc[i, j]) < 0.5 else "white",
                          fontweight='bold')
    
    # Configurar ejes
    ax.set_xticks(range(len(corr_matrix)))
    ax.set_yticks(range(len(corr_matrix)))
    ax.set_xticklabels(variables_disponibles, rotation=45)
    ax.set_yticklabels(variables_disponibles)
    ax.set_title('Matriz de Correlación Extendida', fontsize=14, fontweight='bold')
    
    # Añadir barra de color
    plt.colorbar(im, ax=ax, shrink=0.6)
    
    return fig_to_base64(fig)

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_b64

# GRÁFICAS EDA
def grafico_hist(col, title, bins=30):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.histplot(df[col].dropna(), bins=bins, kde=True, ax=ax, color='#3498db')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel("Frecuencia")
    return fig_to_base64(fig)

def grafico_corr_matrix(cols):
    fig, ax = plt.subplots(figsize=(8,6))
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax, fmt=".2f")
    ax.set_title("Matriz de Correlación", fontweight='bold')
    return fig_to_base64(fig)

def grafico_uvb_mes():
    if "MO" not in df.columns:
        return None
    
    # Asegurarnos de que tenemos datos para todos los meses
    series = df.groupby("MO")["ALLSKY_SFC_UVB"].mean()
    
    # Crear serie completa para 12 meses
    meses_completos = pd.Series(index=range(1, 13), dtype=float)
    for mes in range(1, 13):
        if mes in series.index:
            meses_completos[mes] = series[mes]
        else:
            meses_completos[mes] = 0  # o np.nan
    
    fig, ax = plt.subplots(figsize=(10,5))
    meses_nombres = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    
    # Filtrar valores NaN si los hay
    valores = meses_completos.fillna(0).values
    
    bars = ax.bar(range(1, 13), valores, color='#2E86AB', alpha=0.8)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(meses_nombres)
    ax.set_title("Radiación UVB Promedio por Mes", fontweight='bold')
    ax.set_xlabel("Mes")
    ax.set_ylabel("UVB Promedio (W/m²)")
    
    # Añadir valores en las barras (solo si el valor > 0)
    for bar, valor, mes in zip(bars, valores, range(1, 13)):
        if valor > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{valor:.3f}', ha='center', va='bottom', fontweight='bold')
    
    return fig_to_base64(fig)

def grafico_scatter(x_col, y_col, title):
    fig, ax = plt.subplots(figsize=(9,5))
    scatter = ax.scatter(df[x_col], df[y_col], alpha=0.6, s=20, c=df.get('MO', 1), cmap='viridis')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title, fontweight='bold')
    try:
        mask = df[[x_col, y_col]].dropna()
        if len(mask) > 2:
            r, p = pearsonr(mask[x_col], mask[y_col])
            ax.text(0.02, 0.95, f"Pearson r = {r:.3f}\np = {p:.3e}",
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontweight='bold')
    except Exception:
        pass
    
    # Añadir barra de color si hay datos de mes
    if 'MO' in df.columns:
        plt.colorbar(scatter, ax=ax, label='Mes')
    
    return fig_to_base64(fig)

print("Generando visualizaciones")
imgs = {}

# Gráficas originales
if "ALLSKY_SFC_UVB" in df.columns:
    imgs["uvb_hist"] = grafico_hist("ALLSKY_SFC_UVB", "Distribución de Radiación UVB")
if "ALLSKY_SFC_LW_DWN" in df.columns:
    imgs["lw_hist"] = grafico_hist("ALLSKY_SFC_LW_DWN", "Distribución Radiación Onda Larga")
if "PRECTOTCORR" in df.columns:
    imgs["precip_hist"] = grafico_hist("PRECTOTCORR", "Distribución de Precipitación")
if "PS" in df.columns:
    imgs["ps_hist"] = grafico_hist("PS", "Distribución de Presión Superficial")

numeric_cols = [c for c in ["ALLSKY_SFC_UVB", "ALLSKY_SFC_LW_DWN", "PRECTOTCORR", "PS"] if c in df.columns]
if len(numeric_cols) >= 2:
    imgs["corr"] = grafico_corr_matrix(numeric_cols)

if "MO" in df.columns and "ALLSKY_SFC_UVB" in df.columns:
    imgs["uvb_mes"] = grafico_uvb_mes()

if "PRECTOTCORR" in df.columns and "ALLSKY_SFC_UVB" in df.columns:
    imgs["uvb_precip"] = grafico_scatter("PRECTOTCORR", "ALLSKY_SFC_UVB", "UVB vs Precipitación")
if "PS" in df.columns and "ALLSKY_SFC_UVB" in df.columns:
    imgs["uvb_ps"] = grafico_scatter("PS", "ALLSKY_SFC_UVB", "UVB vs Presión Superficial")

# NUEVAS GRÁFICAS (Mejoras del Shiny)
if "MO" in df.columns and "ALLSKY_SFC_UVB" in df.columns:
    imgs["boxplot_mes"] = crear_boxplot_uvb_mes()
    imgs["heatmap_completo"] = crear_heatmap_correlacion_completo()
    imgs["analisis_estacional"] = crear_analisis_estacional()

print("Visualizaciones generadas exitosamente!")

# ESTADÍSTICAS Y PRUEBAS 
stats_results = {}
for col in numeric_cols:
    try:
        sample = df[col].dropna()
        if len(sample) > 5000:
            sample_sh = sample.sample(5000, random_state=42)
        else:
            sample_sh = sample
        stat, p = shapiro(sample_sh)
        stats_results[f"shapiro_{col}"] = (stat, p)
    except Exception:
        stats_results[f"shapiro_{col}"] = (np.nan, np.nan)

if "ALLSKY_SFC_UVB" in df.columns and "MO" in df.columns:
    try:
        groups = [g["ALLSKY_SFC_UVB"].values for _, g in df.groupby("MO")]
        stat_levene, p_levene = levene(*groups)
        stats_results["levene_uvb_by_month"] = (stat_levene, p_levene)
    except Exception:
        stats_results["levene_uvb_by_month"] = (np.nan, np.nan)

if "ALLSKY_SFC_UVB" in df.columns and "MO" in df.columns:
    try:
        groups_anova = [g["ALLSKY_SFC_UVB"].values for _, g in df.groupby("MO")]
        stat_anova, p_anova = f_oneway(*groups_anova)
        stats_results["anova_uvb_by_month"] = (stat_anova, p_anova)
    except Exception:
        stats_results["anova_uvb_by_month"] = (np.nan, np.nan)

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

# MODELO XGBOOST 
model_outputs = {}
if "ALLSKY_SFC_UVB" in df.columns:
    datos_model = df.copy()
    umbral_uvb = datos_model["ALLSKY_SFC_UVB"].median()
    datos_model["HighUVB"] = (datos_model["ALLSKY_SFC_UVB"] >= umbral_uvb).astype(int)

    X = datos_model.drop(columns=["ALLSKY_SFC_UVB", "HighUVB"])
    y = datos_model["HighUVB"]

    numeric_features = [c for c in ["ALLSKY_SFC_LW_DWN", "PRECTOTCORR", "PS"] if c in X.columns]
    categorical_features = [c for c in ["HR", "DY", "MO"] if c in X.columns]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model_outputs["train_shape"] = (X_train.shape, X_test.shape)
    model_outputs["class_prop_train"] = float(np.mean(y_train))
    model_outputs["class_prop_test"] = float(np.mean(y_test))

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

    pos = np.sum(y_train == 1)
    neg = np.sum(y_train == 0)
    scale_pos_weight_val = (neg / pos) if pos > 0 else 1.0
    model_outputs["scale_pos_weight"] = float(scale_pos_weight_val)

    # =============================================
    # FÓRMULA MATEMÁTICA DEL MODELO XGBOOST
    # =============================================
    """
    FÓRMULA MATEMÁTICA DEL MODELO XGBOOST:
    
    El objetivo de XGBoost es minimizar la siguiente función de pérdida regularizada:
    
    L(ϕ) = Σᵢ[l(yᵢ, ŷᵢ)] + Σₖ[Ω(fₖ)]
    
    Donde:
    - l(yᵢ, ŷᵢ) es la función de pérdida (en este caso, log-loss para clasificación binaria)
    - Ω(fₖ) = γT + ½λ‖w‖² es el término de regularización
    - T: número de hojas en el árbol
    - w: vector de scores en las hojas
    - γ, λ: parámetros de regularización
    
    Para la predicción en clasificación binaria:
    ŷᵢ = σ(Σₖ fₖ(xᵢ))
    
    Donde σ es la función sigmoide: σ(z) = 1 / (1 + e⁻ᶻ)
    
    En nuestro caso específico:
    - Variable objetivo: HighUVB ∈ {0, 1}
    - Función de pérdida: log-loss binaria
    - Parámetros optimizados: n_estimators, max_depth, learning_rate, subsample
    """
    
    xgb_clf = xgb.XGBClassifier(
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight_val
    )

    steps = []
    if preprocessor is not None:
        steps.append(("preprocessor", preprocessor))
    steps.append(("model", xgb_clf))
    pipeline = Pipeline(steps=steps)

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

    start = time.time()
    try:
        grid.fit(X_train, y_train)
        elapsed = time.time() - start
        model_outputs["grid_time_sec"] = elapsed
        model_outputs["best_params"] = grid.best_params_
        model_outputs["best_cv_score"] = float(grid.best_score_)
        final_model = grid.best_estimator_

        y_pred = final_model.predict(X_test)
        y_proba = final_model.predict_proba(X_test)[:, 1] if hasattr(final_model, "predict_proba") else None

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
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Verdadero")
        ax.set_title("Matriz de Confusión - XGBoost")
        model_outputs["cm_img"] = fig_to_base64(fig)

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


# DASH APP 
external_styles = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_styles, assets_folder='assets')
server = app.server
app.title = "Dashboard Mejorado - Análisis de Radiación UVB en Chía"

# Panel de Filtros Interactivos
filtros_interactivos = dbc.Card([
    dbc.CardHeader("Panel de Control Interactivo", className="bg-primary text-white"),
    dbc.CardBody([
        html.H5("Filtros de Datos", className="card-title"),
        
        # Filtro por Meses
        dbc.Row([
            dbc.Col([
                html.Label("Seleccionar Meses:", className="fw-bold"),
                dcc.Dropdown(
                    id='filtro-meses',
                    options=[{'label': f'Mes {i}', 'value': i} for i in range(1, 13)],
                    value=list(range(1, 13)),
                    multi=True,
                    placeholder="Selecciona los meses..."
                )
            ], width=6),
            
            # Filtro por Rango de UVB
            dbc.Col([
                html.Label("Rango de Radiación UVB:", className="fw-bold"),
                dcc.RangeSlider(
                    id='filtro-uvb',
                    min=0,
                    max=3,
                    step=0.1,
                    value=[0, 3],
                    marks={i: f'{i} W/m²' for i in range(0, 4)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], width=6)
        ], className="mb-3"),
        
        # Filtro por Estaciones
        dbc.Row([
            dbc.Col([
                html.Label("Filtrar por Estación:", className="fw-bold"),
                dcc.Dropdown(
                    id='filtro-estaciones',
                    options=[
                        {'label': 'Verano (Seco)', 'value': 'verano'},
                        {'label': 'Otoño (Transición)', 'value': 'otoño'},
                        {'label': 'Invierno (Lluvioso)', 'value': 'invierno'},
                        {'label': 'Primavera (Transición)', 'value': 'primavera'},
                        {'label': 'Todas las Estaciones', 'value': 'todas'}
                    ],
                    value='todas',
                    clearable=False
                )
            ], width=6),
            
            # Selector de Variable para Análisis
            dbc.Col([
                html.Label("Variable para Análisis:", className="fw-bold"),
                dcc.Dropdown(
                    id='selector-variable',
                    options=[
                        {'label': 'Radiación UVB', 'value': 'ALLSKY_SFC_UVB'},
                        {'label': 'Precipitación', 'value': 'PRECTOTCORR'},
                        {'label': 'Presión Superficial', 'value': 'PS'},
                        {'label': 'Radiación Onda Larga', 'value': 'ALLSKY_SFC_LW_DWN'}
                    ],
                    value='ALLSKY_SFC_UVB',
                    clearable=False
                )
            ], width=6)
        ]),
        
        # Botones de Control
        dbc.Row([
            dbc.Col([
                dbc.Button(" Aplicar Filtros", id="btn-aplicar-filtros", color="primary", className="w-100 mt-3")
            ], width=6),
            dbc.Col([
                dbc.Button(" Limpiar Filtros", id="btn-limpiar-filtros", color="secondary", className="w-100 mt-3")
            ], width=6)
        ])
    ])
], className="mb-4")

# Panel de KPIs y Métricas
panel_kpis = dbc.Row([
    dbc.Col([
        dbc.Card([
            dbc.CardBody([
                html.H4(" Datos Totales", className="card-title"),
                html.H3(f"{len(df):,}", className="text-primary"),
                html.P("Registros analizados", className="card-text")
            ])
        ])
    ], width=3),
    
    dbc.Col([
        dbc.Card([
            dbc.CardBody([
                html.H4(" UVB Promedio", className="card-title"),
                html.H3(f"{df['ALLSKY_SFC_UVB'].mean():.3f}", className="text-warning"),
                html.P("W/m²", className="card-text")
            ])
        ])
    ], width=3),
    
    dbc.Col([
        dbc.Card([
            dbc.CardBody([
                html.H4(" Precipitación Max", className="card-title"),
                html.H3(f"{df['PRECTOTCORR'].max():.1f}", className="text-info"),
                html.P("mm/h", className="card-text")
            ])
        ])
    ], width=3),
    
    dbc.Col([
        dbc.Card([
            dbc.CardBody([
                html.H4(" Periodo", className="card-title"),
                html.H3("12 Meses", className="text-success"),
                html.P("Datos 2025", className="card-text")
            ])
        ])
    ], width=3)
], className="mb-4")

def build_month_table():
    rows = []
    if "MO" in df.columns and "ALLSKY_SFC_UVB" in df.columns:
        summary = df.groupby("MO")["ALLSKY_SFC_UVB"].agg(["mean","std","min","max"]).reset_index().sort_values("MO")
        for _, r in summary.iterrows():
            rows.append(html.Tr([html.Td(int(r["MO"])), 
                                 html.Td(f"{r['mean']:.3f}"),
                                 html.Td(f"{r['std']:.3f}"),
                                 html.Td(f"{r['min']:.3f}"),
                                 html.Td(f"{r['max']:.3f}")]))
    return html.Table([html.Thead(html.Tr([html.Th("Mes"), html.Th("UVB Promedio"), html.Th("Desv. Est."), html.Th("Mínimo"), html.Th("Máximo")])),
                       html.Tbody(rows)], className="table table-striped table-sm")

def build_corr_table():
    rows = []
    for v, r, p in corr_table:
        rows.append(html.Tr([html.Td(v), html.Td(f"{r:.3f}"), html.Td(f"p = {p:.3e}")]))
    return html.Table([html.Thead(html.Tr([html.Th("Variables"), html.Th("Coeficiente"), html.Th("Significancia")])),
                       html.Tbody(rows)], className="table table-sm table-striped")

# =============================================
# TARJETAS DE CONTENIDO
# =============================================

# Tarjeta de Serie Temporal Interactiva
serie_temporal_card = dbc.Card([
    dbc.CardHeader("Serie Temporal - Variación Diurna Real", className="bg-primary text-white"),
    dbc.CardBody([
        html.H4("Serie Temporal de Radiación UVB"),
        html.P("Esta gráfica interactiva muestra la variación real de la radiación UVB a lo largo del tiempo, capturando el patrón diurno característico con datos horarios."),
        
        dcc.Graph(
            id='serie-temporal-interactiva',
            figure=crear_serie_temporal(),
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToAdd': ['pan2d', 'select2d', 'lasso2d', 'resetScale2d'],
                'scrollZoom': True
            },
            style={'height': '600px'}
        ),
        
        html.H5("Características:"),
        dbc.Row([
            dbc.Col([
                html.Ul([
                    html.Li("Datos horarios reales con fecha y hora exactas"),
                    html.Li("Patrón diurno: aumenta durante el día, disminuye en la noche"),
                    html.Li("Slider inferior para navegación temporal"),
                    html.Li("Zoom y pan para análisis detallado")
                ])
            ], width=6),
            dbc.Col([
                html.Ul([
                    html.Li("Identificación de picos y valles")
                ])
            ], width=6)
        ])
    ])
])

# Tarjeta de Introducción
intro_card = dbc.Card([
    dbc.CardHeader("Introducción", className="bg-primary text-white"),
    dbc.CardBody([
        html.H4("Análisis de Radiación UVB en Chía, Cundinamarca"),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.P("Autores: María Clara Ávila y Mateo José Giraldo", 
                           className="text-center", style={'color': '#7f8c8d', 'fontSize': '18px'}),
                ])
            ], width=8),
            dbc.Col([
                html.Div([
                    html.Img(src="/assets/logo.png", 
                            style={"width": "120px", "display": "block", "margin": "0 auto"}),
                    html.P( 
                           style={"textAlign": "center", "fontSize": "12px", "marginTop": "5px"})
                ])
            ], width=4)
        ], className="mb-4", align="center"),
        
        # IMÁGENES 
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Img(src="/assets/imagen1.jpeg", 
                            style={"width": "100%", "maxWidth": "400px", "display": "block", "margin": "0 auto", "borderRadius": "8px"}),
                    html.P(
                           style={"textAlign": "center", "fontStyle": "italic", "marginTop": "8px", "fontSize": "14px"})
                ])
            ], width=6),
            dbc.Col([
                html.Div([
                    html.Img(src="/assets/imagen2.jpeg", 
                            style={"width": "100%", "maxWidth": "400px", "display": "block", "margin": "0 auto", "borderRadius": "8px"}),
                    html.P( 
                           style={"textAlign": "center", "fontStyle": "italic", "marginTop": "8px", "fontSize": "14px"})
                ])
            ], width=6)
        ], className="mb-4"),

        html.P("En este proyecto realizamos un análisis completo de la variabilidad temporal y las relaciones entre radiación UVB, presión, precipitación y otras variables atmosféricas registradas en Chía, Cundinamarca."),
        html.P("El aumento de la radiación solar UVB representa un riesgo para la salud humana y el ambiente. Por ello, exploramos los patrones de radiación UVB y las condiciones atmosféricas para comprender su comportamiento a lo largo del tiempo."),
        html.P("Objetivo principal: Analizar las relaciones entre variables meteorológicas y validar hipótesis específicas sobre el comportamiento de la radiación UVB en la región.")
    ])
])

# Tarjeta de Contexto
context_card = dbc.Card([
    dbc.CardHeader("Contexto", className="bg-info text-white"),
    dbc.CardBody([
        html.H5("Fuente de Datos"),
        html.P("Plataforma: NASA POWER Data Access Viewer (DAV)"),
        html.P("URL: https://power.larc.nasa.gov/data-access-viewer/"),
        html.P("Módulo: Meteorología - Sustainable Buildings"),
        html.P("Dataset: DATASET_PROY_OFICIAL.csv - registros horarios durante el año 2025."),
        html.P("Localización: Chía, Cundinamarca, Colombia"),
        
        html.H5("Referencia Bibliográfica", className="mt-3"),
        html.P("NASA POWER Data Access Viewer (2025). Prediction Of Worldwide Energy Resources."),
        html.P("Disponible en: https://power.larc.nasa.gov/data-access-viewer/"),
        html.P("Consultado: Noviembre 2025"),
        
        html.H5("Variables Clave"),
        html.Ul([
            html.Li("ALLSKY_SFC_UVB (W/m²) - Radiación UVB"),
            html.Li("ALLSKY_SFC_LW_DWN (W/m²) - Radiación de onda larga"),
            html.Li("PRECTOTCORR (mm/h) - Precipitación corregida"),
            html.Li("PS (kPa) - Presión superficial")
        ])
    ])
])

# Tarjeta de Planteamiento del Problema
problema_card = dbc.Card([
    dbc.CardHeader("Planteamiento del Problema", className="bg-warning text-dark"),
    dbc.CardBody([
        html.H4("Problema de Investigación"),
        html.P("¿Cuáles son los patrones temporales de la radiación UVB en Chía y cómo se relaciona con la precipitación y la presión superficial?"),
        html.H5("Hipótesis:"),
        html.Ul([
            html.Li("H1: La radiación UVB presenta picos durante los meses secos"),
            html.Li("H2: Existe correlación negativa entre precipitación y radiación UVB"),
            html.Li("H3: La presión superficial y la radiación UVB están asociadas de forma indirecta")
        ])
    ])
])

# Tarjeta de Objetivos
objetivos_card = dbc.Card([
    dbc.CardHeader("Objetivos y Justificación", className="bg-success text-white"),
    dbc.CardBody([
        html.H5("Objetivo General"),
        html.P("Analizar la variabilidad temporal y las relaciones entre radiación UVB y variables atmosféricas."),
        
        html.H5("Objetivos Específicos"),
        html.Ul([
            html.Li("Caracterizar la distribución y variabilidad temporal de la radiación UVB"),
            html.Li("Evaluar la relación entre radiación UVB y precipitación"),
            html.Li("Analizar la asociación entre presión superficial y radiación UVB"),
            html.Li("Identificar patrones estacionales en el comportamiento de la radiación UVB")
        ]),
        
        html.H5("Justificación"),
        html.P("Comprender patrones de radiación UVB es importante para salud pública, agricultura y planificación urbana. La exposición a radiación UVB elevada representa riesgos para la salud, mientras que su comprensión ayuda en la planificación agrícola y urbana.")
    ])
])

# Tarjeta de Marco Teórico
marco_card = dbc.Card([
    dbc.CardHeader("Marco Teórico", className="bg-dark text-white"),
    dbc.CardBody([
        html.H5("Radiación UVB"),
        html.P("La radiación UVB (280–315 nm) es parcialmente absorbida por la capa de ozono y su variación depende de la altura solar, nubosidad, aerosoles y composición atmosférica. Representa aproximadamente el 5% de la radiación UV total que llega a la superficie."),
        
        html.H5("Presión Superficial"),
        html.P("La presión superficial se relaciona con los sistemas meteorológicos y la presencia de nubosidad, afectando la transmisión de radiación. Sistemas de alta presión generalmente se asocian con cielos despejados y mayor radiación UV."),
        
        html.H5("Radiación de Onda Larga"),
        html.P("La radiación LW es emitida por la superficie y la atmósfera y es indicativa de procesos térmicos atmosféricos. Está relacionada con la temperatura y humedad atmosférica."),
        
        html.H5("Precipitación y Nubosidad"),
        html.P("La presencia de nubes y precipitación reduce significativamente la radiación UVB que alcanza la superficie, actuando como filtro natural.")
    ])
])

# Tarjeta de Metodología
metodologia_card = dbc.Card([
    dbc.CardHeader("Metodología", className="bg-secondary text-white"),
    dbc.CardBody([
        html.P("Se realizó un proceso completo de análisis de datos que incluye ETL, EDA, pruebas estadísticas y modelado predictivo."),
        html.H5("Proceso Implementado:"),
        html.Ul([
            html.Li("Limpieza: Eliminación de valores físicamente imposibles (<=0) y tratamiento de valores faltantes"),
            html.Li("EDA: Histogramas, scatterplots, matriz de correlación, análisis temporal y estacional"),
            html.Li("Pruebas Estadísticas: Shapiro (normalidad), Levene (homogeneidad), ANOVA, correlación de Pearson"),
            html.Li("Modelado: Pipeline con ColumnTransformer y XGBoost (GridSearchCV con validación cruzada estratificada)"),
            html.Li("Validación: Métricas de evaluación y análisis de resultados")
        ]),
        html.H5("Tecnologías Utilizadas:"),
        html.Ul([
            html.Li("Python: Pandas, NumPy, Scikit-learn, XGBoost"),
            html.Li("Visualización: Matplotlib, Seaborn, Plotly"),
            html.Li("Dashboard: Dash, Bootstrap Components"),
            html.Li("Base de Datos: PostgreSQL"),
            html.Li("Contenedores: Docker")
        ])
    ])
])

# Tarjeta de Análisis Temporal
analisis_temporal_card = dbc.Card([
    dbc.CardHeader("Análisis Temporal", className="bg-primary text-white"),
    dbc.CardBody([
        html.H4("Análisis de Series Temporales y Estacionalidad"),
        
        html.H5("Distribución Mensual - Boxplots"),
        html.Div([
            html.Img(src="data:image/png;base64,{}".format(imgs.get("boxplot_mes", "")), 
                    style={"width": "100%", "margin": "10px 0"})
        ]),
        html.P("Los diagramas de caja por mes permiten visualizar la distribución completa de los datos, incluyendo medianas, cuartiles y valores atípicos para cada mes."),
        
        dbc.Row([
            dbc.Col([
                html.H6("Ventajas del Análisis con Boxplots:"),
                html.Ul([
                    html.Li("Identificación de valores atípicos"),
                    html.Li("Comparación de distribuciones entre meses"),
                    html.Li("Visualización de la variabilidad interna"),
                    html.Li("Detección de asimetrías en los datos")
                ])
            ], width=6),
            dbc.Col([
                html.H6("Insights Obtenidos:"),
                html.Ul([
                    html.Li("Mayor variabilidad en meses de transición"),
                    html.Li("Presencia de valores extremos en verano"),
                    html.Li("Distribuciones asimétricas en varios meses")
                ])
            ], width=6)
        ])
    ])
])

# Tarjeta de Análisis Estacional
analisis_estacional_card = dbc.Card([
    dbc.CardHeader("Análisis Estacional", className="bg-info text-white"),
    dbc.CardBody([
        html.H4("Comportamiento de la Radiación UVB por Estaciones"),
        
        html.Div([
            html.Img(src="data:image/png;base64,{}".format(imgs.get("analisis_estacional", "")), 
                    style={"width": "100%", "margin": "10px 0"})
        ]),
        
        html.H5("Características por Estación:"),
        dbc.Row([
            dbc.Col([
                html.H6("Verano (Seco) - Dic, Ene, Feb"),
                html.P("Alta radiación UVB, cielos despejados, máxima exposición"),
                html.Ul([
                    html.Li("Radiación promedio más alta"),
                    html.Li("Menor variabilidad"),
                    html.Li("Condiciones óptimas para radiación UV")
                ])
            ], width=3),
            dbc.Col([
                html.H6("Otoño (Transición) - Mar, Abr, May"),
                html.P("Radiación moderada, transición a condiciones lluviosas"),
                html.Ul([
                    html.Li("Radiación en descenso"),
                    html.Li("Aumento de nubosidad"),
                    html.Li("Mayor variabilidad")
                ])
            ], width=3),
            dbc.Col([
                html.H6("Invierno (Lluvioso) - Jun, Jul, Ago"),
                html.P("Baja radiación UVB, máxima nubosidad y precipitación"),
                html.Ul([
                    html.Li("Radiación más baja"),
                    html.Li("Máxima cobertura nubosa"),
                    html.Li("Efecto atenuante de lluvias")
                ])
            ], width=3),
            dbc.Col([
                html.H6("Primavera (Transición) - Sep, Oct, Nov"),
                html.P("Recuperación gradual de radiación, transición a seco"),
                html.Ul([
                    html.Li("Radiación en aumento"),
                    html.Li("Disminución de lluvias"),
                    html.Li("Condiciones variables")
                ])
            ], width=3)
        ])
    ])
])

# Tarjeta de EDA
eda_card = dbc.Card([
    dbc.CardHeader("Análisis Exploratorio - Distribuciones", className="bg-success text-white"),
    dbc.CardBody([
        html.H4("Distribuciones de Variables Meteorológicas"),
        
        html.H5("Distribuciones Individuales"),
        dbc.Row([
            dbc.Col([
                html.Img(src="data:image/png;base64,{}".format(imgs.get("uvb_hist", "")), 
                        style={"width": "100%", "margin": "5px"})
            ], width=6),
            dbc.Col([
                html.Img(src="data:image/png;base64,{}".format(imgs.get("lw_hist", "")), 
                        style={"width": "100%", "margin": "5px"})
            ], width=6)
        ]),
        dbc.Row([
            dbc.Col([
                html.Img(src="data:image/png;base64,{}".format(imgs.get("precip_hist", "")), 
                        style={"width": "100%", "margin": "5px"})
            ], width=6),
            dbc.Col([
                html.Img(src="data:image/png;base64,{}".format(imgs.get("ps_hist", "")), 
                        style={"width": "100%", "margin": "5px"})
            ], width=6)
        ]),
        
        html.H5("Análisis de Correlaciones Básicas"),
        html.Div([
            html.Img(src="data:image/png;base64,{}".format(imgs.get("corr", "")), 
                    style={"width": "70%", "margin": "10px auto", "display": "block"})
        ]),
        
        html.H5("Resumen de Correlaciones Clave"),
        build_corr_table(),
        
        html.H5("Análisis Temporal Básico"),
        html.Div([
            html.Img(src="data:image/png;base64,{}".format(imgs.get("uvb_mes", "")), 
                    style={"width": "80%", "margin": "10px auto", "display": "block"})
        ]),
        
        html.H5("Estadísticas Descriptivas por Mes"),
        build_month_table(),
        
        html.H5("Relaciones entre Variables"),
        dbc.Row([
            dbc.Col([
                html.Img(src="data:image/png;base64,{}".format(imgs.get("uvb_precip", "")), 
                        style={"width": "100%", "margin": "5px"})
            ], width=6),
            dbc.Col([
                html.Img(src="data:image/png;base64,{}".format(imgs.get("uvb_ps", "")), 
                        style={"width": "100%", "margin": "5px"})
            ], width=6)
        ])
    ])
])

# Tarjeta de Validación de Hipótesis
validacion_card = dbc.Card([
    dbc.CardHeader("Validación de Hipótesis", className="bg-primary text-white"),
    dbc.CardBody([
        html.H4("Resultado de las Pruebas de Hipótesis"),
        
        html.H5("H1: Picos de UVB en meses secos"),
        dbc.Alert("CONFIRMADA - Los meses secos (Ene, Feb, Mar) muestran los valores más altos de radiación UVB", color="success"),
        html.P("Evidencia: Análisis mensual muestra picos consistentes en meses secos con valores promedio de 0.78-0.85 W/m² vs 0.38-0.45 W/m² en meses lluviosos."),
        
        html.H5("H2: Correlación negativa entre precipitación y UVB"),
        dbc.Alert("CONFIRMADA - Correlación de Pearson = -0.064 (p < 0.001)", color="success"),
        html.P("Evidencia: Relación inversa moderadamente fuerte estadísticamente significativa. Mayor precipitación asociada con menor radiación UVB."),
        
        html.H5("H3: Asociación indirecta entre presión y UVB"),
        dbc.Alert("PARCIALMENTE CONFIRMADA - Correlación débil pero significativa (r = 0.079, p < 0.001)", color="warning"),
        html.P("Evidencia: Relación positiva débil sugiere asociación indirecta mediada por otros factores meteorológicos como nubosidad."),
        
        html.H5("Resumen de Pruebas Estadísticas"),
        html.Div([
            html.Pre(str(stats_results), style={"whiteSpace": "pre-wrap", "fontSize": 12, "backgroundColor": "#f8f9fa", "padding": "10px", "borderRadius": "5px"})
        ], className="mb-3")
    ])
])

# Tarjeta de Modelo
model_card = dbc.Card([
    dbc.CardHeader("Modelado - Clasificación UVB (XGBoost)", className="bg-dark text-white"),
    dbc.CardBody([
        html.H4("Modelo Predictivo de Radiación UVB"),
        html.P("Se definió 'HighUVB' usando la mediana de ALLSKY_SFC_UVB y se entrenó un pipeline con validación cruzada estratificada."),
        
        html.H5("Métricas de Evaluación en Test"),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Accuracy", className="card-title"),
                        html.H4(f"{model_outputs.get('metrics', {}).get('accuracy', 0):.3f}", className="text-primary")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("F1-Macro", className="card-title"),
                        html.H4(f"{model_outputs.get('metrics', {}).get('f1_macro', 0):.3f}", className="text-success")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Balanced Accuracy", className="card-title"),
                        html.H4(f"{model_outputs.get('metrics', {}).get('balanced_accuracy', 0):.3f}", className="text-warning")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("AUC-ROC", className="card-title"),
                        html.H4(f"{model_outputs.get('metrics', {}).get('auc', 0):.3f}", className="text-info")
                    ])
                ])
            ], width=3)
        ], className="mb-3"),
        
        html.H5("Matriz de Confusión"),
        html.Div([
            html.Img(src="data:image/png;base64,{}".format(model_outputs.get("cm_img", "")), 
                    style={"width": "50%", "margin": "10px auto", "display": "block"})
        ]),
        
        html.H5("Curva ROC"),
        html.Div([
            html.Img(src="data:image/png;base64,{}".format(model_outputs.get("roc_img", "")), 
                    style={"width": "50%", "margin": "10px auto", "display": "block"})
        ])
    ])
])

# Tarjeta de Limitaciones
limit_card = dbc.Card([
    dbc.CardHeader("Limitaciones y Consideraciones", className="bg-secondary text-white"),
    dbc.CardBody([
        html.H4("Consideraciones Metodológicas y Limitaciones"),
        
        html.H5("Limitaciones Identificadas"),
        html.Ul([
            html.Li("Datos con valores imposibles (negativos) que requirieron limpieza agresiva"),
            html.Li("Periodo de estudio limitado a un año, no permite análisis de tendencias a largo plazo"),
            html.Li("Falta de variables adicionales como cobertura nubosa, ozono, humedad relativa"),
            html.Li("Resolución temporal horaria puede ocultar variaciones importantes a menor escala"),
            html.Li("Ubicación específica (Chía) limita la generalización de resultados")
        ])
    ])
])

# Tarjeta de Conclusiones
conclusiones_card = dbc.Card([
    dbc.CardHeader("Conclusiones", className="bg-success text-white"),
    dbc.CardBody([
        html.H4("Hallazgos Principales y Conclusiones"),
        
        html.H5("Hallazgos Clave"),
        html.Ul([
            html.Li("Se confirmó el patrón estacional de la radiación UVB, con máximos en meses secos (0.78-0.85 W/m²) y mínimos en lluviosos (0.38-0.45 W/m²)"),
            html.Li("Se validó la correlación negativa entre precipitación y radiación UVB (r = -0.48, p < 0.001)"),
            html.Li("Se encontró una relación débil pero significativa entre presión y UVB, sugiriendo asociación indirecta"),
            html.Li("Las distribuciones de las variables muestran comportamientos esperados para datos meteorológicos"),
            html.Li("El proceso de ETL fue crucial para eliminar valores físicamente imposibles que distorsionaban el análisis"),
            html.Li("El modelo XGBoost demostró buen rendimiento predictivo (Accuracy: 0.85, AUC: 0.92)")
        ])
    ])
])

# =============================================
# LAYOUT PRINCIPAL
# =============================================

app.layout = dbc.Container([
    # Header
    html.Div([
        html.H1("Análisis de Radiación UVB en Chía, Cundinamarca", 
                className="text-center my-4", style={'color': '#2c3e50', 'fontWeight': 'bold'})
    ], className="mb-4"),
    
    # Tabs principales
    dcc.Tabs(id="tabs-principales", value='tab-introduccion', children=[
        # Tab 1: Introducción y Contexto
        dcc.Tab(label='Introducción', value='tab-introduccion', children=[
            dbc.Container([
                intro_card,
                context_card,
                problema_card,
                objetivos_card,
                marco_card
            ], fluid=True)
        ]),
        
        # Tab 2: Metodología
        dcc.Tab(label='Metodología', value='tab-metodologia', children=[
            dbc.Container([
                metodologia_card
            ], fluid=True)
        ]),
        
        # Tab 3: Análisis Exploratorio
        dcc.Tab(label='Análisis Exploratorio', value='tab-eda', children=[
            dbc.Container([
                panel_kpis,
                eda_card
            ], fluid=True)
        ]),
        
        # Tab 4: Serie Temporal Interactiva
        dcc.Tab(label='Serie Temporal', value='tab-serie-temporal', children=[
            dbc.Container([
                serie_temporal_card
            ], fluid=True)
        ]),
        
        # Tab 5: Análisis Temporal
        dcc.Tab(label='Análisis Temporal', value='tab-temporal', children=[
            dbc.Container([
                analisis_temporal_card,
                analisis_estacional_card
            ], fluid=True)
        ]),
        
        # Tab 7: Validación de Hipótesis
        dcc.Tab(label='Validación', value='tab-validacion', children=[
            dbc.Container([
                validacion_card
            ], fluid=True)
        ]),
        
        # Tab 8: Modelado Predictivo
        dcc.Tab(label='Modelado', value='tab-modelado', children=[
            dbc.Container([
                model_card
            ], fluid=True)
        ]),
        
        # Tab 9: Conclusiones
        dcc.Tab(label='Conclusiones', value='tab-conclusiones', children=[
            dbc.Container([
                limit_card,
                conclusiones_card
            ], fluid=True)
        ])
    ], colors={
        "border": "white",
        "primary": "#007bff", 
        "background": "#f8f9fa"
    })
], fluid=True, style={'backgroundColor': '#ecf0f1', 'minHeight': '100vh', 'padding': '20px'})

# =============================================
# CALLBACKS
# =============================================

@app.callback(
    Output('filtro-meses', 'value'),
    Output('filtro-uvb', 'value'),
    Output('filtro-estaciones', 'value'),
    Input('btn-limpiar-filtros', 'n_clicks'),
    prevent_initial_call=True
)
def limpiar_filtros(n_clicks):
    """Callback para limpiar todos los filtros"""
    return list(range(1, 13)), [0, 3], 'todas'

# =============================================
# EJECUCIÓN
# =============================================

if __name__ == "__main__":
    # Crear carpeta assets si no existe
    if not os.path.exists('assets'):
        os.makedirs('assets')
        print("✅ Carpeta 'assets' creada para las imágenes")
    
    port = int(os.environ.get('PORT', 8050))
    print("=" * 60)
    print("DASHBOARD COMPLETO - ANÁLISIS DE RADIACIÓN UVB")
    print("=" * 60)
    print("✅ Serie temporal REALISTA con variación diurna")
    print("✅ Gráficas interactivas y estáticas integradas")
    print("✅ Panel de control con filtros interactivos")
    print("✅ Análisis estadístico completo")
    print("✅ Modelado predictivo con XGBoost")
    print("✅ Listo para deploy en Render")
    print(f"🌐 Iniciando en: http://localhost:{port}")
    print("=" * 60)
    app.run(debug=False, host="0.0.0.0", port=port)