import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tkinter as tk
from tkinter import ttk, Toplevel, Text, Scrollbar, messagebox, font, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import datetime
import calendar
import os
from scipy.interpolate import CubicSpline

# --- IMPORTACIONES CIENTÍFICAS ---
# Implementación basada en Tesis Miranda Chinlli (SARIMA)
try:
    import statsmodels.api as sm
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_DISPONIBLE = True
except ImportError:
    STATSMODELS_DISPONIBLE = False
    print("Advertencia: statsmodels no instalado.")
try:
    from PIL import Image, ImageTk
    PIL_DISPONIBLE = True
except ImportError:
    PIL_DISPONIBLE = False

NOMBRE_ARCHIVO_DATOS = None
NOMBRE_BASE_DISPLAY = "PENDIENTE"
DF_GLOBAL = None

patron_tmax_g, patron_tmin_g, slope_tmax_g, slope_tmin_g, intercept_tmax_g, intercept_tmin_g, mae_tmax_g, mae_tmin_g = None, None, None, None, None, None, None, None
SERIE_MENSUAL_TMAX = None
SERIE_MENSUAL_TMIN = None
SERIE_ANUAL_MAX_TMAX = None
SERIE_ANUAL_MIN_TMIN = None

# Rutas de imágenes (ajusta según tu entorno si es necesario)
NOMBRE_IMAGEN = r'C:\Users\LENOVO\Desktop\RECETAS PROMODEL\imagen1.png'
RUTA_EMOJI_CALIENTE = r'C:\Users\LENOVO\Desktop\RECETAS PROMODEL\EMOJICALIENTE.jpg'
RUTA_EMOJI_FRIO = r'C:\Users\LENOVO\Desktop\RECETAS PROMODEL\EMOJIFRIO.jpg'
RUTA_EMOJI_MEDIA = r'C:\Users\LENOVO\Desktop\RECETAS PROMODEL\EMOJIMEDIA.jpg'

BG_COLOR = '#1a1a1a'
TEXT_COLOR_GREEN = '#00ff00'
TEXT_COLOR_BLUE = '#00ffff'
TEXT_COLOR_RED = '#ff4d4d'
TEXT_COLOR_WHITE = '#ffffff'
SELECT_BG = '#333333'
BG_GRAFICA = '#2b2b2b'

DEFAULT_FONT = ("Courier New", 14)
DEFAULT_FONT_BOLD = ("Courier New", 14, "bold")
TITLE_FONT = ("Courier New", 30, "bold")
SUB_TITLE_FONT = ("Courier New", 22, "bold")
DESCRIPTION_FONT = ("Courier New", 18, "italic")
DATA_FONT = ("Courier New", 12)
CALC_FONT = ("Courier New", 11)
CALC_HEADER_FONT = ("Courier New", 16, "bold", 'underline')
LISTBOX_FONT = ("Courier New", 14)
EMOJI_TEXT_FONT = ("Courier New", 18, "bold")
background_photo_dict = {}
emoji_photo_dict = {}

ECUACION_DISPLAY_INFO = ""
SLOPE_MENSUAL_DISPLAY = 0.0
ANO_BASE_DISPLAY = 2000
RIESGO_GUMBEL_TEXTO = ""

def cargar_datos_y_calcular_tendencias():
    global DF_GLOBAL, SERIE_MENSUAL_TMAX, SERIE_MENSUAL_TMIN, SERIE_ANUAL_MAX_TMAX, SERIE_ANUAL_MIN_TMIN
    if NOMBRE_ARCHIVO_DATOS is None:
        return None, None, 0, 0, 0, 0, 0, 0
    try:
        df = pd.read_csv(NOMBRE_ARCHIVO_DATOS, on_bad_lines='skip')
        df = df.astype(str)
        df.columns = [c.strip() for c in df.columns]
        df = df[df['Fecha'].str.strip() != 'Fecha']
        df['Tmax'] = pd.to_numeric(df['Tmax'], errors='coerce')
        df['Tmin'] = pd.to_numeric(df['Tmin'], errors='coerce')
        
        fecha_convertida = None
        for fmt in ('%d/%m/%Y', '%Y-%m-%d', '%m/%d/%Y'):
            try:
                temp_fechas = pd.to_datetime(df['Fecha'], format=fmt, errors='coerce')
                if temp_fechas.notnull().sum() > 0:
                    fecha_convertida = temp_fechas
                    break
            except:
                continue
        if fecha_convertida is not None:
            df['Fecha'] = fecha_convertida
        else:
            df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
        df = df.dropna(subset=['Tmax', 'Tmin', 'Fecha'])
        df = df.sort_values('Fecha')
        DF_GLOBAL = df.copy()

        # --- PREPARACIÓN DE SERIES PARA ANÁLISIS ESTOCÁSTICO ---
        # Tesis Miranda Chinlli: Agregación temporal para modelos SARIMA
        SERIE_MENSUAL_TMAX = df.set_index('Fecha')['Tmax'].resample('ME').mean().fillna(method='ffill')
        SERIE_MENSUAL_TMIN = df.set_index('Fecha')['Tmin'].resample('ME').mean().fillna(method='ffill')
        
        # Tesis Ledezma Velázquez: Series de Valor Extremo (Máximos Anuales) para Gumbel
        SERIE_ANUAL_MAX_TMAX = df.set_index('Fecha')['Tmax'].resample('YE').max().dropna()
        SERIE_ANUAL_MIN_TMIN = df.set_index('Fecha')['Tmin'].resample('YE').min().dropna()

        df_temp = df.copy()
        df_temp = df_temp[df_temp['Fecha'].dt.year >= 2000]
        df_temp.set_index('Fecha', inplace=True)
        df_temp.interpolate(method='time', inplace=True)
        if df_temp.isnull().values.any():
            df_temp.fillna(method='ffill', inplace=True)
            df_temp.fillna(method='bfill', inplace=True)
            
        df_anual_tmax = df_temp['Tmax'].resample('YE').mean().dropna()
        df_anual_tmin = df_temp['Tmin'].resample('YE').mean().dropna()
        
        if len(df_anual_tmax) < 2:
             slope_tmax, intercept_tmax = 0, df_anual_tmax.mean()
             slope_tmin, intercept_tmin = 0, df_anual_tmin.mean()
             mae_tmax, mae_tmin = 0, 0
        else:
            anios_x_tmax = df_anual_tmax.index.year
            slope_tmax, intercept_tmax = np.polyfit(anios_x_tmax, df_anual_tmax, 1)
            anios_x_tmin = df_anual_tmin.index.year
            slope_tmin, intercept_tmin = np.polyfit(anios_x_tmin, df_anual_tmin, 1)
            residuos_tmax = df_anual_tmax - (anios_x_tmax * slope_tmax + intercept_tmax)
            mae_tmax = np.mean(np.abs(residuos_tmax))
            residuos_tmin = df_anual_tmin - (anios_x_tmin * slope_tmin + intercept_tmin)
            mae_tmin = np.mean(np.abs(residuos_tmin))
            
        df_temp['Year'] = df_temp.index.year
        df_temp['Tmax_trend'] = df_temp['Year'] * slope_tmax + intercept_tmax
        df_temp['Tmin_trend'] = df_temp['Year'] * slope_tmin + intercept_tmin
        df_temp['Seasonality_Max'] = df_temp['Tmax'] - df_temp['Tmax_trend']
        df_temp['Seasonality_Min'] = df_temp['Tmin'] - df_temp['Tmin_trend']
        dia_del_anio = df_temp.index.dayofyear
        patron_tmax = df_temp.groupby(dia_del_anio)['Seasonality_Max'].mean()
        patron_tmin = df_temp.groupby(dia_del_anio)['Seasonality_Min'].mean()
        full_index = pd.RangeIndex(start=1, stop=367)
        patron_tmax = patron_tmax.reindex(full_index).interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
        patron_tmin = patron_tmin.reindex(full_index).interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
        return patron_tmax, patron_tmin, slope_tmax, slope_tmin, intercept_tmax, intercept_tmin, mae_tmax, mae_tmin
    except Exception as e:
        print(f"Error: {e}")
        return None, None, 0, 0, 0, 0, 0, 0

# ==============================================================================
# MÉTODO CIENTÍFICO 1: MODELO SARIMA (Tesis Miranda Chinlli)
# ==============================================================================
def predecir_con_sarima(serie_historica, anio_objetivo):
    if not STATSMODELS_DISPONIBLE or serie_historica is None:
        return None
    
    # 1. DEFINICIÓN DEL MODELO ESTOCÁSTICO
    # Ecuación General SARIMA (p,d,q)x(P,D,Q)s - Tesis Miranda, Pág 30, Ec. 1.53
    # Formula: phi(B)Phi(B^s)(1-B)^d(1-B^s)^D Y_t = theta(B)Theta(B^s)epsilon_t
    # Configuración: SARIMA(1,0,1)(0,1,1)12 (Basado en Conclusión Tesis, Pág 56)

    modelo = SARIMAX(serie_historica,
                     order=(1, 0, 1),
                     seasonal_order=(0, 1, 1, 12),
                     enforce_stationarity=False,
                     enforce_invertibility=False)
    
    # 2. ESTIMACIÓN DE PARÁMETROS (MÁXIMA VEROSIMILITUD)
    # Método: Maximum Likelihood Estimation (MLE) - Tesis Miranda, Pág 35, Ec 2.7
    # Maximización de la función L(phi, mu, sigma) para encontrar coeficientes óptimos.

    modelo_ajustado = modelo.fit(disp=False)
    
    ultimo_fecha = serie_historica.index[-1]
    fecha_fin_pred = pd.Timestamp(f"{anio_objetivo}-12-31")
    if fecha_fin_pred <= ultimo_fecha:
        return None
    
    steps = (fecha_fin_pred.year - ultimo_fecha.year) * 12 + (fecha_fin_pred.month - ultimo_fecha.month)
    
    # 3. PRONÓSTICO RECURSIVO
    # Cálculo de esperanza matemática condicional - Tesis Miranda, Pág 40, Ec 2.16

    prediccion = modelo_ajustado.get_forecast(steps=steps)
    pred_media = prediccion.predicted_mean
    
    pred_anio = pred_media[pred_media.index.year == anio_objetivo]
    return pred_anio

# ==============================================================================
# MÉTODO CIENTÍFICO 2: ANÁLISIS DE RIESGO GUMBEL (Tesis Ledezma Velázquez)
# ==============================================================================
def calcular_riesgo_gumbel(serie_anual_extrema, tipo="max"):
    if serie_anual_extrema is None or len(serie_anual_extrema) < 2:
        return "Datos insuficientes."
        
    # 1. PARÁMETROS ESTADÍSTICOS DE LA MUESTRA
    # Tesis Ledezma, Pág 36 (Ec 13) y Pág 38 (Ec 16)
  
    mu = serie_anual_extrema.mean()
    sigma = serie_anual_extrema.std()
    
    resultados = []
    periodos_retorno = [2, 5, 10, 50]
    
    for Tr in periodos_retorno:
      
        # 2. FACTOR DE FRECUENCIA (Kt) PARA DISTRIBUCIÓN GUMBEL
        # Tesis Ledezma, Pág 47, Ecuación 32
        # Formula: Kt = -sqrt(6)/pi * [0.5772 + ln(ln(T/(T-1)))]
     
        kt = -(np.sqrt(6)/np.pi) * (0.5772 + np.log(np.log(Tr / (Tr - 1))))
        
        # 3. CÁLCULO DE LA MAGNITUD DEL EVENTO EXTREMO (Xt)
        # Tesis Ledezma, Pág 45, Ecuación 28 (Método de Factores de Frecuencia)
        # Formula: Xt = media + Kt * desviacion_estandar

        val = mu + kt * sigma if tipo == "max" else mu - kt * sigma
        resultados.append((Tr, val))
        
    return resultados

# --- CÁLCULO PRINCIPAL ---
def calcular_prediccion_para_anio(anio, patron_tmax, patron_tmin, slope_tmax, slope_tmin, intercept_tmax, intercept_tmin):
    global SERIE_MENSUAL_TMAX, SERIE_MENSUAL_TMIN
    
    # Cálculo Estocástico usando SARIMA (Metodología Tesis Miranda)
    pred_mensual_max = predecir_con_sarima(SERIE_MENSUAL_TMAX, anio)
    pred_mensual_min = predecir_con_sarima(SERIE_MENSUAL_TMIN, anio)
    
    metodo_usado = "Estocástico (SARIMA)"
    
    if pred_mensual_max is None or pred_mensual_min is None or len(pred_mensual_max) == 0:
        # Fallback a regresión lineal clásica si SARIMA no converge o no hay datos suficientes
        metodo_usado = "Regresión Lineal"
        trend_val_max = (slope_tmax * anio) + intercept_tmax
        trend_val_min = (slope_tmin * anio) + intercept_tmin
        es_bisiesto = (anio % 4 == 0 and anio % 100 != 0) or (anio % 400 == 0)
        fechas_anio = pd.date_range(start=f'{anio}-01-01', end=f'{anio}-12-31', freq='D')
        dias_anio = fechas_anio.dayofyear
        df_pred = pd.DataFrame(index=fechas_anio)
        dias_a_mapear = dias_anio.map(lambda x: 366 if x > 366 and es_bisiesto else (365 if x >= 366 else x))
        vals_tmax = patron_tmax.loc[dias_a_mapear].values
        vals_tmin = patron_tmin.loc[dias_a_mapear].values
        df_pred['Tmax_pred'] = trend_val_max + vals_tmax
        df_pred['Tmin_pred'] = trend_val_min + vals_tmin
        return df_pred, metodo_usado
        
    # Desagregación Temporal usando Splines Cúbicos
    # Convierte el pronóstico mensual de SARIMA en una curva diaria suave.
    fechas_mensuales = pred_mensual_max.index
    dias_x = np.array([d.dayofyear for d in fechas_mensuales])
    x_points = np.concatenate(([dias_x[-1]-365], dias_x, [dias_x[0]+365]))
    y_points_max = np.concatenate(([pred_mensual_max.values[-1]], pred_mensual_max.values, [pred_mensual_max.values[0]]))
    y_points_min = np.concatenate(([pred_mensual_min.values[-1]], pred_mensual_min.values, [pred_mensual_min.values[0]]))
    spline_max = CubicSpline(x_points, y_points_max)
    spline_min = CubicSpline(x_points, y_points_min)
    fechas_anio = pd.date_range(start=f'{anio}-01-01', end=f'{anio}-12-31', freq='D')
    dias_objetivo = np.arange(1, len(fechas_anio) + 1)
    tmax_diaria = spline_max(dias_objetivo)
    tmin_diaria = spline_min(dias_objetivo)
    df_pred = pd.DataFrame(index=fechas_anio)
    df_pred['Tmax_pred'] = tmax_diaria
    df_pred['Tmin_pred'] = tmin_diaria
    return df_pred, metodo_usado

def crear_grafica(df_pred, titulo):
    fig = plt.Figure(figsize=(12, 6), dpi=100)
    fig.patch.set_facecolor(BG_COLOR)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG_GRAFICA)
    ax.plot(df_pred.index, df_pred['Tmax_pred'], label='Temp. Máxima', color=TEXT_COLOR_RED, alpha=0.8)
    ax.plot(df_pred.index, df_pred['Tmin_pred'], label='Temp. Mínima', color=TEXT_COLOR_BLUE, alpha=0.8)
    ax.fill_between(df_pred.index, df_pred['Tmax_pred'], df_pred['Tmin_pred'], color='gray', alpha=0.2)
    ax.set_title(titulo, fontsize=16, color=TEXT_COLOR_BLUE)
    ax.set_ylabel('Temperatura (°C)', fontsize=12, color=TEXT_COLOR_GREEN)
    ax.set_xlabel('Fecha', fontsize=12, color=TEXT_COLOR_GREEN)
    if len(df_pred) > 31:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate(rotation=45, ha='right')
    min_temp = df_pred['Tmin_pred'].min()
    max_temp = df_pred['Tmax_pred'].max()
    y_ticks_min = np.floor(min_temp / 2) * 2 - 2
    y_ticks_max = np.ceil(max_temp / 2) * 2 + 2
    y_ticks = np.arange(y_ticks_min, y_ticks_max, 2)
    ax.set_yticks(y_ticks)
    ax.set_ylim(y_ticks_min, y_ticks_max)
    ax.tick_params(axis='x', colors=TEXT_COLOR_GREEN)
    ax.tick_params(axis='y', colors=TEXT_COLOR_GREEN)
    ax.spines['top'].set_color(TEXT_COLOR_GREEN)
    ax.spines['bottom'].set_color(TEXT_COLOR_GREEN)
    ax.spines['left'].set_color(TEXT_COLOR_GREEN)
    ax.spines['right'].set_color(TEXT_COLOR_GREEN)
    legend = ax.legend()
    legend.get_frame().set_facecolor(BG_COLOR)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR_GREEN)
    ax.grid(True, linestyle='--', alpha=0.3, color=TEXT_COLOR_GREEN)
    fig.tight_layout()
    return fig

def mostrar_emoji_dia(frame_grafica, tmax, tmin):
    global emoji_photo_dict
    for widget in frame_grafica.winfo_children():
        widget.destroy()
    promedio = (tmax + tmin) / 2
    ruta_imagen = ""
    texto_leyenda = ""
    color_leyenda = TEXT_COLOR_GREEN
    if promedio < 15:
        ruta_imagen = RUTA_EMOJI_FRIO
        texto_leyenda = "EL DIA SERA FRIO"
        color_leyenda = TEXT_COLOR_BLUE
    elif 15 <= promedio <= 25:
        ruta_imagen = RUTA_EMOJI_MEDIA
        texto_leyenda = "EL DIA SERA TEMPLADO"
        color_leyenda = TEXT_COLOR_GREEN
    elif promedio >= 26:
        ruta_imagen = RUTA_EMOJI_CALIENTE
        texto_leyenda = "EL DIA SERA CALIDO"
        color_leyenda = TEXT_COLOR_RED
    if not PIL_DISPONIBLE or not os.path.exists(ruta_imagen):
        fallback_label = tk.Label(frame_grafica, text=texto_leyenda, font=TITLE_FONT, fg=color_leyenda, bg=BG_GRAFICA)
        fallback_label.pack(expand=True)
        return
    try:
        center_frame = ttk.Frame(frame_grafica, style="Dark.TFrame")
        center_frame.pack(expand=True)
        frame_grafica.update_idletasks()
        max_height = int(frame_grafica.winfo_height() * 0.6)
        if max_height < 100: max_height = 200
        img = Image.open(ruta_imagen)
        ratio = img.width / img.height
        new_height = max_height
        new_width = int(new_height * ratio)
        if new_width > frame_grafica.winfo_width() * 0.9:
            new_width = int(frame_grafica.winfo_width() * 0.9)
            new_height = int(new_width / ratio)
        img = img.resize((new_width, new_height), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        emoji_photo_dict['current_emoji'] = img_tk
        label_img = tk.Label(center_frame, image=img_tk, bg=BG_GRAFICA)
        label_img.image = img_tk
        label_img.pack(pady=(20,0))
        label_txt = tk.Label(center_frame, text=texto_leyenda, font=EMOJI_TEXT_FONT, fg=color_leyenda, bg=BG_GRAFICA)
        label_txt.pack(pady=(10,20))
    except:
        fallback_label = tk.Label(frame_grafica, text=texto_leyenda, font=TITLE_FONT, fg=color_leyenda, bg=BG_GRAFICA)
        fallback_label.pack(expand=True)

def cargar_imagen_fondo_label(target_widget, ruta_imagen):
    global background_photo_dict
    widget_id = str(target_widget)
    if not PIL_DISPONIBLE or not os.path.exists(ruta_imagen):
        return None
    try:
        target_widget.update_idletasks()
        w = target_widget.winfo_width()
        h = target_widget.winfo_height()
        if w <= 1 or h <= 1:
            w = target_widget.winfo_screenwidth()
            h = target_widget.winfo_screenheight()
        image = Image.open(ruta_imagen).resize((w, h), Image.LANCZOS)
        background_photo_dict[widget_id] = ImageTk.PhotoImage(image)
        label_key = f"bg_label_{widget_id}"
        if label_key in background_photo_dict:
            background_photo_dict[label_key].config(image=background_photo_dict[widget_id])
        else:
            background_label = tk.Label(target_widget, image=background_photo_dict[widget_id], bd=0)
            background_label.place(x=0, y=0, relwidth=1, relheight=1)
            background_label.lower()
            background_photo_dict[label_key] = background_label
        def on_resize(event, widget_key=widget_id, label_key=label_key):
            try:
                new_w, new_h = event.width, event.height
                if new_w <= 1 or new_h <= 1: return
                img = Image.open(ruta_imagen).resize((new_w, new_h), Image.LANCZOS)
                resized_key = f"{widget_key}_resized"
                background_photo_dict[resized_key] = ImageTk.PhotoImage(img)
                if label_key in background_photo_dict:
                    background_photo_dict[label_key].config(image=background_photo_dict[resized_key])
            except:
                pass
        target_widget.bind('<Configure>', on_resize, add='+')
        return background_photo_dict[label_key]
    except:
        return None

def abrir_pantalla_predictor():
    global background_photo_dict, emoji_photo_dict, patron_tmax_g, SERIE_ANUAL_MAX_TMAX
    if patron_tmax_g is None:
        messagebox.showerror("Error", "Debes CARGAR UNA BASE de datos antes de iniciar.")
        return
    riesgos_max = calcular_riesgo_gumbel(SERIE_ANUAL_MAX_TMAX, "max")
    riesgos_min = calcular_riesgo_gumbel(SERIE_ANUAL_MIN_TMIN, "min")
    global RIESGO_GUMBEL_TEXTO
    if isinstance(riesgos_max, list):
        RIESGO_GUMBEL_TEXTO = "ANÁLISIS DE FRECUENCIA DE VALORES EXTREMOS (GUMBEL):\n\n"
        RIESGO_GUMBEL_TEXTO += "Probabilidad de Eventos (Factor Kt):\n"
        RIESGO_GUMBEL_TEXTO += f"Retorno 2 años : Max {riesgos_max[0][1]:.1f}°C | Min {riesgos_min[0][1]:.1f}°C\n"
        RIESGO_GUMBEL_TEXTO += f"Retorno 5 años : Max {riesgos_max[1][1]:.1f}°C | Min {riesgos_min[1][1]:.1f}°C\n"
        RIESGO_GUMBEL_TEXTO += f"Retorno 10 años : Max {riesgos_max[2][1]:.1f}°C | Min {riesgos_min[2][1]:.1f}°C\n"
        RIESGO_GUMBEL_TEXTO += f"Retorno 50 años : Max {riesgos_max[3][1]:.1f}°C | Min {riesgos_min[3][1]:.1f}°C\n"
    else:
        RIESGO_GUMBEL_TEXTO = "Datos insuficientes para análisis extremo."
    root.withdraw()
    predictor_window = Toplevel(root)
    predictor_window.title("Pantalla de Predicción")
    w = predictor_window.winfo_screenwidth()
    h = predictor_window.winfo_screenheight()
    predictor_window.geometry(f"{w}x{h}+0+0")
    predictor_window.update_idletasks()
    
    current_graph_frame = None
    current_data_frame = None
    current_calc_frame = None
    current_canvas = None
    current_data_widget = None
    current_calc_widget = None
    current_mae_label = None
    
    meses_map = {"Seleccionar": 0, "Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4, "Mayo": 5, "Junio": 6, "Julio": 7, "Agosto": 8, "Septiembre": 9, "Octubre": 10, "Noviembre": 11, "Diciembre": 12}
    
    def cerrar_y_mostrar_root():
        predictor_window.destroy()
        root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")
        root.state('zoomed')
        root.update_idletasks()
        root.deiconify()
    predictor_window.protocol("WM_DELETE_WINDOW", salir)
    main_frame = ttk.Frame(predictor_window, padding="10", style="TFrame")
    main_frame.pack(expand=True, fill="both")
    top_frame = ttk.Frame(main_frame, padding="10", style="TFrame")
    top_frame.pack(fill='x')
    content_frame = ttk.Frame(main_frame, padding="10", style="TFrame")
    content_frame.pack(expand=True, fill="both")

    def poblar_datos_texto(data_text_widget, df_para_mostrar):
        data_text_widget.config(state='normal')
        data_text_widget.delete('1.0', 'end')
        header_fecha = "FECHA".ljust(12)
        header_tmax = " TEMP. MAX".ljust(18)
        header_tmin = "TEMP. MIN".ljust(20)
        data_text_widget.insert('end', header_fecha, ('header'))
        data_text_widget.insert('end', header_tmax, ('tmax_header'))
        data_text_widget.insert('end', header_tmin + "\n", ('tmin_header'))
        data_text_widget.insert('end', "=" * (12+18+20) + "\n", ('header'))
        for index, row in df_para_mostrar.iterrows():
            fecha_str = index.strftime('%Y-%m-%d').ljust(12)
            tmax_str = (" " + f"{row['Tmax_pred']:.2f}°C").ljust(18)
            tmin_str = (" " + f"{row['Tmin_pred']:.2f}°C").ljust(20) + "\n"
            data_text_widget.insert('end', fecha_str, ('header'))
            data_text_widget.insert('end', tmax_str, ('tmax_data'))
            data_text_widget.insert('end', tmin_str, ('tmin_data'))
        data_text_widget.config(state='disabled')

    def poblar_calculo_texto(calc_text_widget, anio, mes_str, dia_str, modo="diario"):
        calc_text_widget.config(state='normal')
        calc_text_widget.delete('1.0', 'end')
        calc_text_widget.tag_configure('header', font=CALC_HEADER_FONT, foreground=TEXT_COLOR_BLUE)
        calc_text_widget.tag_configure('math', font=CALC_FONT, foreground=TEXT_COLOR_WHITE)
        calc_text_widget.tag_configure('variable', font=CALC_FONT, foreground=TEXT_COLOR_GREEN)
        
        if modo == "diario":
            calc_text_widget.insert('end', "MODELO MATEMÁTICO: SARIMA ESTOCÁSTICO\n", 'header')
            texto_sarima = f"""
CONFIGURACIÓN: SARIMA(1,0,1)(0,1,1)12
1. ECUACIÓN DEL MODELO:
   phi(B)Phi(B^s)(1-B)^d(1-B^s)^D Y_t = theta(B)Theta(B^s)epsilon_t
2. ESTIMACIÓN DE PARÁMETROS:
   Método: Máxima Verosimilitud (MLE)
   L(theta) -> Max
3. ECUACIÓN DE PRONÓSTICO:
   E[X_{{t+h}} | X_t] (Esperanza Condicional Recursiva)
"""
            calc_text_widget.insert('end', texto_sarima, 'math')
            calc_text_widget.insert('end', "-"*40 + "\n", 'math')
            calc_text_widget.insert('end', RIESGO_GUMBEL_TEXTO, 'variable')
            texto_gumbel_teoria = """
    
    Fórmulas de Riesgo (Gumbel):
    P(x <= xi) = e^(-e^(-yi))
    Xt = mu + Kt * sigma
    Kt = -sqrt(6)/pi * [0.5772 + ln(ln(T/(T-1)))]
"""
            calc_text_widget.insert('end', texto_gumbel_teoria, 'math')
        else:
            calc_text_widget.insert('end', "CALCULOS DE PROYECCIÓN\n", 'header')
            texto_math = f"""
MÉTODO: REGRESIÓN POLINOMIAL + TENDENCIA
1. ECUACIÓN DEL POLINOMIO (Curva Estacional):
   P(x) = c5*x^5 + c4*x^4 + ... + c0
   Donde 'x' es el mes (1-12).
   COEFICIENTES CALCULADOS:
{ECUACION_DISPLAY_INFO}
2. FACTOR DE TENDENCIA SECULAR:
   m_promedio = {SLOPE_MENSUAL_DISPLAY:.5f}
   Ajuste = {anio - ANO_BASE_DISPLAY} * {SLOPE_MENSUAL_DISPLAY:.5f}
3. CÁLCULO FINAL:
   T_pred = P(mes) + Ajuste
"""
            calc_text_widget.insert('end', texto_math, 'math')
        calc_text_widget.config(state='disabled')

    def limpiar_paneles():
        nonlocal current_graph_frame, current_data_frame, current_calc_frame, current_canvas, current_data_widget, current_calc_widget, current_mae_label
        if current_graph_frame: current_graph_frame.destroy(); current_graph_frame = None
        if current_data_frame: current_data_frame.destroy(); current_data_frame = None
        if current_calc_frame: current_calc_frame.destroy(); current_calc_frame = None
        if current_mae_label: current_mae_label.destroy(); current_mae_label = None
        if current_canvas: current_canvas.get_tk_widget().destroy(); current_canvas = None
        current_data_widget = None
        current_calc_widget = None

    def actualizar_estado_boton_mensual():
        ano_sel = combo_ano.get()
        mes_sel = combo_mes.get()
        dia_sel = combo_dia.get()
        if ano_sel != "Seleccionar" and mes_sel == "Seleccionar" and dia_sel == "Seleccionar":
            btn_pred_anual.pack(side='left', padx=5)
        else:
            btn_pred_anual.pack_forget()

    def actualizar_dias_combobox(event=None):
        try:
            limpiar_paneles()
            actualizar_estado_boton_mensual()
            ano_sel = combo_ano.get()
            mes_sel = combo_mes.get()
            if ano_sel == "Seleccionar" or mes_sel == "Seleccionar":
                dias_en_mes = 31
            else:
                ano_num = int(ano_sel)
                mes_num = meses_map[mes_sel]
                es_bisiesto = (ano_num % 4 == 0 and ano_num % 100 != 0) or (ano_num % 400 == 0)
                dias_en_mes = 29 if (mes_num == 2 and es_bisiesto) else (28 if mes_num == 2 else (30 if mes_num in [4, 6, 9, 11] else 31))
            lista_dias = ["Seleccionar"] + list(range(1, dias_en_mes + 1))
            combo_dia['values'] = lista_dias
            combo_dia.set("Seleccionar")
        except:
            combo_dia['values'] = ["Seleccionar"] + list(range(1, 32))
            combo_dia.set("Seleccionar")
            
    def on_ano_selected_wrapper(event):
        combo_mes.set("Seleccionar"); combo_dia.set("Seleccionar")
        limpiar_paneles(); actualizar_estado_boton_mensual(); actualizar_dias_combobox(event)
        
    def on_mes_selected_wrapper(event):
        combo_dia.set("Seleccionar")
        limpiar_paneles(); actualizar_estado_boton_mensual(); actualizar_dias_combobox(event)
        
    def on_dia_selected_wrapper(event):
        limpiar_paneles(); actualizar_estado_boton_mensual()

    def logica_prediccion():
        nonlocal current_graph_frame, current_data_frame, current_calc_frame, current_canvas, current_data_widget, current_calc_widget, current_mae_label
        limpiar_paneles()
        content_frame.columnconfigure(0, weight=6)
        content_frame.columnconfigure(1, weight=4)
        content_frame.rowconfigure(0, weight=1)
        content_frame.rowconfigure(1, weight=1)
        current_graph_frame = ttk.Frame(content_frame, padding="10", relief="sunken", borderwidth=1, style="Dark.TFrame")
        current_graph_frame.grid(row=0, column=0, rowspan=2, sticky='nsew', padx=5, pady=5)
        current_data_frame = ttk.Frame(content_frame, padding="10", relief="sunken", borderwidth=1, style="Dark.TFrame")
        current_data_frame.grid(row=0, column=1, sticky='nsew', padx=5, pady=(5,2))
        current_calc_frame = ttk.Frame(content_frame, padding="10", relief="sunken", borderwidth=1, style="Dark.TFrame")
        current_calc_frame.grid(row=1, column=1, sticky='nsew', padx=5, pady=(3,5))
        content_frame.rowconfigure(2, weight=0)
        current_mae_label = ttk.Label(content_frame, text="", style="Mae.TLabel")
        current_mae_label.grid(row=2, column=0, sticky='n', pady=5, padx=5)
        current_data_widget = Text(current_data_frame, wrap='none', font=DATA_FONT, bg=BG_COLOR, fg=TEXT_COLOR_GREEN, relief="flat", insertbackground=TEXT_COLOR_GREEN, borderwidth=0)
        scroll_y_data = ttk.Scrollbar(current_data_frame, orient='vertical', command=current_data_widget.yview, style="Vertical.TScrollbar")
        current_data_widget.config(yscrollcommand=scroll_y_data.set)
        scroll_y_data.pack(side='right', fill='y')
        current_data_widget.pack(expand=True, fill='both')
        current_calc_widget = Text(current_calc_frame, wrap='word', font=CALC_FONT, bg=BG_COLOR, fg=TEXT_COLOR_GREEN, relief="flat", insertbackground=TEXT_COLOR_GREEN, borderwidth=0)
        scroll_y_calc = ttk.Scrollbar(current_calc_frame, orient='vertical', command=current_calc_widget.yview, style="Vertical.TScrollbar")
        current_calc_widget.config(yscrollcommand=scroll_y_calc.set)
        scroll_y_calc.pack(side='right', fill='y')
        current_calc_widget.pack(expand=True, fill='both')
        current_data_widget.tag_configure('header', font=DEFAULT_FONT_BOLD, foreground=TEXT_COLOR_GREEN)
        current_data_widget.tag_configure('tmax_header', foreground=TEXT_COLOR_RED, font=DEFAULT_FONT_BOLD)
        current_data_widget.tag_configure('tmin_header', foreground=TEXT_COLOR_BLUE, font=DEFAULT_FONT_BOLD)
        current_data_widget.tag_configure('tmax_data', foreground=TEXT_COLOR_RED)
        current_data_widget.tag_configure('tmin_data', foreground=TEXT_COLOR_BLUE)
        current_data_widget.config(state='disabled')
        current_calc_widget.config(state='disabled')
        ano_str = combo_ano.get()
        mes_str = combo_mes.get()
        dia_str = combo_dia.get()
        if ano_str == "Seleccionar":
            messagebox.showerror("Error", "Por favor, selecciona un año.", parent=predictor_window)
            limpiar_paneles()
            return
        ano_num = int(ano_str)
        if ano_num == 2060:
            messagebox.showinfo("Fin del Mundo", "El mundo acabo el 31 de diciembre de 2059", parent=predictor_window)
            limpiar_paneles()
            return
        df_pred, metodo_usado = calcular_prediccion_para_anio(ano_num, patron_tmax_g, patron_tmin_g, slope_tmax_g, slope_tmin_g, intercept_tmax_g, intercept_tmin_g)
        mae_texto = f"Método: {metodo_usado} | (SARIMA + Gumbel)"
        current_mae_label.config(text=mae_texto)
        if mes_str == "Seleccionar" and dia_str == "Seleccionar":
            titulo = f"Predicción Anual {ano_num}"
            fig = crear_grafica(df_pred, titulo)
            current_canvas = FigureCanvasTkAgg(fig, master=current_graph_frame)
            current_canvas.draw()
            current_canvas.get_tk_widget().pack(side='top', fill='both', expand=True)
            poblar_datos_texto(current_data_widget, df_pred)
            poblar_calculo_texto(current_calc_widget, ano_num, mes_str, dia_str, modo="diario")
        elif mes_str != "Seleccionar" and dia_str == "Seleccionar":
            mes_num = meses_map[mes_str]
            df_pred_mes = df_pred[df_pred.index.month == mes_num]
            titulo = f"Predicción de Mes {mes_str}/{ano_num}"
            fig = crear_grafica(df_pred_mes, titulo)
            current_canvas = FigureCanvasTkAgg(fig, master=current_graph_frame)
            current_canvas.draw()
            current_canvas.get_tk_widget().pack(side='top', fill='both', expand=True)
            poblar_datos_texto(current_data_widget, df_pred_mes)
            poblar_calculo_texto(current_calc_widget, ano_num, mes_str, dia_str, modo="diario")
        elif mes_str != "Seleccionar" and dia_str != "Seleccionar":
            mes_num = meses_map[mes_str]
            dia_num = int(dia_str)
            try:
                fecha_str = f"{ano_num}-{mes_num:02d}-{dia_num:02d}"
                df_pred_dia = df_pred.loc[fecha_str:fecha_str]
                if not df_pred_dia.empty:
                    tmax_dia = df_pred_dia.iloc[0]['Tmax_pred']
                    tmin_dia = df_pred_dia.iloc[0]['Tmin_pred']
                    mostrar_emoji_dia(current_graph_frame, tmax_dia, tmin_dia)
                poblar_datos_texto(current_data_widget, df_pred_dia)
                poblar_calculo_texto(current_calc_widget, ano_num, mes_str, dia_str, modo="diario")
            except:
                messagebox.showerror("Fecha Inválida", f"La fecha {dia_num}/{mes_str}/{ano_num} no existe.", parent=predictor_window)
                limpiar_paneles()
        else:
            messagebox.showerror("Selección Inválida", "Para predecir por día, debes seleccionar un mes.", parent=predictor_window)
            limpiar_paneles()

    def calcular_metodo_cansa(anio):
        global DF_GLOBAL, ECUACION_DISPLAY_INFO, SLOPE_MENSUAL_DISPLAY, ANO_BASE_DISPLAY
        meses_x = np.array(range(1, 13))
        
        # Lógica Universal de Regresión Polinomial
        ANO_BASE_DISPLAY = 2000
        df_calc = DF_GLOBAL.copy()
        df_calc['Tavg'] = (df_calc['Tmax'] + df_calc['Tmin']) / 2
        df_calc['Mes'] = df_calc['Fecha'].dt.month
        datos_entrenamiento = df_calc.groupby('Mes')['Tavg'].mean().values
        if len(datos_entrenamiento) != 12:
            datos_entrenamiento = np.resize(datos_entrenamiento, 12)
        coefs = np.polyfit(meses_x, datos_entrenamiento, 5)
        SLOPE_MENSUAL_DISPLAY = (slope_tmax_g + slope_tmin_g) / 2
        avg_slope = SLOPE_MENSUAL_DISPLAY
        delta_anios = anio - 2000
        ajuste_tendencia = delta_anios * avg_slope
        polinomio = np.poly1d(coefs)
        
        def funcion_proyeccion_universal(m):
            val_poly = polinomio(m)
            return val_poly + ajuste_tendencia
        
        ecuacion_cansa = funcion_proyeccion_universal
        
        ECUACION_DISPLAY_INFO = f""" c5 = {coefs[0]:.6f}
   c4 = {coefs[1]:.6f}
   c3 = {coefs[2]:.6f}
   c2 = {coefs[3]:.6f}
   c1 = {coefs[4]:.6f}
   c0 = {coefs[5]:.6f}"""
        
        resultados_meses = []
        meses_txt = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        for i, mes_nombre in enumerate(meses_txt):
            mes_numero = i + 1
            valor_predicho = ecuacion_cansa(mes_numero)
            if abs(valor_predicho) < 1e-5: valor_predicho = 0.0
            resultados_meses.append((mes_nombre, valor_predicho))
        return resultados_meses

    def logica_prediccion_anual():
        ano_str = combo_ano.get()
        if ano_str == "Seleccionar":
            messagebox.showerror("Error", "Por favor, selecciona un año.", parent=predictor_window)
            return
        ano_num = int(ano_str)
        datos_mensuales = calcular_metodo_cansa(ano_num)
        mostrar_tabla_mensual(ano_num, datos_mensuales)

    def mostrar_tabla_mensual(anio, datos_mensuales):
        nonlocal current_graph_frame, current_data_frame, current_calc_frame, current_data_widget, current_calc_widget, current_canvas
        limpiar_paneles()
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)
        content_frame.rowconfigure(0, weight=1)
        current_data_frame = ttk.Frame(content_frame, padding="10", relief="sunken", borderwidth=1, style="Dark.TFrame")
        current_data_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        current_calc_frame = ttk.Frame(content_frame, padding="10", relief="sunken", borderwidth=1, style="Dark.TFrame")
        current_calc_frame.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)
        current_data_widget = Text(current_data_frame, wrap='none', font=DATA_FONT, bg=BG_COLOR, fg=TEXT_COLOR_GREEN, relief="flat", insertbackground=TEXT_COLOR_GREEN, borderwidth=0)
        scroll_y_data = ttk.Scrollbar(current_data_frame, orient='vertical', command=current_data_widget.yview, style="Vertical.TScrollbar")
        current_data_widget.config(yscrollcommand=scroll_y_data.set)
        scroll_y_data.pack(side='right', fill='y')
        current_data_widget.pack(expand=True, fill='both')
        current_calc_widget = Text(current_calc_frame, wrap='word', font=CALC_FONT, bg=BG_COLOR, fg=TEXT_COLOR_GREEN, relief="flat", insertbackground=TEXT_COLOR_GREEN, borderwidth=0)
        current_calc_widget.pack(expand=True, fill='both', pady=(0, 10))
        btn_back = ttk.Button(current_calc_frame, text="Regresar a Gráfica", command=logica_prediccion, style="TButton")
        btn_back.pack(side='bottom', pady=10)
        current_data_widget.tag_configure('header', font=DEFAULT_FONT_BOLD, foreground=TEXT_COLOR_GREEN)
        current_data_widget.tag_configure('col_header', font=DEFAULT_FONT_BOLD, foreground=TEXT_COLOR_BLUE)
        current_data_widget.tag_configure('row_data', foreground=TEXT_COLOR_WHITE)
        current_data_widget.insert('end', f"RESULTADOS MÉTODO CANSA: {NOMBRE_BASE_DISPLAY}\n", 'header')
        current_data_widget.insert('end', "-"*40 + "\n", 'header')
        header = f"{'MES AÑO'.ljust(20)}\t{'VALOR'}\n"
        current_data_widget.insert('end', header, 'col_header')
        current_data_widget.insert('end', "-"*40 + "\n", 'header')
        for mes_nombre, valor in datos_mensuales:
            col_fecha = f"{mes_nombre} {anio}"
            row_str = f"{col_fecha.ljust(20)}\t{valor:.7f}\n"
            current_data_widget.insert('end', row_str, 'row_data')
        current_data_widget.config(state='disabled')
        poblar_calculo_texto(current_calc_widget, anio, "", "", modo="mensual")

    titulo_text = f"PREDICCIONES DE LAS TEMPERATURAS MINIMAS Y MAXIMAS DEL DIA EN {NOMBRE_BASE_DISPLAY.upper()}"
    ttk.Label(top_frame, text=titulo_text, style="SubTitle.TLabel").pack(pady=10)
    controles_frame = ttk.Frame(top_frame, style="TFrame")
    controles_frame.pack()
    controles_frame.columnconfigure(0, weight=1)
    controles_frame.columnconfigure(1, weight=1)
    controles_frame.columnconfigure(2, weight=1)
    controles_frame.columnconfigure(3, weight=2)
    controles_frame.columnconfigure(4, weight=2)
    controles_frame.columnconfigure(5, weight=2)
    combo_frame = ttk.Frame(controles_frame, style="TFrame")
    combo_frame.grid(row=0, column=0, columnspan=3, sticky='w', padx=10)
    btn_pred_anual = ttk.Button(combo_frame, text="PROMEDIO MENSUAL", command=logica_prediccion_anual, style="TButton")
    btn_pred_anual.pack(side='left', padx=5)
    ttk.Label(combo_frame, text="Año:", style="TLabel").pack(side='left', padx=(10, 2))
    lista_anos = ["Seleccionar"] + list(range(2020, 2061))
    combo_ano = ttk.Combobox(combo_frame, values=lista_anos, state="readonly", width=12, font=DEFAULT_FONT)
    combo_ano.pack(side='left', padx=5)
    combo_ano.set("Seleccionar")
    ttk.Label(combo_frame, text="Mes:", style="TLabel").pack(side='left', padx=(10, 2))
    lista_meses = ["Seleccionar"] + list(meses_map.keys())[1:]
    combo_mes = ttk.Combobox(combo_frame, values=lista_meses, state="readonly", width=12, font=DEFAULT_FONT)
    combo_mes.pack(side='left', padx=5)
    combo_mes.set("Seleccionar")
    ttk.Label(combo_frame, text="Día:", style="TLabel").pack(side='left', padx=(10, 2))
    lista_dias = ["Seleccionar"] + list(range(1, 32))
    combo_dia = ttk.Combobox(combo_frame, values=lista_dias, state="readonly", width=12, font=DEFAULT_FONT)
    combo_dia.pack(side='left', padx=5)
    combo_dia.set("Seleccionar")
    combo_ano.bind("<<ComboboxSelected>>", on_ano_selected_wrapper)
    combo_mes.bind("<<ComboboxSelected>>", on_mes_selected_wrapper)
    combo_dia.bind("<<ComboboxSelected>>", on_dia_selected_wrapper)
    btn_predecir = ttk.Button(controles_frame, text="Predecir", command=logica_prediccion, style="Large.TButton")
    btn_predecir.grid(row=0, column=3, padx=20)
    btn_regresar = ttk.Button(controles_frame, text="Regresar al Inicio", command=cerrar_y_mostrar_root, style="TButton")
    btn_regresar.grid(row=0, column=4, padx=10)
    actualizar_estado_boton_mensual()
    predictor_window.state('zoomed')

def salir():
    if root:
        root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Predictor de Clima")
    w = root.winfo_screenwidth()
    h = root.winfo_screenheight()
    root.geometry(f"{w}x{h}+0+0")
    root_canvas = tk.Canvas(root, highlightthickness=0, bd=0)
    root_canvas.pack(fill="both", expand=True)
    BACKGROUND_LOADED = False
    try:
        if PIL_DISPONIBLE and os.path.exists(NOMBRE_IMAGEN):
            img = Image.open(NOMBRE_IMAGEN).resize((w, h), Image.LANCZOS)
            background_photo_dict['root'] = ImageTk.PhotoImage(img)
            root_canvas.create_image(0, 0, image=background_photo_dict['root'], anchor='nw', tags="bg")
            BACKGROUND_LOADED = True
        else:
            root_canvas.configure(bg=BG_COLOR)
    except:
        root_canvas.configure(bg=BG_COLOR)
    style = ttk.Style(root)
    style.theme_use('clam')
    style.configure('.', foreground=TEXT_COLOR_GREEN, font=DEFAULT_FONT, fieldbackground=BG_COLOR, borderwidth=0)
    style.configure('TFrame', background=BG_COLOR)
    style.configure('Dark.TFrame', background=BG_GRAFICA)
    style.configure('TLabel', background=BG_COLOR, foreground=TEXT_COLOR_GREEN, font=DEFAULT_FONT)
    style.configure('Title.TLabel', background=BG_COLOR, foreground=TEXT_COLOR_BLUE, font=TITLE_FONT)
    style.configure('SubTitle.TLabel', background=BG_COLOR, foreground=TEXT_COLOR_BLUE, font=SUB_TITLE_FONT)
    style.configure('Description.TLabel', background=BG_COLOR, foreground=TEXT_COLOR_BLUE, font=DESCRIPTION_FONT)
    style.configure('Mae.TLabel', background=BG_COLOR, foreground=TEXT_COLOR_GREEN, font=("Courier New", 14, "italic"))
    style.configure('TButton', background=SELECT_BG, foreground=TEXT_COLOR_GREEN, font=DEFAULT_FONT, borderwidth=1, focusthickness=0, focuscolor='none')
    style.map('TButton', background=[('active', '#555555')], foreground=[('active', TEXT_COLOR_BLUE)])
    style.configure('Large.TButton', font=("Courier New", 22, "bold"), padding=20)
    style.configure('TCombobox', fieldbackground=SELECT_BG, background=SELECT_BG, foreground=TEXT_COLOR_GREEN, insertcolor=TEXT_COLOR_GREEN, arrowcolor=TEXT_COLOR_GREEN)
    style.map('TCombobox', fieldbackground=[('readonly', SELECT_BG)], background=[('readonly', SELECT_BG)], foreground=[('readonly', TEXT_COLOR_GREEN)], arrowcolor=[('readonly', TEXT_COLOR_GREEN)])
    root.option_add('*TCombobox*Listbox.font', LISTBOX_FONT)
    root.option_add('*TCombobox*Listbox.background', SELECT_BG)
    root.option_add('*TCombobox*Listbox.foreground', TEXT_COLOR_WHITE)
    style.configure('Vertical.TScrollbar', background=SELECT_BG, troughcolor=BG_COLOR, bordercolor=SELECT_BG, arrowcolor=TEXT_COLOR_GREEN)
    style.map('Vertical.TScrollbar', background=[('active', '#555555')], arrowcolor=[('active', TEXT_COLOR_BLUE)])
    desc_y = h // 2 - 180
    desc_dynamic_y = h // 2 - 150
    title_y = h // 2 - 100
    btn_inicio_y = h // 2 + 20
    btn_salir_y = h // 2 + 110
    instr_y = h // 2 + 230
    desc_text_static = "Un predictor de clima basado en tendencias históricas"
    root_canvas.create_text(w // 2, desc_y, text=desc_text_static, font=DESCRIPTION_FONT, fill=TEXT_COLOR_BLUE, justify="center", tags="desc_text")
    root_canvas.create_text(w // 2, desc_dynamic_y, text=f"(Datos 2000-2024 de {NOMBRE_BASE_DISPLAY})", font=DESCRIPTION_FONT, fill=TEXT_COLOR_BLUE, justify="center", tags="desc_dynamic_text")
    title_text = "Bienvenido a su programa de prediccion\nde temperatura CLIMAS LOCOS"
    root_canvas.create_text(w // 2, title_y, text=title_text, font=TITLE_FONT, fill=TEXT_COLOR_BLUE, justify="center", tags="title_text")
    def cargar_base_func():
        global NOMBRE_ARCHIVO_DATOS, NOMBRE_BASE_DISPLAY, patron_tmax_g, patron_tmin_g, slope_tmax_g, slope_tmin_g, intercept_tmax_g, intercept_tmin_g, mae_tmax_g, mae_tmin_g
        f = filedialog.askopenfilename(title="Seleccionar Base de Datos", filetypes=[("Archivos CSV", "*.csv")])
        if f:
            NOMBRE_ARCHIVO_DATOS = f
            NOMBRE_BASE_DISPLAY = os.path.splitext(os.path.basename(f))[0]
            ptmax, ptmin, stmax, stmin, itmax, itmin, mtmax, mtmin = cargar_datos_y_calcular_tendencias()
            if ptmax is not None:
                patron_tmax_g, patron_tmin_g, slope_tmax_g, slope_tmin_g, intercept_tmax_g, intercept_tmin_g, mae_tmax_g, mae_tmin_g = ptmax, ptmin, stmax, stmin, itmax, itmin, mtmax, mtmin
                root_canvas.itemconfigure("desc_dynamic_text", text=f"(Datos 2000-2024 de {NOMBRE_BASE_DISPLAY})")
                root_canvas.itemconfigure("status_btn_text", text=f"Base {NOMBRE_BASE_DISPLAY} cargada")
                messagebox.showinfo("Exito", "Base de datos cargada correctamente.")
            else:
                messagebox.showerror("Error", "El archivo seleccionado no tiene el formato correcto.")
                NOMBRE_ARCHIVO_DATOS = None
                NOMBRE_BASE_DISPLAY = "PENDIENTE"
    btn_inicio = ttk.Button(root_canvas, text="INICIO", command=abrir_pantalla_predictor, style="Large.TButton")
    root_canvas.create_window(w // 2 - 160, btn_inicio_y, window=btn_inicio, tags="btn_inicio")
    btn_cargar = ttk.Button(root_canvas, text="CARGAR BASE", command=cargar_base_func, style="Large.TButton")
    root_canvas.create_window(w // 2 + 160, btn_inicio_y, window=btn_cargar, tags="btn_cargar")
    root_canvas.create_text(w // 2 + 350, btn_inicio_y, text="", font=("Courier New", 12, "italic"), fill=TEXT_COLOR_GREEN, anchor="w", tags="status_btn_text")
    btn_salir = ttk.Button(root_canvas, text="Salir", command=salir, style="TButton")
    root_canvas.create_window(w // 2, btn_salir_y, window=btn_salir, tags="btn_salir")
    instrucciones_texto = ("INSTRUCCIONES DE USO:\n\n1. Primero, carga una base de datos CSV.\n2. Presiona 'INICIO' para abrir el panel de predicción.\n3. Selecciona un 'Año' (obligatorio).\n4. Opcional: Selecciona 'Mes' y 'Día'.\n5. Presiona 'Predecir'.")
    root_canvas.create_text(w // 2, instr_y, text=instrucciones_texto, font=("Courier New", 14, "italic"), fill=TEXT_COLOR_GREEN, justify="center", tags="instr_text")
    def on_root_resize(event):
        global background_photo_dict
        try:
            w, h = event.width, event.height
            if w <= 1 or h <= 1: return
            if PIL_DISPONIBLE and os.path.exists(NOMBRE_IMAGEN):
                img = Image.open(NOMBRE_IMAGEN).resize((w, h), Image.LANCZOS)
                background_photo_dict['root_resize'] = ImageTk.PhotoImage(img)
                root_canvas.itemconfig("bg", image=background_photo_dict['root_resize'])
            desc_y = h // 2 - 180
            desc_dynamic_y = h // 2 - 150
            title_y = h // 2 - 100
            btn_inicio_y = h // 2 + 20
            btn_salir_y = h // 2 + 110
            instr_y = h // 2 + 230
            root_canvas.coords("desc_text", w // 2, desc_y)
            root_canvas.coords("desc_dynamic_text", w // 2, desc_dynamic_y)
            root_canvas.coords("title_text", w // 2, title_y)
            root_canvas.coords("btn_inicio", w // 2 - 160, btn_inicio_y)
            root_canvas.coords("btn_cargar", w // 2 + 160, btn_inicio_y)
            root_canvas.coords("status_btn_text", w // 2 + 350, btn_inicio_y)
            root_canvas.coords("btn_salir", w // 2, btn_salir_y)
            root_canvas.coords("instr_text", w // 2, instr_y)
        except:
            pass
    root_canvas.bind('<Configure>', on_root_resize, add='+')
    root.state('zoomed')
    root.mainloop()