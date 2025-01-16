import os
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib import pyplot as plt
from pandas.io.formats.style import Styler
from dotenv import load_dotenv
import numpy as np
from datetime import datetime
import requests
from sklearn.linear_model import LinearRegression
import random
# ===============================
# 2. CONFIGURACIÓN DE LA PÁGINA + ESTILO META ADS
# (Debe ser lo primero que hace Streamlit)
# ===============================
st.set_page_config(
    page_title="Análisis de Campañas de Meta con IA",
    layout="wide",
)

# =============================================================================
# ELIMINAMOS/COMENTAMOS LA LÓGICA DE EXPERIMENTAL QUERY PARAMS (DEPRECADAS)
# =============================================================================
# if "auto_refreshed" not in st.session_state:
#     st.session_state["auto_refreshed"] = False
# query_params = st.experimental_get_query_params()
# if "refresh" in query_params and not st.session_state["auto_refreshed"]:
#     st.session_state["auto_refreshed"] = True
# =============================================================================

# ===============================
# 1. CARGA DE VARIABLES DE ENTORNO
# ===============================
load_dotenv("a.env")

# -----------------------------------------------------
# Agregar estilos inspirados en la línea de diseño de Meta Ads
# -----------------------------------------------------
st.markdown(
    """
    <style>
    /* FUENTE GENERAL */
    html, body, [class^="st"], [data-testid="stHeader"], [data-testid="stSidebar"] {
        font-family: "Helvetica Neue", "Arial", sans-serif;
    }
    /* BACKGROUND GENERAL */
    .css-18e3th9 {
        background-color: #F0F2F5;
    }
    /* TITULOS */
    h1, h2, h3 {
        color: #1877f2;
        font-weight: 700;
    }
    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            rgba(0, 72, 183, 0.65) 20%,
            rgba(0, 72, 183, 0.65) 90%
        );
        color: #ffffff;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] h4, 
    [data-testid="stSidebar"] h5, 
    [data-testid="stSidebar"] h6, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] span {
        color: #ffffff !important;
    }
    .stRadio > label {
        display: none;
    }
    .stRadio div[role="radiogroup"] {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        margin-top: 0.5rem;
    }
    .stRadio div[role="radio"] {
        background-color: rgba(255, 255, 255, 0.2);
        color: #ffffff;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        border: 1px solid #ffffff;
        cursor: pointer;
        transition: background-color 0.2s ease;
    }
    .stRadio div[role="radio"]:hover {
        background-color: rgba(255, 255, 255, 0.3);
    }
    .stRadio div[role="radio"][aria-checked="true"] {
        background-color: rgba(255, 255, 255, 0.4);
        border-color: #ffffff;
    }
    .stRadio label, .stRadio span {
        color: #ffffff !important;
    }
    .css-1aumxhk h1 {
        color: #fff !important;
    }
    .stButton>button {
        background-color: #1877f2;
        color: #ffffff;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1rem;
        border: 1px solid #1877f2;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #0056d6;
        border: 1px solid #0056d6;
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
        padding: 0.6rem;
        border: 1px solid #ccc;
        font-size: 0.95rem;
    }
    .stCheckbox>div>label>div[data-testid="stMarkdownContainer"]>p {
        font-size: 0.9rem;
        font-weight: 500;
    }
    .css-1d391kg {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 10px;
    }
    .stSelectbox>div>div>div>button {
        border-radius: 8px;
        border: 1px solid #ccc;
    }
    .stDateInput>div>div>input {
        border-radius: 8px;
        padding: 0.6rem;
        border: 1px solid #ccc;
        font-size: 0.9rem;
    }
    .stDataFrame tbody tr:hover {
        background-color: #f1f3f5;
    }
    .dataframe th, .dataframe td {
        font-size: 0.88rem;
        padding: 6px 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ===============================
# 3. CONFIGURACIÓN DE LA API DE GEMINI
# ===============================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("No se encontró 'GOOGLE_API_KEY' en variables de entorno. Por favor, configúralo.")
    st.stop()

DEFAULT_MODEL_NAME = "gemini-1.5-flash"

# ===============================
# 4. CARGA DE DATOS
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("df_activos.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

data = load_data()

# ===============================
# CONFIGURACIÓN DE NAVEGACIÓN ENTRE PÁGINAS
# ===============================
st.sidebar.title("Navegación")
selected_page = st.sidebar.radio(
    "Selecciona una página:",
    ["Conversión", "Engagement", "Personalizado"] 
)

if selected_page == "Conversión":

    # ===============================
    # 5. ANÁLISIS AUTOMATIZADO POR IA
    # ===============================
    def generar_resumen_ultimas_2_semanas(df):
            """
            1) Detecta cambios reales de presupuesto (1 por día) en los últimos 20 días.
            - Menciona 1 sola vez/día si delta != 0 en 'Dailybudget'.
            - Compara 7 días antes vs 7 días después para ver top 3 y bottom 3 en Participación y CPA.
            2) Detecta 'apagados' en los últimos 20 días, donde un anuncio reduce >=50% su 'TotalCost' e 'Impressions'
            respecto al día anterior (mencionado solo 1 vez por anuncio).
            - También compara 7 días antes vs 7 días después del día de apagado.
            3) Mantiene Variaciones Generales del CPA (2 sem vs 2 sem ant),
            Leads del mes y de la última semana,
            Regresión lineal de CPA,
            Conclusión y Estrategia.
            """
            resumen = ""
            try:
                import numpy as np
                from sklearn.linear_model import LinearRegression

                if df.empty:
                    return "No hay datos disponibles."

                fecha_actual = df["Date"].max()
                if pd.isnull(fecha_actual):
                    return "No hay fechas válidas."

                # ============ PERIODOS PARA CPA, LEADS, ETC. ============
                fecha_inicio_2sem = fecha_actual - pd.Timedelta(days=14)
                df_ult_2sem = df[df["Date"] >= fecha_inicio_2sem].copy()

                fecha_inicio_4sem = fecha_inicio_2sem - pd.Timedelta(days=14)
                df_2sem_ant = df[(df["Date"] >= fecha_inicio_4sem) & (df["Date"] < fecha_inicio_2sem)].copy()

                inicio_mes = fecha_actual.replace(day=1)
                df_mes_actual = df[df["Date"] >= inicio_mes].copy()

                fecha_semana = fecha_actual - pd.Timedelta(days=6)
                df_ult_semana = df[df["Date"] >= fecha_semana].copy()

                                # ============ RESUMEN GENERAL  ============
                # ============ 4) Leads del mes + última semana ============
                resumen += "\n##### Resumen General\n"
                leads_mes = df_mes_actual["PTP_total"].sum()
                leads_semana = df_ult_semana["PTP_total"].sum()
                pct_semana = (leads_semana/leads_mes*100) if leads_mes>0 else 0
                resumen += (f"- Total leads del mes: **{leads_mes:.0f}**.\n"
                            f"- Última semana: **{leads_semana:.0f}** leads "
                            f"({pct_semana:.2f}% del total).\n")
                
                # ============ 3) Variación CPA (últ. 2 sem vs 2 sem ant) ============
                cpa_2n = 0
                cpa_2n_ant = 0

                if df_ult_2sem["PTP_total"].sum() > 0:
                    cpa_2n = df_ult_2sem["TotalCost"].sum() / df_ult_2sem["PTP_total"].sum()
                if df_2sem_ant["PTP_total"].sum() > 0:
                    cpa_2n_ant = df_2sem_ant["TotalCost"].sum() / df_2sem_ant["PTP_total"].sum()

                var_cpa_global = ((cpa_2n - cpa_2n_ant) / cpa_2n_ant * 100) if cpa_2n_ant > 0 else 0
                diff_dollar = cpa_2n - cpa_2n_ant  # Diferencia en dólares

                if var_cpa_global > 0:
                    difference_str = f"${abs(diff_dollar):,.2f} más alto ({abs(var_cpa_global):,.2f}%)"
                elif var_cpa_global < 0:
                    difference_str = f"${abs(diff_dollar):,.2f} más bajo ({abs(var_cpa_global):,.2f}%)"
                else:
                    difference_str = "igual"

                resumen += (
                    f"\n- CPA promedio de las últimas 2 semanas: ${cpa_2n:,.2f}; "
                    f"${abs(diff_dollar):,.2f} {'más alto' if var_cpa_global > 0 else 'más bajo' if var_cpa_global < 0 else 'igual'} "
                    f"({abs(var_cpa_global):,.2f}%) respecto a las 2 semanas anteriores.\n"
)



                # ============ 5) Regresión lineal de CPA (últ. 2 sem) ============
                df_ok = df_ult_2sem[
                    (df_ult_2sem["PTP_total"]>0) &
                    (df_ult_2sem["TotalCost"]>0) &
                    (df_ult_2sem["Impressions"]>0)
                ].copy()
                tendencias_cpa = []
                if not df_ok.empty:
                    for adname, grupo in df_ok.groupby("Adname"):
                        grupo = grupo.sort_values("Date")
                        if len(grupo)<2:
                            continue
                        x = np.arange(len(grupo)).reshape(-1,1)
                        y = grupo.apply(lambda row: row["TotalCost"]/row["PTP_total"], axis=1).values.reshape(-1,1)
                        model = LinearRegression().fit(x, y)
                        pen = model.coef_[0][0]
                        tendencias_cpa.append((adname, pen))
                    tendencias_cpa.sort(key=lambda x: x[1], reverse=True)


                # ============ 1) ANALISIS DE PRESUPUESTO: ULTIMOS 20 DIAS ============
                resumen += "\n\n ##### Análisis de Cambios de Presupuesto (últimos 20 días)\n\n"
                fecha_inicio_20dias = fecha_actual - pd.Timedelta(days=20)
                df_20 = df[df["Date"] >= fecha_inicio_20dias].copy()

                # a) Cálculo de 'delta' diario (1 ocurrencia/día)
                df_presup = (
                    df_20.groupby("Date", as_index=False)
                        .agg(Dailybudget=("Dailybudget","mean"))
                        .sort_values("Date")
                )
                df_presup["Delta_Presupuesto"] = df_presup["Dailybudget"].diff()

                # Filtra cambios reales (delta != 0)
                df_cambios = df_presup[
                    (df_presup["Delta_Presupuesto"].notnull()) &
                    (df_presup["Delta_Presupuesto"] != 0)
                ].copy()

                # b) Itera cada cambio y hace la comparación (7 días pre/post)
                if df_cambios.empty:
                    resumen += "No hubo cambios de presupuesto en los últimos 20 días.\n"
                else:
                    for idx in df_cambios.index:
                        dia_cambio = df_cambios.loc[idx, "Date"]
                        delta_pres = df_cambios.loc[idx, "Delta_Presupuesto"]
                        new_val = df_cambios.loc[idx, "Dailybudget"]
                        # Valor anterior (si existe)
                        if idx-1 in df_presup.index:
                            old_val = df_presup.loc[idx-1, "Dailybudget"]
                        else:
                            old_val = 0

                        dia_str = dia_cambio.strftime("%Y-%m-%d")
                        resumen += (f"- #### El [{dia_str}] hubo un cambio de presupuesto: "
                                    f"desde: {old_val:.0f} hasta: {new_val:.0f} (Δ={delta_pres:.2f}).\n")


                        # 7 días pre/post
                        dmin_pre = dia_cambio - pd.Timedelta(days=7)
                        dmax_pre = dia_cambio - pd.Timedelta(days=1)
                        dmin_post = dia_cambio
                        dmax_post = dia_cambio + pd.Timedelta(days=6)

                        df_pre = df[(df["Date"]>=dmin_pre) & (df["Date"]<=dmax_pre)].copy()
                        df_post = df[(df["Date"]>=dmin_post) & (df["Date"]<=dmax_post)].copy()

                        if not df_pre.empty and not df_post.empty:
                            tot_pre = df_pre["TotalCost"].sum()
                            tot_post = df_post["TotalCost"].sum()
                            resumen += (f"")

                            # ---- Participación (TotalCost) ----
                            df_part_pre = (
                                df_pre.groupby("Adname")["TotalCost"].sum()
                                / tot_pre * 100 if tot_pre>0 else 0
                            )
                            df_part_post = (
                                df_post.groupby("Adname")["TotalCost"].sum()
                                / tot_post * 100 if tot_post>0 else 0
                            )
                            df_merge = pd.merge(
                                df_part_pre.reset_index().rename(columns={"TotalCost":"Part_pre"}),
                                df_part_post.reset_index().rename(columns={"TotalCost":"Part_post"}),
                                on="Adname", how="outer"
                            ).fillna(0)
                            df_merge["Delta_part"] = df_merge["Part_post"] - df_merge["Part_pre"]

                            # Subtítulos claros
                            resumen += "\n##### TOP 3 suben participación:\n"
                            df_merge.sort_values("Delta_part", ascending=False, inplace=True)
                            for _, r2 in df_merge.head(3).iterrows():
                                if r2["Delta_part"]>0:
                                    resumen += f"    - {r2['Adname']}: +{r2['Delta_part']:.2f} pts.\n"

                            resumen += "\n##### TOP 3 bajan participación:\n"
                            df_merge.sort_values("Delta_part", ascending=True, inplace=True)
                            for _, r2 in df_merge.head(3).iterrows():
                                if r2["Delta_part"]<0:
                                    resumen += f"    - {r2['Adname']}: {r2['Delta_part']:.2f} pts.\n"

                            # ---- CPA (7 días pre vs post) ----
                            df_cpa_pre = df_pre.groupby("Adname").agg({
                                "TotalCost":"sum","PTP_total":"sum"
                            }).reset_index()
                            df_cpa_pre["CPA_pre"] = df_cpa_pre.apply(
                                lambda row: row["TotalCost"]/row["PTP_total"] if row["PTP_total"]>0 else 0, axis=1
                            )

                            df_cpa_post = df_post.groupby("Adname").agg({
                                "TotalCost":"sum","PTP_total":"sum"
                            }).reset_index()
                            df_cpa_post["CPA_post"] = df_cpa_post.apply(
                                lambda row: row["TotalCost"]/row["PTP_total"] if row["PTP_total"]>0 else 0, axis=1
                            )

                            df_cpa_m = pd.merge(
                                df_cpa_pre[["Adname","CPA_pre"]],
                                df_cpa_post[["Adname","CPA_post"]],
                                on="Adname", how="outer"
                            ).fillna(0)
                            df_cpa_m["Delta_cpa"] = df_cpa_m["CPA_post"] - df_cpa_m["CPA_pre"]

                            resumen += "\n##### TOP 3 mayor aumento de CPA:\n"
                            df_cpa_m.sort_values("Delta_cpa", ascending=False, inplace=True)
                            for _, rowc in df_cpa_m.head(3).iterrows():
                                if rowc["Delta_cpa"]>0:
                                    resumen += (f"    - {rowc['Adname']}: CPA {rowc['CPA_pre']:.2f}->{rowc['CPA_post']:.2f} "
                                                f"(Δ={rowc['Delta_cpa']:.2f}).\n")

                            resumen += "\n##### TOP 3 mayor disminución de CPA:\n"
                            df_cpa_m.sort_values("Delta_cpa", ascending=True, inplace=True)
                            for _, rowc in df_cpa_m.head(3).iterrows():
                                if rowc["Delta_cpa"]<0:
                                    resumen += (f"    - {rowc['Adname']}: CPA {rowc['CPA_pre']:.2f}->{rowc['CPA_post']:.2f} "
                                                f"(Δ={rowc['Delta_cpa']:.2f}).\n")

                        else:
                            resumen += "  (No hubo datos para comparar 7 días pre/post)\n"

                # ============ 2) ANALISIS DE APAGADOS (últimos 20 días) ============
                resumen += "\n##### Anuncios Nuevos y Apagados (últimos 20 días)\n\n"
                df_apag = df[df["Date"] >= (fecha_actual - pd.Timedelta(days=20))].copy()

                # Iteramos día a día en df_apag, comparando con el día anterior
                # Si un anuncio reduce >=50% su TotalCost e Impressions => se APAGA (1 vez)
                apagados_mencionados = set()

                dias_ordenados = sorted(df_apag["Date"].unique())
                for i in range(1, len(dias_ordenados)):
                    dia_ant = dias_ordenados[i-1]
                    dia_act = dias_ordenados[i]

                    df_dia_ant = df_apag[df_apag["Date"]==dia_ant].groupby("Adname")[["TotalCost","Impressions"]].sum()
                    df_dia_act = df_apag[df_apag["Date"]==dia_act].groupby("Adname")[["TotalCost","Impressions"]].sum()

                    # Cruce por índice (Adname)
                    df_join = df_dia_ant.join(df_dia_act, lsuffix="_ant", rsuffix="_act", how="inner")

                    # Filtra los "apagados"
                    df_off = df_join[
                        (df_join["TotalCost_ant"]>0) &
                        (df_join["Impressions_ant"]>0) &
                        ((df_join["TotalCost_act"] <= df_join["TotalCost_ant"]*0.5) &
                        (df_join["Impressions_act"] <= df_join["Impressions_ant"]*0.5))
                    ]

                    for ad_off in df_off.index:
                        if ad_off not in apagados_mencionados:
                            dia_str = pd.to_datetime(dia_act).strftime("%Y-%m-%d")
                            resumen += (f"- '{ad_off}' se APAGÓ el {dia_str}")

                            # 7 días pre/post a "dia_act"
                            dmin_pre = pd.to_datetime(dia_act) - pd.Timedelta(days=7)
                            dmax_pre = pd.to_datetime(dia_act) - pd.Timedelta(days=1)
                            dmin_post = pd.to_datetime(dia_act)
                            dmax_post = pd.to_datetime(dia_act) + pd.Timedelta(days=6)

                            df_pre = df[(df["Date"]>=dmin_pre)&(df["Date"]<=dmax_pre)].copy()
                            df_post = df[(df["Date"]>=dmin_post)&(df["Date"]<=dmax_post)].copy()
                            if not df_pre.empty and not df_post.empty:
                                tot_pre = df_pre["TotalCost"].sum()
                                tot_post = df_post["TotalCost"].sum()
                                resumen += (f"")

                                # Top 3 / Bottom 3 Participación
                                df_part_pre = (
                                    df_pre.groupby("Adname")["TotalCost"].sum()/tot_pre*100 if tot_pre>0 else 0
                                )
                                df_part_post = (
                                    df_post.groupby("Adname")["TotalCost"].sum()/tot_post*100 if tot_post>0 else 0
                                )
                                df_m = pd.merge(
                                    df_part_pre.reset_index().rename(columns={"TotalCost":"Part_pre"}),
                                    df_part_post.reset_index().rename(columns={"TotalCost":"Part_post"}),
                                    on="Adname", how="outer"
                                ).fillna(0)
                                df_m["Delta_part"] = df_m["Part_post"] - df_m["Part_pre"]

                                resumen += "\n##### TOP 3 suben participación:\n"
                                df_m.sort_values("Delta_part", ascending=False, inplace=True)
                                for _, r2 in df_m.head(3).iterrows():
                                    if r2["Delta_part"]>0:
                                        resumen += f"    - {r2['Adname']}: +{r2['Delta_part']:.2f} pts.\n"

                                resumen += "\n##### TOP 3 bajan participación:\n"
                                df_m.sort_values("Delta_part", ascending=True, inplace=True)
                                for _, r2 in df_m.head(3).iterrows():
                                    if r2["Delta_part"]<0:
                                        resumen += f"    - {r2['Adname']}: {r2['Delta_part']:.2f} pts.\n"

                                # CPA en pre vs post
                                df_cpa_pre = df_pre.groupby("Adname").agg({"TotalCost":"sum","PTP_total":"sum"}).reset_index()
                                df_cpa_pre["CPA_pre"] = df_cpa_pre.apply(
                                    lambda row: row["TotalCost"]/row["PTP_total"] if row["PTP_total"]>0 else 0, axis=1
                                )
                                df_cpa_post = df_post.groupby("Adname").agg({"TotalCost":"sum","PTP_total":"sum"}).reset_index()
                                df_cpa_post["CPA_post"] = df_cpa_post.apply(
                                    lambda row: row["TotalCost"]/row["PTP_total"] if row["PTP_total"]>0 else 0, axis=1
                                )
                                df_cpam = pd.merge(
                                    df_cpa_pre[["Adname","CPA_pre"]],
                                    df_cpa_post[["Adname","CPA_post"]],
                                    on="Adname", how="outer"
                                ).fillna(0)
                                df_cpam["Delta_cpa"] = df_cpam["CPA_post"] - df_cpam["CPA_pre"]

                                resumen += "\n##### TOP 3 mayor subida de CPA:\n"
                                df_cpam.sort_values("Delta_cpa", ascending=False, inplace=True)
                                for _, rr in df_cpam.head(3).iterrows():
                                    if rr["Delta_cpa"]>0:
                                        resumen += (f"    - {rr['Adname']}: "
                                                    f"{rr['CPA_pre']:.2f}->{rr['CPA_post']:.2f} (Δ={rr['Delta_cpa']:.2f}).\n")
                                resumen += "\n##### Top 3 mayor bajada de CPA:\n"
                                df_cpam.sort_values("Delta_cpa", ascending=True, inplace=True)
                                for _, rr in df_cpam.head(3).iterrows():
                                    if rr["Delta_cpa"]<0:
                                        resumen += (f"    - {rr['Adname']}: "
                                                    f"{rr['CPA_pre']:.2f}->{rr['CPA_post']:.2f} (Δ={rr['Delta_cpa']:.2f}).\n")
                            else:
                                resumen += "  (No hubo datos para comparar 7 días pre/post del apagado)\n"

                            apagados_mencionados.add(ad_off)

                # ============ 6) Conclusión y Estrategia ============
                resumen += "\n##### Conclusión y Estrategia\n"
                if tendencias_cpa:
                    resumen += "Observa estos anuncios con mayor tendencia negativa de CPA:\n"
                    for ad, pen in tendencias_cpa[:3]:
                        resumen += f"- `{ad}` con pendiente {pen:.2f}.\n"
                else:
                    resumen += "No hubo suficientes datos para tendencias de CPA.\n"
                
            except Exception as e:
                resumen = f"Ocurrió un error al generar el resumen: {e}"

            return resumen
    # ===============================
    # 6. CONSULTA EN LENGUAJE NATURAL
    # ===============================
    def consulta_lenguaje_natural(pregunta, datos):
        """Realiza una consulta a Gemini con el DataFrame como contexto."""
        try:
            datos_csv = datos.to_csv(index=False)

            prompt = f"""
            Actúas como un analista experto en marketing digital, especializado en meta ads. Los datos relevantes se encuentran en formato CSV y tienen las siguientes columnas:
            {', '.join(datos.columns)}.

            Los datos son los siguientes:

    csv
            {datos_csv}


            - Los datos representan métricas de campañas publicitarias en Meta.
            - La columna 'Date' contiene las fechas en formato yyyy-mm-dd y esta en formato Date.
            - La columna 'PTP_total' contiene los leads generados.
            - La columna 'TotalCost' contiene el costo total en la moneda local.
            - La columna 'CPA' contiene el costo por adquisición (CPA).
            - Otras columnas contienen información relevante de campañas y anuncios.
            - Si requieres evaluar el CPA de un periodo para un conjunto o anuncio en particular, deberás calcularlo como la sumatoria del TotalCost / la sumatoria del PTP_total para el conjunto o anuncio respectivo. Asegúrate de que estos cálculos sean precisos..
                    
            La pregunta del usuario es: "{pregunta}"

            Responde basándote exclusivamente en los datos proporcionados por 'datos_csv'. Si la consulta implica un cálculo, realiza el cálculo con precisión y entrega la respuesta en un formato claro y estructurado. Si faltan datos para responder la pregunta, indícalo explícitamente.

            Asegúrate de que las fechas estén correctamente interpretadas y que las respuestas sean precisas y directamente relacionadas con la consulta.
            """

            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{DEFAULT_MODEL_NAME}:generateContent",
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": GOOGLE_API_KEY
                },
                json={
                   "contents": [{
                        "parts": [{"text": prompt}]
                    }]
                }
            )

            if response.status_code == 200:
                respuesta = response.json().get("candidates")[0].get("content").get("parts")[0].get("text", "No se pudo obtener una respuesta.")
            else:
                respuesta = f"Error en la solicitud: {response.status_code} - {response.text}"

            return respuesta

        except Exception as e:
            return f"Error procesando la consulta: {e}"
    # ===============================
    # 7. MOSTRAR RESUMEN IA Y CONSULTA
    # ===============================
    st.title("Análisis de Campañas con IA")
    st.markdown("<h3 style='color: #1877f2; font-weight: 700;'>Resumen Automatizado (IA)</h3>", unsafe_allow_html=True)
    resumen = generar_resumen_ultimas_2_semanas(data)
    st.markdown(resumen)

    st.markdown("<h3 style='color: #1877f2; font-weight: 700;'>Consulta en Lenguaje Natural</h3>", unsafe_allow_html=True)
    pregunta = st.text_input("Escribe tu consulta en lenguaje natural")
    hacer_consulta = st.button("Consultar")
    if hacer_consulta and pregunta:
        respuesta = consulta_lenguaje_natural(pregunta, data)
        st.markdown("<h4 style='color: #1877f2;'>Respuesta IA:</h4>", unsafe_allow_html=True)
        st.markdown(respuesta)

    # ===============================
    # 8. FUNCIONES AUXILIARES
    # ===============================
    @st.cache_data
    def load_hitos_csv(file_path="hitos.csv"):
        import os
        if not os.path.exists(file_path):
            df = pd.DataFrame(columns=["Date","Description"])
            df.to_csv(file_path, index=False)
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        else:
            df = pd.read_csv(file_path, parse_dates=["Date"])
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        return df

    def save_hitos_csv(df, file_path="hitos.csv"):
        df.to_csv(file_path, index=False)

    def find_tuesday_to_monday_week(date):
        if pd.isnull(date):
            return (pd.NaT, pd.NaT)
        offset = (date.weekday() - 1) % 7
        week_start = date - pd.Timedelta(days=offset)
        week_end = week_start + pd.Timedelta(days=6)
        return (week_start, week_end)

    def calcular_metricas_semanales(df):
        df_copy = df.copy()
        df_copy["Date"] = pd.to_datetime(df_copy["Date"], errors="coerce")
        df_copy[["Inicio_periodo", "Fin_periodo"]] = df_copy["Date"].apply(
            lambda x: pd.Series(find_tuesday_to_monday_week(x))
        )
        # Agregamos 'Clicks' en aggregator
        weekly_group = (
            df_copy
            .groupby(["AdSetname", "Inicio_periodo"], as_index=False)
            .agg({
                "TotalCost": "sum",
                "PTP_total": "sum",
                "Impressions": "sum",
                "Clicks": "sum"
            })
        )
        # Emparejamos con Fin_periodo
        week_ends = df_copy[["Inicio_periodo", "Fin_periodo"]].drop_duplicates()
        weekly_group = pd.merge(weekly_group, week_ends, on="Inicio_periodo", how="left")

        # CPA
        weekly_group["CPA"] = weekly_group.apply(
            lambda row: row["TotalCost"]/row["PTP_total"] if row["PTP_total"]>0 else 0,
            axis=1
        )

        # Tasa de Conversión
        weekly_group["Tasa_conversion"] = weekly_group.apply(
            lambda row: (row["PTP_total"]/row["Impressions"]*100) if row["Impressions"]>0 else 0,
            axis=1
        )

        # CTR
        weekly_group["CTR"] = weekly_group.apply(
            lambda row: (row["Clicks"]/row["Impressions"]*100) if row["Impressions"]>0 else 0,
            axis=1
        )

        weekly_group.sort_values(by=["AdSetname","Inicio_periodo"], inplace=True)

        # Variaciones
        weekly_group["TotalCost_variation"] = weekly_group.groupby("AdSetname")["TotalCost"].diff().fillna(0)
        weekly_group["CPA_variation"] = weekly_group.groupby("AdSetname")["CPA"].diff().fillna(0)
        weekly_group["Tasa_conversion_variation"] = weekly_group.groupby("AdSetname")["Tasa_conversion"].diff().fillna(0)
        weekly_group["CTR_variation"] = weekly_group.groupby("AdSetname")["CTR"].diff().fillna(0)

        # Participación de costo
        weekly_group["participación_costo"] = (
            weekly_group["TotalCost"] /
            weekly_group.groupby("Inicio_periodo")["TotalCost"].transform("sum") * 100
        ).fillna(0).round(2)

        # Redondeos
        weekly_group["TotalCost"] = weekly_group["TotalCost"].round(0).astype(int)
        weekly_group["PTP_total"] = weekly_group["PTP_total"].round(0).astype(int)
        weekly_group["Impressions"] = weekly_group["Impressions"].round(0).astype(int)
        weekly_group["CPA"] = weekly_group["CPA"].round(2)
        weekly_group["Tasa_conversion"] = weekly_group["Tasa_conversion"].round(2)
        weekly_group["CTR"] = weekly_group["CTR"].round(2)
        weekly_group["TotalCost_variation"] = weekly_group["TotalCost_variation"].round(2)
        weekly_group["CPA_variation"] = weekly_group["CPA_variation"].round(2)
        weekly_group["Tasa_conversion_variation"] = weekly_group["Tasa_conversion_variation"].round(2)
        weekly_group["CTR_variation"] = weekly_group["CTR_variation"].round(2)

        final_cols = [
            "AdSetname", "Inicio_periodo", "Fin_periodo",
            "TotalCost", "PTP_total", "Impressions", "Clicks",
            "CPA", "Tasa_conversion", "CTR",
            "TotalCost_variation", "CPA_variation", "Tasa_conversion_variation", "CTR_variation",
            "participación_costo"
        ]
        return weekly_group[final_cols]

    def calcular_metricas_generales(df):
        general_group = (
            df.groupby("Inicio_periodo", as_index=False)
            .agg({
                "TotalCost": "sum",
                "PTP_total": "sum",
                "Impressions": "sum"
            })
        )
        general_group["CPA"] = general_group.apply(
            lambda row: row["TotalCost"]/row["PTP_total"] if row["PTP_total"]>0 else 0,
            axis=1
        )
        general_group["Tasa_conversion"] = general_group.apply(
            lambda row: (row["PTP_total"]/row["Impressions"]*100) if row["Impressions"]>0 else 0,
            axis=1
        )
        general_group.sort_values(by=["Inicio_periodo"], inplace=True)
        general_group["TotalCost_variation"] = general_group["TotalCost"].diff().fillna(0).round(2)
        general_group["CPA_variation"] = general_group["CPA"].diff().fillna(0).round(2)
        general_group["Tasa_conversion_variation"] = general_group["Tasa_conversion"].diff().fillna(0).round(2)
        general_group["participación_costo"] = 100
        return general_group

    def generar_tabla_pivot(df, adsetname, metrics_order, general_df=None):
        if adsetname == "General" and general_df is not None:
            df_adset = general_df
        else:
            df_adset = df[df["AdSetname"] == adsetname]

        df_melted = df_adset.melt(
            id_vars=["AdSetname","Inicio_periodo"] if adsetname!="General" else ["Inicio_periodo"],
            value_vars=[col for col in metrics_order if col in df_adset.columns],
            var_name="Metric",
            value_name="Value"
        )
        df_pivot = df_melted.pivot_table(
            index="Metric",
            columns="Inicio_periodo",
            values="Value",
            aggfunc="first"
        )

        # Renombrar columnas
        new_cols = {}
        for c in df_pivot.columns:
            if isinstance(c, pd.Timestamp):
                new_cols[c] = c.strftime("%Y-%m-%d")
            else:
                new_cols[c] = c
        df_pivot.rename(columns=new_cols, inplace=True)

        df_pivot.rename(index={"participación_costo":"Participación en el costo total (%)"}, inplace=True)
        return df_pivot

    def estilizar_tabla(df_pivot):
        def format_metric(val, metric):
            try:
                val_float = float(val)
            except:
                return val
            if metric == "Tasa_conversion":
                return f"{val_float:.2f}%"
            elif metric == "CPA":
                return f"{val_float:.2f}"
            elif metric == "CPA_variation":
                return f"{val_float:.2f}"
            elif metric == "TotalCost":
                return f"{int(round(val_float, 0))}"
            elif metric == "TotalCost_variation":
                return f"{int(round(val_float, 0))}"
            elif metric == "PTP_total":
                return f"{int(round(val_float, 0))}"
            elif metric == "Tasa_conversion_variation":
                return f"{val_float:.2f}"
            elif metric == "Participación en el costo total (%)":
                return f"{val_float:.2f}"
            return val
        def highlight_cpa_variation(val):
            try:
                numeric_val = float(val)
            except:
                return ""
            if numeric_val>0:
                return "color: red; font-weight: bold;"
            elif numeric_val<0:
                return "color: green; font-weight: bold;"
            else:
                return ""
        df_formatted = df_pivot.copy()
        for metric_name in df_formatted.index:
            for c in df_formatted.columns:
                cell_val = df_formatted.loc[metric_name, c]
                df_formatted.loc[metric_name, c] = format_metric(cell_val, metric_name)
        styled_df = df_formatted.style.set_properties(**{
            "font-family":"Helvetica Neue, Arial, sans-serif", 
            "font-size":"14px"
        }).applymap(
            highlight_cpa_variation,
            subset=pd.IndexSlice[["CPA_variation"], :]
        )
        return styled_df

    import plotly.express as px
    def agregar_hitos_a_grafico(fig, hitos):
        for hito in hitos:
            try:
                date_str = str(hito["Date"])
                date_ts = pd.to_datetime(date_str, errors="coerce")
                if not pd.isnull(date_ts):
                    fig.add_shape(
                        type="line",
                        x0=date_ts, x1=date_ts,
                        y0=0, y1=1,
                        xref="x", yref="paper",
                        line=dict(color="red", width=2, dash="dash")
                    )
                    fig.add_annotation(
                        x=date_ts, y=1,
                        xref="x", yref="paper",
                        showarrow=False,
                        xanchor="left",
                        text=hito["descripcion"],
                        font=dict(color="red")
                    )
            except Exception as e:
                st.warning(f"Error procesando el hito '{hito['descripcion']}': {e}")
        return fig

    def generar_grafico_cpa_diario_adset(df, adsetname):
        df_copy = df.copy()
        df_copy["Date"] = pd.to_datetime(df_copy["Date"], errors="coerce")
        adset_daily = df_copy.groupby(["AdSetname","Date"], as_index=False).agg({
            "TotalCost":"sum",
            "PTP_total":"sum"
        })
        adset_daily["CPA"] = adset_daily.apply(
            lambda row: row["TotalCost"]/row["PTP_total"] if row["PTP_total"]>0 else 0,
            axis=1
        )
        if adsetname!="General":
            adset_daily = adset_daily[adset_daily["AdSetname"]==adsetname]
        fig = px.line(adset_daily, x="Date", y="CPA", color="AdSetname", markers=True,
                      title="Evolución del CPA diario por Conjunto (AdSet)")
        fig.update_layout(xaxis_title="Fecha", yaxis_title="CPA")
        if "hitos" in st.session_state:
            fig = agregar_hitos_a_grafico(fig, st.session_state["hitos"])
        return fig

    def generar_grafico_ptp_diario(df, adsetname):
        df_copy = df.copy()
        df_copy["Date"] = pd.to_datetime(df_copy["Date"], errors="coerce")
        adset_daily = df_copy.groupby(["AdSetname","Date"], as_index=False).agg({
            "PTP_total":"sum"
        })
        if adsetname!="General":
            adset_daily = adset_daily[adset_daily["AdSetname"]==adsetname]
        fig = px.line(adset_daily, x="Date", y="PTP_total", color="AdSetname", markers=True,
                      title="Evolución del PTP diario por Conjunto (AdSet)")
        fig.update_layout(xaxis_title="Fecha", yaxis_title="PTP_total")
        if "hitos" in st.session_state:
            fig = agregar_hitos_a_grafico(fig, st.session_state["hitos"])
        return fig

    def generar_grafico_cpa_diario(df, adsetname):
        df_copy = df.copy()
        df_copy["Date"] = pd.to_datetime(df_copy["Date"], errors="coerce")
        ads_daily = df_copy.groupby(["AdSetname","Adname","Date"], as_index=False).agg({
            "TotalCost":"sum",
            "PTP_total":"sum"
        })
        ads_daily["CPA"] = ads_daily.apply(
            lambda row: row["TotalCost"]/row["PTP_total"] if row["PTP_total"]>0 else 0,
            axis=1
        )
        if adsetname!="General":
            ads_daily = ads_daily[ads_daily["AdSetname"]==adsetname]
        fig = px.line(ads_daily, x="Date", y="CPA", color="Adname", markers=True,
                      title="Evolución del CPA diario por Anuncio")
        fig.update_layout(xaxis_title="Fecha", yaxis_title="CPA")
        if "hitos" in st.session_state:
            fig = agregar_hitos_a_grafico(fig, st.session_state["hitos"])
        return fig

    def generar_grafico_ptp_diario_ads(df, adsetname):
        df_copy = df.copy()
        df_copy["Date"] = pd.to_datetime(df_copy["Date"], errors="coerce")
        ads_daily = df_copy.groupby(["AdSetname","Adname","Date"], as_index=False).agg({
            "TotalCost":"sum",
            "PTP_total":"sum"
        })
        if adsetname!="General":
            ads_daily = ads_daily[ads_daily["AdSetname"]==adsetname]
        fig = px.line(ads_daily, x="Date", y="PTP_total", color="Adname", markers=True,
                      title="Evolución del PTP_total diario por Anuncio")
        fig.update_layout(xaxis_title="Fecha", yaxis_title="Leads")
        if "hitos" in st.session_state:
            fig = agregar_hitos_a_grafico(fig, st.session_state["hitos"])
        return fig

    def generar_grafico_tc_diario_adset(df, adsetname):
        df_copy = df.copy()
        df_copy["Date"] = pd.to_datetime(df_copy["Date"], errors="coerce")
        adset_daily = df_copy.groupby(["AdSetname","Date"], as_index=False).agg({
            "PTP_total":"sum",
            "Impressions":"sum"
        })
        adset_daily["TasaConversion"] = adset_daily.apply(
            lambda row: (row["PTP_total"]/row["Impressions"]*100) if row["Impressions"]>0 else 0,
            axis=1
        )
        if adsetname!="General":
            adset_daily = adset_daily[adset_daily["AdSetname"]==adsetname]
        fig = px.line(adset_daily, x="Date", y="TasaConversion", color="AdSetname", markers=True,
                      title="Evolución de la Tasa de Conversión (%) diaria por Conjunto (AdSet)")
        fig.update_layout(xaxis_title="Fecha", yaxis_title="Tasa de Conversión (%)")
        if "hitos" in st.session_state:
            fig = agregar_hitos_a_grafico(fig, st.session_state["hitos"])
        return fig

    def generar_grafico_tc_diario_ads(df, adsetname):
        df_copy = df.copy()
        df_copy["Date"] = pd.to_datetime(df_copy["Date"], errors="coerce")
        ads_daily = df_copy.groupby(["AdSetname","Adname","Date"], as_index=False).agg({
            "PTP_total":"sum",
            "Impressions":"sum"
        })
        ads_daily["TasaConversion"] = ads_daily.apply(
            lambda row: (row["PTP_total"]/row["Impressions"]*100) if row["Impressions"]>0 else 0,
            axis=1
        )
        if adsetname!="General":
            ads_daily = ads_daily[ads_daily["AdSetname"]==adsetname]
        fig = px.line(ads_daily, x="Date", y="TasaConversion", color="Adname", markers=True,
                      title="Evolución de la Tasa de Conversión (%) diaria por Anuncio")
        fig.update_layout(xaxis_title="Fecha", yaxis_title="Tasa de Conversión (%)")
        if "hitos" in st.session_state:
            fig = agregar_hitos_a_grafico(fig, st.session_state["hitos"])
        return fig

    # SIDEBAR - Filtro
    with st.sidebar:
        st.markdown("<h3 style='color: #ffffff; font-weight: 700;'>Selección del Conjunto de Anuncios</h3>", unsafe_allow_html=True)
        adset_list = ["General"] + sorted(data["AdSetname"].dropna().unique().tolist())
        selected_adset = st.selectbox("Conjunto de Anuncios", adset_list)

        st.markdown("<h3 style='color: #ffffff; font-weight: 700;'>Segmentador de Fechas</h3> <p style='color:#ffffff;'>(Sólo afecta Resumen Histórico y Gráficos)</p>", unsafe_allow_html=True)
        start_date = st.date_input("Fecha inicio (Hist/Gráficos)", value=data["Date"].min())
        end_date = st.date_input("Fecha fin (Hist/Gráficos)", value=data["Date"].max())

    # Filtrado
    df_segmentado = data[
        (data["Date"] >= pd.to_datetime(start_date)) &
        (data["Date"] <= pd.to_datetime(end_date))
    ].copy()
    if selected_adset != "General":
        df_segmentado = df_segmentado[df_segmentado["AdSetname"] == selected_adset]

    # (C) RESUMEN HISTÓRICO
    st.markdown("<h3 style='color: #1877f2; font-weight: 700;'>Resumen Histórico</h3>", unsafe_allow_html=True)
    mostrar_tabla_historico = st.checkbox("Mostrar Resumen Histórico")
    if mostrar_tabla_historico:
        df_historico = df_segmentado.groupby(["AdSetname","Adname"], as_index=False).agg({
            "TotalCost":"sum",
            "PTP_total":"sum",
            "Impressions":"sum",
            "Clicks":"sum"
        })
        total_leads = df_historico["PTP_total"].sum()
        total_cost = df_historico["TotalCost"].sum()

        df_historico["CPA"] = df_historico.apply(
            lambda row: row["TotalCost"]/row["PTP_total"] if row["PTP_total"]>0 else 0,
            axis=1
        )
        df_historico["CTR"] = df_historico.apply(
            lambda row: (row["Clicks"]/row["Impressions"]*100) if row["Impressions"]>0 else 0,
            axis=1
        )
        df_historico["Tasa_de_conversion"] = df_historico.apply(
            lambda row: (row["PTP_total"]/row["Impressions"]*100) if row["Impressions"]>0 else 0,
            axis=1
        )
        df_historico["Participacion_Leads"] = df_historico.apply(
            lambda row: (row["PTP_total"]/total_leads*100) if total_leads>0 else 0,
            axis=1
        )
        df_historico["Participacion_Costo"] = df_historico.apply(
            lambda row: (row["TotalCost"]/total_cost*100) if total_cost>0 else 0,
            axis=1
        )
        df_historico["CPA"] = df_historico["CPA"].round(2)
        df_historico["CTR"] = df_historico["CTR"].round(2)
        df_historico["Tasa_de_conversion"] = df_historico["Tasa_de_conversion"].round(2)
        df_historico["Participacion_Leads"] = df_historico["Participacion_Leads"].round(2)
        df_historico["Participacion_Costo"] = df_historico["Participacion_Costo"].round(2)

        df_historico = df_historico[[
            "AdSetname","Adname","TotalCost","Participacion_Costo",
            "CPA","CTR","Tasa_de_conversion","Participacion_Leads"
        ]]
        st.dataframe(df_historico, use_container_width=True)

    # (D) Resumen Semanal
    st.markdown("<h3 style='color: #1877f2; font-weight: 700;'>Resumen Semanal</h3>", unsafe_allow_html=True)
    from copy import deepcopy
    weekly_group = calcular_metricas_semanales(data)  # USAMOS data COMPLETO
    general_group = calcular_metricas_generales(weekly_group)
    metrics_order = [
        "TotalCost","participación_costo","Tasa_conversion",
        "PTP_total","CPA","CPA_variation","TotalCost_variation"
    ]
    df_filtrado_ads = data.copy()
    if selected_adset != "General":
        st.markdown("<h4 style='color: #1877f2;'>Selecciona los Anuncios a Incluir</h4>", unsafe_allow_html=True)
        anuncios_en_conjunto = df_filtrado_ads[df_filtrado_ads["AdSetname"]==selected_adset]["Adname"].unique().tolist()

        anuncios_seleccionados = []
        fecha_maxima = df_filtrado_ads["Date"].max()
        fecha_inicio_semana = fecha_maxima - pd.Timedelta(days=6)
        df_ultima_semana = df_filtrado_ads[
            (df_filtrado_ads["Date"]>=fecha_inicio_semana) &
            (df_filtrado_ads["Date"]<=fecha_maxima)
        ].copy()
        df_cpa_anuncio = df_ultima_semana.groupby("Adname", as_index=False).agg({
            "TotalCost":"sum","PTP_total":"sum"
        })
        df_cpa_anuncio["CPA"] = df_cpa_anuncio.apply(
            lambda row: row["TotalCost"]/row["PTP_total"] if row["PTP_total"]>0 else 0,
            axis=1
        )
        cpa_dict = dict(zip(df_cpa_anuncio["Adname"], df_cpa_anuncio["CPA"]))

        for anuncio in anuncios_en_conjunto:
            cpa_val = cpa_dict.get(anuncio, 0)
            label_text = f"{anuncio} [CPA: ${cpa_val:.2f}]"
            is_checked = st.checkbox(label_text, value=True)
            if is_checked:
                anuncios_seleccionados.append(anuncio)

        df_filtrado_ads = df_filtrado_ads[df_filtrado_ads["Adname"].isin(anuncios_seleccionados)].copy()
        weekly_group_filtered = calcular_metricas_semanales(df_filtrado_ads)
        pivot_table = generar_tabla_pivot(weekly_group_filtered, selected_adset, metrics_order)
        styled_table = estilizar_tabla(pivot_table)
        st.dataframe(styled_table, use_container_width=True)

    else:
        pivot_table = generar_tabla_pivot(weekly_group, selected_adset, metrics_order, general_df=general_group)
        styled_table = estilizar_tabla(pivot_table)
        st.dataframe(styled_table, use_container_width=True)

    # GESTIÓN DE HITOS
    file_hitos = "hitos.csv"
    df_hitos = load_hitos_csv(file_path=file_hitos)
    if "hitos" not in st.session_state:
        st.session_state["hitos"] = []
    else:
        st.session_state["hitos"].clear()

    # Copiamos cada fila
    for i, row in df_hitos.iterrows():
        st.session_state["hitos"].append({
            "Date": row["Date"],
            "descripcion": row["Description"]
        })

    st.markdown("<h4 style='color: #1877f2;'>Hitos Registrados</h4>", unsafe_allow_html=True)
    if df_hitos.empty:
        st.info("No hay hitos registrados aún.")

    desc_hito = st.text_input("Descripción del hito")
    fecha_hito = st.date_input("Fecha del hito")
    agregar_hito = st.button("Agregar hito")
    mostrar_hitos = st.checkbox("Mostrar Hitos Registrados")

    if mostrar_hitos:
        st.markdown("<h4 style='color: #1877f2;'>Agregar/Eliminar Hitos</h4>", unsafe_allow_html=True)
        if df_hitos.empty:
            st.info("No hay hitos registrados aún.")
        else:
            for i, row in df_hitos.iterrows():
                col1, col2 = st.columns([4,1])
                with col1:
                    st.write(f"- **{row['Date'].strftime('%Y-%m-%d')}**: {row['Description']}")
                with col2:
                    if st.button("Eliminar", key=f"btn_eliminar_{i}"):
                        df_hitos.drop(index=i, inplace=True)
                        df_hitos.reset_index(drop=True, inplace=True)
                        df_save = df_hitos.copy()
                        df_save["Date"] = pd.to_datetime(df_save["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
                        save_hitos_csv(df_save, file_hitos)
                        st.session_state["hitos"] = [
                            h for h in st.session_state["hitos"]
                            if not (
                                h["descripcion"] == row["Description"] and
                                pd.to_datetime(h["Date"]) == pd.to_datetime(row["Date"])
                            )
                        ]
                        st.success("Hito eliminado correctamente.")

    if agregar_hito:
        if desc_hito.strip():
            new_row = {"Date": fecha_hito, "Description": desc_hito.strip()}
            df_hitos = pd.concat([df_hitos, pd.DataFrame([new_row])], ignore_index=True)
            df_hitos.reset_index(drop=True, inplace=True)
            df_save = df_hitos.copy()
            df_save["Date"] = pd.to_datetime(df_save["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
            save_hitos_csv(df_save, file_hitos)
            st.session_state["hitos"].append({
                "Date": fecha_hito,
                "descripcion": desc_hito.strip()
            })
            st.success("Hito agregado exitosamente")
        else:
            st.warning("La descripción del hito no puede estar vacía.")

    # (E) GRÁFICOS FILTRADOS
    st.markdown("<h3 style='color: #1877f2; font-weight: 700;'>Gráficos (Filtrados por Rango de Fechas y Conjunto)</h3>", unsafe_allow_html=True)
    st.write("---")
    st.markdown("<h4 style='color: #1877f2;'>Mostrar Gráficos</h4>", unsafe_allow_html=True)

    if st.checkbox("Mostrar gráfico de Evolución del CPA Diario por Conjunto"):
        fig_cpa_adset = generar_grafico_cpa_diario_adset(df_segmentado, selected_adset)
        st.plotly_chart(fig_cpa_adset, use_container_width=True)

    if st.checkbox("Mostrar gráfico de Evolución del PTP Diario por Conjunto"):
        fig_ptp = generar_grafico_ptp_diario(df_segmentado, selected_adset)
        st.plotly_chart(fig_ptp, use_container_width=True)

    if st.checkbox("Mostrar gráfico de Evolución del CPA Diario (Anuncios)"):
        fig_cpa = generar_grafico_cpa_diario(df_segmentado, selected_adset)
        st.plotly_chart(fig_cpa, use_container_width=True)

    if st.checkbox("Mostrar gráfico de Evolución del PTP_total Diario por Anuncio"):
        fig_ptp_ads = generar_grafico_ptp_diario_ads(df_segmentado, selected_adset)
        st.plotly_chart(fig_ptp_ads, use_container_width=True)

    if st.checkbox("Mostrar gráfico de Evolución de la Tasa de Conversión Diario por Conjunto"):
        fig_tc_adset = generar_grafico_tc_diario_adset(df_segmentado, selected_adset)
        st.plotly_chart(fig_tc_adset, use_container_width=True)

    if st.checkbox("Mostrar gráfico de Evolución de la Tasa de Conversión Diario por Anuncio"):
        fig_tc_ads = generar_grafico_tc_diario_ads(df_segmentado, selected_adset)
        st.plotly_chart(fig_tc_ads, use_container_width=True)

    # (F) COMPARACIÓN DE LANDINGS
    st.markdown("<h3 style='color: #1877f2; font-weight: 700;'>Comparación de Landings</h3> <p>(Filtrado por Fechas y Conjunto)</p>", unsafe_allow_html=True)

    # Usamos df_segmentado
    df_filtered = df_segmentado.copy()
    df_agg = df_filtered.groupby("Linkurl", as_index=False).agg({
        "Landingpageviews":"sum",
        "PTP_total":"sum",
        "Impressions":"sum"
    })
    # NUEVA columna Tasa_de_Conversion => PTP_total / Landingpageviews
    df_agg["Tasa_de_Conversion"] = df_agg.apply(
        lambda row: (row["PTP_total"]/row["Landingpageviews"]) if row["Landingpageviews"]>0 else 0,
        axis=1
    )

    # df_table para "Adcount" y "Adname"
    df_table = df_filtered.groupby("Linkurl", as_index=False).agg({
        "Adname": lambda x: list(x.unique())
    })
    df_table["Adcount"] = df_table["Adname"].apply(len)
    df_table["Adname"] = df_table["Adname"].apply(lambda ads: ", ".join(ads))

    # Mezclamos df_agg con df_table para obtener las columnas en 1 solo df
    df_merge = pd.merge(
        df_agg, df_table, on="Linkurl", how="left"
    )

    pd.set_option('display.max_colwidth', None)

    show_landing = st.checkbox("Mostrar gráfico y tabla de Landings", value=False)
    if show_landing:
        df_sorted = df_merge.sort_values("Tasa_de_Conversion", ascending=False)
        # TABLA debajo del gráfico, con las columnas solicitadas
        st.write("## Landings y sus Anuncios")
        # Preparamos el DataFrame final
        df_final_table = df_sorted[[
            "Linkurl", 
            "Impressions",
            "Landingpageviews",
            "PTP_total",
            "Tasa_de_Conversion",
            "Adcount",
            "Adname"
        ]].copy()

        # Formateamos un poco
        df_final_table["Tasa_de_Conversion"] = df_final_table["Tasa_de_Conversion"].round(4)
        st.dataframe(df_final_table, use_container_width=True)

elif selected_page == "Engagement":
    # =========================================================================================
    # TÍTULO DE LA PÁGINA
    # =========================================================================================
    st.title("Análisis de Engagement")

    # =========================================================================================
    # 2) FILTROS PRINCIPALES (EN SIDEBAR)
    # =========================================================================================
    with st.sidebar:
        st.markdown("<h3 style='color: #ffffff; font-weight: 700;'>Filtros de Datos</h3>", unsafe_allow_html=True)

        # 2.1) Rango de Fechas
        start_date = st.date_input("Fecha de inicio", value=data["Date"].min())
        end_date = st.date_input("Fecha de fin", value=data["Date"].max())

        # 2.2) Campaignname
        campaign_list = sorted(data["Campaignname"].dropna().unique())
        selected_campaign = st.selectbox("Selecciona la Campaña", ["Todas"] + campaign_list)

        # 2.3) Conjunto de Anuncios (AdSetname)
        adset_list = sorted(data["AdSetname"].dropna().unique())
        selected_adset = st.selectbox("Selecciona el Conjunto de Anuncios", ["Todos"] + adset_list)

    # =========================================================================================
    # APLICAR FILTROS PRINCIPALES
    # =========================================================================================
    filtered_data = data.copy()

    # Filtrar por rango de fechas
    filtered_data = filtered_data[
        (filtered_data["Date"] >= pd.to_datetime(start_date)) &
        (filtered_data["Date"] <= pd.to_datetime(end_date))
    ]

    # Filtrar por Campaign
    if selected_campaign != "Todas":
        filtered_data = filtered_data[filtered_data["Campaignname"] == selected_campaign]

    # Filtrar por AdSet
    if selected_adset != "Todos":
        filtered_data = filtered_data[filtered_data["AdSetname"] == selected_adset]

    # =========================================================================================
    # 3) FILTRO DE Link URL (dependiente de lo ya filtrado)
    # =========================================================================================
    link_list = sorted(filtered_data["Linkurl"].dropna().unique())
    selected_links = st.sidebar.multiselect("Selecciona Link URL", link_list, default=link_list)

    # Aplicar filtro de LinkURL
    filtered_data = filtered_data[filtered_data["Linkurl"].isin(selected_links)]

    # =========================================================================================
    # 4) MULTISELECT de Anuncios (también dependiente)
    # =========================================================================================
    adname_list = sorted(filtered_data["Adname"].dropna().unique())
    selected_adnames = st.sidebar.multiselect("Selecciona los Anuncios", adname_list, default=adname_list)

    # Aplicar filtro de Anuncios
    filtered_data = filtered_data[filtered_data["Adname"].isin(selected_adnames)]

    # =========================================================================================
    # 5) GRÁFICO ORIGINAL: AWT_Trend vs CTR_Trend
    # =========================================================================================
    st.markdown("<h3 style='color: #1877f2; font-weight: 700;'>Gráfico de Engagement con Tendencias</h3>", unsafe_allow_html=True)

    if not filtered_data.empty:
        # 5.1) Construir daily_engagement con las columnas relevantes
        daily_engagement = filtered_data.groupby("Date", as_index=False).agg({
            "Videoaveragewatchtime": "mean",
            "Clicks": "sum",
            "Impressions": "sum"
        })

        # CTR y orden
        daily_engagement["CTR"] = (daily_engagement["Clicks"] / daily_engagement["Impressions"]) * 100
        daily_engagement = daily_engagement.fillna(0).sort_values("Date")

        # Suavizado CTR
        daily_engagement["CTR_Smoothed"] = daily_engagement["CTR"].rolling(window=3, min_periods=1).mean()

        # Tendencias (regresión lineal) para AWT y CTR
        import numpy as np
        x = np.arange(len(daily_engagement))

        slope_awt, intercept_awt = np.polyfit(x, daily_engagement["Videoaveragewatchtime"], 1)
        daily_engagement["AWT_Trend"] = slope_awt * x + intercept_awt

        slope_ctr, intercept_ctr = np.polyfit(x, daily_engagement["CTR_Smoothed"], 1)
        daily_engagement["CTR_Trend"] = slope_ctr * x + intercept_ctr

        # 5.2) GRAFICO con Plotly Express
        import plotly.express as px
        fig = px.line(
            daily_engagement,
            x="Date",
            y=["Videoaveragewatchtime", "AWT_Trend"],
            title="Engagement Diario con Tendencias",
            labels={"value": "Valores", "variable": "Métrica"},
            markers=True,
        )
        # Ajustar la traza de AW_TTrend
        fig.update_traces(
            line=dict(color="red", dash="solid"),
            marker=dict(symbol="circle", size=8),
            selector=dict(name="AWT_Trend"),
        )
        # Agregar la CTR trend en eje Y2
        fig.add_scatter(
            x=daily_engagement["Date"],
            y=daily_engagement["CTR_Trend"],
            mode="lines+markers",
            name="CTR Tendencia",
            line=dict(color="darkcyan", dash="solid"),
            marker=dict(symbol="circle", size=8),
            yaxis="y2",
        )
        fig.update_layout(
            yaxis=dict(title="Tiempo Promedio (segundos)", range=[0, 6]),
            yaxis2=dict(title="CTR (%)", overlaying="y", side="right", range=[0.5, 0.8]),
            xaxis_title="Fecha",
            legend_title="Métricas",
            height=600,
            width=900,
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Tendencia AWT**: y = {slope_awt:.2f}x + {intercept_awt:.2f}")
        with col2:
            st.markdown(f"**Tendencia CTR**: y = {slope_ctr:.4f}x + {intercept_ctr:.4f}")

        # =====================================================================================
        # 6) NUEVO GRÁFICO: Landingpageviews, PTP_total, ratio (PTP_total / Landingpageviews)
        # =====================================================================================
        daily_data2 = filtered_data.groupby("Date", as_index=False).agg({
            "Landingpageviews": "sum",
            "PTP_total": "sum"
        }).sort_values("Date")

        # ratio => en porcentaje
        daily_data2["ratio_pct"] = daily_data2.apply(
            lambda row: (row["PTP_total"] / row["Landingpageviews"]) * 100 if row["Landingpageviews"]>0 else 0,
            axis=1
        )

        import plotly.graph_objects as go
        fig2 = go.Figure()

        # BARRAS => Landingpageviews
        fig2.add_trace(
            go.Bar(
                x=daily_data2["Date"],
                y=daily_data2["Landingpageviews"],
                name="Landingpageviews",
                marker_color="cornflowerblue"
            )
        )

        # LÍNEA => PTP_total (Eje Y2)
        fig2.add_trace(
            go.Scatter(
                x=daily_data2["Date"],
                y=daily_data2["PTP_total"],
                name="PTP_total",
                mode="lines+markers",
                line=dict(color="firebrick", width=3),
                yaxis="y2"
            )
        )

        # LÍNEA => ratio_pct (también en Eje Y2, estilo dash)
        fig2.add_trace(
            go.Scatter(
                x=daily_data2["Date"],
                y=daily_data2["ratio_pct"],
                name="Ratio (%)",
                mode="lines+markers",
                line=dict(color="green", width=2, dash="dash"),
                yaxis="y2"
            )
        )

        fig2.update_layout(
            title="Conversión de Landings",
            xaxis=dict(title="Fecha"),
            yaxis=dict(title="Landingpageviews"),
            yaxis2=dict(
                title="PTP_total / Ratio (%)",
                overlaying="y",
                side="right"
            ),
            legend_title="Métricas",
            hovermode="x unified",
            width=900,
            height=600
        )
        st.plotly_chart(fig2, use_container_width=True)

        # =====================================================================================
        # 7) TABLA DEPENDIENTE SÓLO DE FECHA Y CONJUNTO
        #     (AdSetname, Adname, Impressions, Reach, Landingpageviews, ...
        #      Postshares, Postsaves, Frequency, PTP_total, CPA, CPM, CPV, CTR_c, RatioConversion)
        #  +  última fila de "TOTALES"
        # =====================================================================================
        st.markdown("## Resumen de Métricas (sólo filtrado por Fecha y Conjunto)")

        # 7.1) Crear df_table con la lógica pedida:
        df_table = data.copy()

        # Filtrar sólo por fecha y adset (ignorando campaign, link, adname)
        df_table = df_table[
            (df_table["Date"] >= pd.to_datetime(start_date)) &
            (df_table["Date"] <= pd.to_datetime(end_date))
        ]
        if selected_adset != "Todos":
            df_table = df_table[df_table["AdSetname"] == selected_adset]

        # 7.2) Agrupar por AdSet, Adname
        df_agg = df_table.groupby(["AdSetname","Adname"], as_index=False).agg({
            "Impressions":"sum",
            "Reach":"sum",
            "Landingpageviews":"sum",
            "TotalCost":"sum",
            "Postshares":"sum",
            "Postsaves":"sum",
            "Frequency":"mean",  # Asumes que Frequency promedia
            "PTP_total":"sum"
        })

        # 7.3) Calcular columnas derivadas
        df_agg["CPA"] = df_agg.apply(
            lambda row: row["TotalCost"]/row["PTP_total"] if row["PTP_total"]>0 else 0,
            axis=1
        )
        # CPM = (TotalCost / Impressions)*1000
        df_agg["CPM"] = df_agg.apply(
            lambda row: (row["TotalCost"]/row["Impressions"]*1000) if row["Impressions"]>0 else 0,
            axis=1
        )
        # CPV = (TotalCost / Landingpageviews)
        df_agg["CPV"] = df_agg.apply(
            lambda row: row["TotalCost"]/row["Landingpageviews"] if row["Landingpageviews"]>0 else 0,
            axis=1
        )
        # CTR_c = (Landingpageviews / Impressions)*100
        df_agg["CTR_c"] = df_agg.apply(
            lambda row: (row["Landingpageviews"]/row["Impressions"]*100) if row["Impressions"]>0 else 0,
            axis=1
        )
        # RatioConversion = PTP_total / Landingpageviews
        df_agg["RatioConversion"] = df_agg.apply(
            lambda row: row["PTP_total"]/row["Landingpageviews"] if row["Landingpageviews"]>0 else 0,
            axis=1
        )

        # 7.4) Ordenar columnas
        df_agg = df_agg[[
            "AdSetname","Adname","Impressions","Reach","Landingpageviews",
            "TotalCost","Postshares","Postsaves","Frequency","PTP_total",
            "CPA","CPM","CPV","CTR_c","RatioConversion"
        ]]

        # 7.5) Crear Fila de "TOTALES"
        # Se suman las columnas que tengan sentido (Impressions, etc.)
        # Frequency se puede promediar (o sumarse?). Aquí lo haremos promediar, 
        # pero con un weighting. Te muestro un ejemplo simple, asumiendo sum:
        # Haremos un approach práctico:
        # - Sumas para la mayoría
        # - Para 'Frequency' => se hace un promedio global (weighted?), 
        #   por simplicidad, asumamos un average normal.
        sum_impressions = df_agg["Impressions"].sum()
        sum_reach = df_agg["Reach"].sum()
        sum_lpviews = df_agg["Landingpageviews"].sum()
        sum_cost = df_agg["TotalCost"].sum()
        sum_postshares = df_agg["Postshares"].sum()
        sum_postsaves = df_agg["Postsaves"].sum()
        sum_ptp = df_agg["PTP_total"].sum()
        # Frequency -> un average normal
        avg_freq = df_agg["Frequency"].mean()

        # Derivados totales
        tot_cpa = (sum_cost/sum_ptp) if sum_ptp>0 else 0
        tot_cpm = (sum_cost/sum_impressions*1000) if sum_impressions>0 else 0
        tot_cpv = (sum_cost/sum_lpviews) if sum_lpviews>0 else 0
        tot_ctr_c = (sum_lpviews/sum_impressions*100) if sum_impressions>0 else 0
        tot_ratio = (sum_ptp/sum_lpviews) if sum_lpviews>0 else 0

        # 7.6) Crear la fila TOT
        df_tot = {
            "AdSetname": "TOTAL",
            "Adname": "",
            "Impressions": sum_impressions,
            "Reach": sum_reach,
            "Landingpageviews": sum_lpviews,
            "TotalCost": sum_cost,
            "Postshares": sum_postshares,
            "Postsaves": sum_postsaves,
            "Frequency": avg_freq,
            "PTP_total": sum_ptp,
            "CPA": tot_cpa,
            "CPM": tot_cpm,
            "CPV": tot_cpv,
            "CTR_c": tot_ctr_c,
            "RatioConversion": tot_ratio
        }

        # 7.7) Convertir en DF y agregar al final
        import pandas as pd
        df_tot_row = pd.DataFrame([df_tot])
        df_agg_final = pd.concat([df_agg, df_tot_row], ignore_index=True)

        # 7.8) Mostrar la tabla
        st.dataframe(df_agg_final, use_container_width=True)

    else:
        st.warning("No hay datos para mostrar con los filtros seleccionados.")


elif selected_page == "Personalizado":
    st.title("Generador de Gráficas Personalizadas")

    # SIDEBAR
    with st.sidebar:
        st.markdown("## Configuraciones de Gráfico")

        # Eje X
        st.markdown("### Eje X")
        all_columns = [col for col in data.columns if col not in ("Date")]
        eje_x = st.selectbox("Selecciona la columna para el Eje X", options=["Date"] + all_columns, index=0)

        # Eje Y
        st.markdown("### Eje Y")
        eje_y = st.selectbox("Columna para Eje Y", options=all_columns, index=0)
        tipo_y = st.selectbox("Tipo de gráfica (Eje Y)", ["Línea", "Barra"], index=0)
        color_y = st.color_picker("Color Eje Y", "#FF0000")
        alpha_y = st.slider("Transparencia Eje Y", 0.0, 1.0, 0.8)

        # Eje Y2
        st.markdown("### Eje Y Secundario")
        eje_y2 = st.selectbox("Columna para Eje Y Secundario (opcional)", options=["(Ninguno)"] + all_columns, index=0)
        tipo_y2 = st.selectbox("Tipo de gráfica (Eje Y2)", ["Línea", "Barra"], index=0)
        color_y2 = st.color_picker("Color Eje Y2", "#0000FF")
        alpha_y2 = st.slider("Transparencia Eje Y2", 0.0, 1.0, 0.8)

        # Dimensión temporal
        st.markdown("### Dimensión Temporal")
        temporalidad = st.selectbox("Seleccione la agrupación de Fechas", ["Día", "Semana", "Mes"], index=0)
        fecha_inicio = st.date_input("Inicio de periodo", value=data["Date"].min())
        fecha_fin = st.date_input("Fin de periodo", value=data["Date"].max())

        generar = st.button("Generar Gráfico")

    if generar:
        # Filtro
        df_filtered = data.copy()
        df_filtered["Date"] = pd.to_datetime(df_filtered["Date"], errors="coerce")
        df_filtered = df_filtered[
            (df_filtered["Date"] >= pd.to_datetime(fecha_inicio)) &
            (df_filtered["Date"] <= pd.to_datetime(fecha_fin))
        ]
        # Agrupación
        if temporalidad == "Día":
            df_filtered["Periodo"] = df_filtered["Date"].dt.date
        elif temporalidad == "Semana":
            df_filtered["Periodo"] = df_filtered["Date"].dt.to_period("W").apply(lambda r: r.start_time.date())
        else:
            df_filtered["Periodo"] = df_filtered["Date"].dt.to_period("M").apply(lambda r: r.start_time.date())

        numeric_cols = [c for c in df_filtered.columns if pd.api.types.is_numeric_dtype(df_filtered[c])]
        df_group = df_filtered.groupby("Periodo", as_index=False).agg({c: "sum" for c in numeric_cols})
        df_group["Periodo"] = pd.to_datetime(df_group["Periodo"], errors="coerce")

        import plotly.graph_objects as go
        fig = go.Figure()
        modo_y = "lines" if (tipo_y == "Línea") else "bar"
        if eje_y and eje_y in df_group.columns:
            if modo_y == "lines":
                fig.add_trace(
                    go.Scatter(
                        x=df_group["Periodo"], 
                        y=df_group[eje_y], 
                        name=eje_y,
                        mode="lines",
                        line=dict(color=color_y),
                        opacity=alpha_y
                    )
                )
            else:
                fig.add_trace(
                    go.Bar(
                        x=df_group["Periodo"], 
                        y=df_group[eje_y], 
                        name=eje_y,
                        marker=dict(color=color_y, opacity=alpha_y)
                    )
                )

        if eje_y2 != "(Ninguno)" and eje_y2 in df_group.columns:
            modo_y2 = "lines" if (tipo_y2 == "Línea") else "bar"
            if modo_y2 == "lines":
                fig.add_trace(
                    go.Scatter(
                        x=df_group["Periodo"],
                        y=df_group[eje_y2],
                        name=eje_y2,
                        mode="lines",
                        line=dict(color=color_y2),
                        opacity=alpha_y2,
                        yaxis="y2"
                    )
                )
            else:
                fig.add_trace(
                    go.Bar(
                        x=df_group["Periodo"],
                        y=df_group[eje_y2],
                        name=eje_y2,
                        marker=dict(color=color_y2, opacity=alpha_y2),
                        yaxis="y2"
                    )
                )
            fig.update_layout(
                yaxis2=dict(
                    overlaying="y",
                    side="right"
                )
            )

        fig.update_layout(
            xaxis_title=f"Periodo ({temporalidad})",
            yaxis_title=eje_y if eje_y else "",
            legend_title="Métricas",
            title="Gráfico Personalizado",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Descargar JPG
        import io, base64
        import kaleido  # Asegúrate de tener instalado 'pip install kaleido'
        buffer = io.BytesIO()
        fig.write_image(buffer, format="jpg")
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode()
        href = f'<a href="data:file/jpg;base64,{b64}" download="grafico.jpg">Descargar JPG</a>'
        st.markdown(href, unsafe_allow_html=True)
