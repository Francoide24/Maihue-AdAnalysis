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
from collections import defaultdict
import plotly.graph_objects as go
import calendar

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

DEFAULT_MODEL_NAME = "gemini-2.0-flash-exp"

# ===============================
# 4. CARGA DE DATOS
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("df_activos.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

data = load_data()

df_esp = data

# ===============================
# CONFIGURACIÓN DE NAVEGACIÓN ENTRE PÁGINAS
# ===============================
st.sidebar.title("Navegación")
selected_page = st.sidebar.radio(
    "Selecciona una página:",
    ["Conversión","Ranking de Anuncios","Tendencias", "Engagement","Personalizado"] 
)

if selected_page == "Conversión":

    # ===============================
    # 5. ANÁLISIS AUTOMATIZADO POR IA
    # ===============================
    def generar_resumen_ultimas_2_semanas(df):
        """
        - Orden de secciones:
            1) Resumen General
            2) Observaciones y Estrategia
            3) Análisis de Cambios de Presupuesto (últimos 20 días)
            4) Anuncios Nuevos y Apagados (últimos 20 días) -> Se retorna separado en apagados_html
        - Devuelve 2 strings:
            1) resumen
            2) apagados_html
        - El bloque apagados_html se mostrará/ocultará mediante un checkbox externo.
        """

        resumen = ""
        apagados_html = ""

        if df.empty:
            return "No hay datos disponibles.", ""

        try:
            fecha_actual = df["Date"].max()
            if pd.isnull(fecha_actual):
                return "No hay fechas válidas.", ""

            # -------------------------------------------------
            # Períodos clave
            # -------------------------------------------------
            fecha_inicio_2sem = fecha_actual - pd.Timedelta(days=14)
            df_ult_2sem = df[df["Date"] >= fecha_inicio_2sem].copy()

            fecha_inicio_4sem = fecha_inicio_2sem - pd.Timedelta(days=14)
            df_2sem_ant = df[(df["Date"] >= fecha_inicio_4sem) & (df["Date"] < fecha_inicio_2sem)].copy()

            # Fechas para leads mes y semana
            inicio_mes = fecha_actual.replace(day=1)
            df_mes_actual = df[df["Date"] >= inicio_mes].copy()

            fecha_semana = fecha_actual - pd.Timedelta(days=6)
            df_ult_semana = df[df["Date"] >= fecha_semana].copy()

            # -------------------------------------------------
            # 1) RESUMEN GENERAL
            # -------------------------------------------------
            resumen += "\n##### Resumen General\n"
            leads_mes = df_mes_actual["PTP_total"].sum()
            leads_semana = df_ult_semana["PTP_total"].sum()
            pct_semana = (leads_semana / leads_mes * 100) if leads_mes > 0 else 0

            resumen += (f"- Total leads del mes: **{leads_mes:.0f}**.\n"
                        f"- Última semana: **{leads_semana:.0f}** leads ({pct_semana:.2f}% del total).\n")

            cpa_2n, cpa_2n_ant = 0, 0
            if df_ult_2sem["PTP_total"].sum() > 0:
                cpa_2n = df_ult_2sem["TotalCost"].sum() / df_ult_2sem["PTP_total"].sum()
            if df_2sem_ant["PTP_total"].sum() > 0:
                cpa_2n_ant = df_2sem_ant["TotalCost"].sum() / df_2sem_ant["PTP_total"].sum()

            var_cpa_global = ((cpa_2n - cpa_2n_ant) / cpa_2n_ant * 100) if cpa_2n_ant > 0 else 0
            diff_dollar = cpa_2n - cpa_2n_ant

            if var_cpa_global > 0:
                difference_str = f"${abs(diff_dollar):,.2f} más alto ({abs(var_cpa_global):,.2f}%)"
            elif var_cpa_global < 0:
                difference_str = f"${abs(diff_dollar):,.2f} más bajo ({abs(var_cpa_global):,.2f}%)"
            else:
                difference_str = "igual"

            resumen += (f"\n- CPA promedio (últ. 2 sem): ${cpa_2n:,.2f}; "
                        f"{difference_str} vs 2 sem anteriores.\n")

            # -------------------------------------------------
            # 2) OBSERVACIONES Y ESTRATEGIA
            # -------------------------------------------------
            df_ok = df_ult_2sem[
                (df_ult_2sem["PTP_total"] > 0) &
                (df_ult_2sem["TotalCost"] > 0) &
                (df_ult_2sem["Impressions"] > 0)
            ].copy()

            tendencias_cpa = []
            if not df_ok.empty:
                for adname, grupo in df_ok.groupby("Adname"):
                    grupo = grupo.sort_values("Date")
                    if len(grupo) < 2:
                        continue
                    x = np.arange(len(grupo)).reshape(-1, 1)
                    y = grupo.apply(lambda row: row["TotalCost"] / row["PTP_total"], axis=1).values.reshape(-1, 1)
                    model = LinearRegression().fit(x, y)
                    pen = model.coef_[0][0]
                    tendencias_cpa.append((adname, pen))
                tendencias_cpa.sort(key=lambda x: x[1], reverse=True)

            resumen += "\n##### Observaciones y Estrategia\n"
            if tendencias_cpa:
                resumen += "Observa estos anuncios con mayor tendencia al alza en CPA:\n"
                for ad, pen in tendencias_cpa[:5]:
                    resumen += f"- `{ad}` con pendiente +{pen:.2f}.\n"
            else:
                resumen += "No hubo suficientes datos para tendencias de CPA.\n"

            # -------------------------------------------------
            # 3) ANÁLISIS DE CAMBIOS DE PRESUPUESTO (últimos 20 días)
            # -------------------------------------------------
            resumen += "\n##### Análisis de Cambios de Presupuesto (últimos 20 días)\n"
            fecha_inicio_20dias = fecha_actual - pd.Timedelta(days=20)
            df_20 = df[df["Date"] >= fecha_inicio_20dias].copy()

            df_presup = (
                df_20.groupby("Date", as_index=False)
                .agg(Dailybudget=("Dailybudget", "mean"))
                .sort_values("Date")
            )
            df_presup["Delta_Presupuesto"] = df_presup["Dailybudget"].diff()

            df_cambios = df_presup[
                (df_presup["Delta_Presupuesto"].notnull()) &
                (df_presup["Delta_Presupuesto"] != 0)
            ].copy()

            if df_cambios.empty:
                resumen += "\nNo hubo cambios de presupuesto en los últimos 20 días.\n"
            else:
                for idx in df_cambios.index:
                    dia_cambio = df_cambios.loc[idx, "Date"]
                    delta_pres = df_cambios.loc[idx, "Delta_Presupuesto"]
                    new_val = df_cambios.loc[idx, "Dailybudget"]
                    old_val = 0
                    if idx - 1 in df_presup.index:
                        old_val = df_presup.loc[idx - 1, "Dailybudget"]

                    dia_str = dia_cambio.strftime("%Y-%m-%d")
                    resumen += (f"\n- #### [{dia_str}] Cambio de presupuesto: "
                                f"{old_val:.0f} → {new_val:.0f} (Δ={delta_pres:.2f}).\n")

                    # 7 días pre/post
                    dmin_pre = dia_cambio - pd.Timedelta(days=7)
                    dmax_pre = dia_cambio - pd.Timedelta(days=1)
                    dmin_post = dia_cambio
                    dmax_post = dia_cambio + pd.Timedelta(days=6)

                    df_pre = df[(df["Date"] >= dmin_pre) & (df["Date"] <= dmax_pre)].copy()
                    df_post = df[(df["Date"] >= dmin_post) & (df["Date"] <= dmax_post)].copy()

                    if not df_pre.empty and not df_post.empty:
                        tot_pre = df_pre["TotalCost"].sum()
                        tot_post = df_post["TotalCost"].sum()

                        # ------ Participación ------
                        df_part_pre = (df_pre.groupby("Adname")["TotalCost"].sum() / tot_pre * 100
                                    if tot_pre > 0 else pd.Series(dtype=float))
                        df_part_post = (df_post.groupby("Adname")["TotalCost"].sum() / tot_post * 100
                                        if tot_post > 0 else pd.Series(dtype=float))
                        df_merge = pd.merge(
                            df_part_pre.reset_index().rename(columns={"TotalCost": "Part_pre"}),
                            df_part_post.reset_index().rename(columns={"TotalCost": "Part_post"}),
                            on="Adname", how="outer"
                        ).fillna(0)
                        df_merge["Delta_part"] = df_merge["Part_post"] - df_merge["Part_pre"]

                        resumen += "\nTOP 3 con mayor subida de participación:\n"
                        df_merge.sort_values("Delta_part", ascending=False, inplace=True)
                        subidas = df_merge[df_merge["Delta_part"] > 0].head(3)
                        if subidas.empty:
                            resumen += "   - No hay subidas de participación.\n"
                        else:
                            for _, r2 in subidas.iterrows():
                                resumen += f"   - {r2['Adname']}: +{r2['Delta_part']:.2f} pts.\n"

                        resumen += "\nTOP 3 con mayor bajada de participación:\n"
                        df_merge.sort_values("Delta_part", ascending=True, inplace=True)
                        bajadas = df_merge[df_merge["Delta_part"] < 0].head(3)
                        if bajadas.empty:
                            resumen += "   - No hay bajadas de participación.\n"
                        else:
                            for _, r2 in bajadas.iterrows():
                                resumen += f"   - {r2['Adname']}: {r2['Delta_part']:.2f} pts.\n"

                        # ------ CPA ------
                        df_cpa_pre = df_pre.groupby("Adname").agg({"TotalCost": "sum", "PTP_total": "sum"}).reset_index()
                        df_cpa_pre["CPA_pre"] = df_cpa_pre.apply(
                            lambda row: row["TotalCost"] / row["PTP_total"] if row["PTP_total"] > 0 else 0, axis=1
                        )
                        df_cpa_post = df_post.groupby("Adname").agg({"TotalCost": "sum", "PTP_total": "sum"}).reset_index()
                        df_cpa_post["CPA_post"] = df_cpa_post.apply(
                            lambda row: row["TotalCost"] / row["PTP_total"] if row["PTP_total"] > 0 else 0, axis=1
                        )

                        df_cpa_m = pd.merge(
                            df_cpa_pre[["Adname", "CPA_pre"]],
                            df_cpa_post[["Adname", "CPA_post"]],
                            on="Adname", how="outer"
                        ).fillna(0)
                        df_cpa_m["Delta_cpa"] = df_cpa_m["CPA_post"] - df_cpa_m["CPA_pre"]

                        resumen += "\nTOP 3 mayor subida de CPA:\n"
                        df_cpa_m.sort_values("Delta_cpa", ascending=False, inplace=True)
                        up_data = df_cpa_m[df_cpa_m["Delta_cpa"] > 0].head(3)
                        if up_data.empty:
                            resumen += "   - No se detectó aumento de CPA.\n"
                        else:
                            for _, rowc in up_data.iterrows():
                                resumen += (f"   - {rowc['Adname']}: "
                                            f"{rowc['CPA_pre']:.2f} → {rowc['CPA_post']:.2f} (Δ={rowc['Delta_cpa']:.2f}).\n")

                        resumen += "\nTOP 3 mayor disminución de CPA:\n"
                        df_cpa_m.sort_values("Delta_cpa", ascending=True, inplace=True)
                        down_data = df_cpa_m[df_cpa_m["Delta_cpa"] < 0].head(3)
                        if down_data.empty:
                            resumen += "   - No se detectó disminución de CPA.\n"
                        else:
                            for _, rowc in down_data.iterrows():
                                resumen += (f"   - {rowc['Adname']}: "
                                            f"{rowc['CPA_pre']:.2f} → {rowc['CPA_post']:.2f} (Δ={rowc['Delta_cpa']:.2f}).\n")
                    else:
                        resumen += "  (No hubo datos para comparar 7 días pre/post)\n"

            # -------------------------------------------------
            # 4) ANUNCIOS NUEVOS Y APAGADOS (1 solo) - Se retorna en apagados_html
            # -------------------------------------------------
            try:
                df_apag = df[df["Date"] >= (fecha_actual - pd.Timedelta(days=20))].copy()

                apagados_info = []
                dias_ordenados = sorted(df_apag["Date"].unique())
                for i in range(1, len(dias_ordenados)):
                    dia_ant = dias_ordenados[i - 1]
                    dia_act = dias_ordenados[i]

                    df_dia_ant = df_apag[df_apag["Date"] == dia_ant].groupby("Adname")[["TotalCost", "Impressions"]].sum()
                    df_dia_act = df_apag[df_apag["Date"] == dia_act].groupby("Adname")[["TotalCost", "Impressions"]].sum()

                    df_join = df_dia_ant.join(df_dia_act, lsuffix="_ant", rsuffix="_act", how="inner")

                    df_off = df_join[
                        (df_join["TotalCost_ant"] > 0) &
                        (df_join["Impressions_ant"] > 0) &
                        ((df_join["TotalCost_act"] <= df_join["TotalCost_ant"] * 0.5) &
                        (df_join["Impressions_act"] <= df_join["Impressions_ant"] * 0.5))
                    ]
                    if not df_off.empty:
                        for ad_off in df_off.index:
                            apagados_info.append((ad_off, dia_act))

                if apagados_info:
                    data_off = []
                    unique_off = list(set([x[0] for x in apagados_info]))  # Todos los adnames que se apagaron
                    for adname_off in unique_off:
                        df_an = df_apag[df_apag["Adname"] == adname_off].copy()
                        sum_cost = df_an["TotalCost"].sum()
                        dias_encendido = df_an[df_an["TotalCost"] > 0]["Date"].nunique()
                        daily_prom = sum_cost / dias_encendido if dias_encendido > 0 else 0

                        fechas_apagado = [fecha for (a, fecha) in apagados_info if a == adname_off]
                        fecha_apagado = min(fechas_apagado) if fechas_apagado else None
                        data_off.append((adname_off, fecha_apagado, daily_prom))

                    data_off.sort(key=lambda x: x[2], reverse=True)
                    adname_mayor, fecha_off, _ = data_off[0]

                    if fecha_off is not None:
                        dia_str = pd.to_datetime(fecha_off).strftime("%Y-%m-%d")
                        apagados_html += f"**'{adname_mayor}'** se APAGÓ el {dia_str}.\n\n"
                        apagados_html += "**Impacto (7 días antes vs 7 días después):**\n"

                        dmin_pre = pd.to_datetime(fecha_off) - pd.Timedelta(days=7)
                        dmax_pre = pd.to_datetime(fecha_off) - pd.Timedelta(days=1)
                        dmin_post = pd.to_datetime(fecha_off)
                        dmax_post = pd.to_datetime(fecha_off) + pd.Timedelta(days=6)

                        df_pre = df[(df["Date"] >= dmin_pre) & (df["Date"] <= dmax_pre)].copy()
                        df_post = df[(df["Date"] >= dmin_post) & (df["Date"] <= dmax_post)].copy()

                        if not df_pre.empty and not df_post.empty:
                            tot_pre = df_pre["TotalCost"].sum()
                            tot_post = df_post["TotalCost"].sum()

                            # Participación
                            df_part_pre = (df_pre.groupby("Adname")["TotalCost"].sum() / tot_pre * 100
                                        if tot_pre > 0 else pd.Series(dtype=float))
                            df_part_post = (df_post.groupby("Adname")["TotalCost"].sum() / tot_post * 100
                                            if tot_post > 0 else pd.Series(dtype=float))
                            df_m = pd.merge(
                                df_part_pre.reset_index().rename(columns={"TotalCost": "Part_pre"}),
                                df_part_post.reset_index().rename(columns={"TotalCost": "Part_post"}),
                                on="Adname", how="outer"
                            ).fillna(0)
                            df_m["Delta_part"] = df_m["Part_post"] - df_m["Part_pre"]

                            apagados_html += "\n- **TOP 3 suben participación**:\n"
                            df_m.sort_values("Delta_part", ascending=False, inplace=True)
                            sub_up = df_m[df_m["Delta_part"] > 0].head(3)
                            if sub_up.empty:
                                apagados_html += "   - No hay subidas de participación.\n"
                            else:
                                for _, r2 in sub_up.iterrows():
                                    apagados_html += f"   - {r2['Adname']}: +{r2['Delta_part']:.2f} pts.\n"

                            apagados_html += "\n- **TOP 3 bajan participación**:\n"
                            df_m.sort_values("Delta_part", ascending=True, inplace=True)
                            sub_down = df_m[df_m["Delta_part"] < 0].head(3)
                            if sub_down.empty:
                                apagados_html += "   - No hay bajadas de participación.\n"
                            else:
                                for _, r2 in sub_down.iterrows():
                                    apagados_html += f"   - {r2['Adname']}: {r2['Delta_part']:.2f} pts.\n"

                            # CPA
                            df_cpa_pre = df_pre.groupby("Adname").agg({"TotalCost": "sum", "PTP_total": "sum"}).reset_index()
                            df_cpa_pre["CPA_pre"] = df_cpa_pre.apply(
                                lambda row: row["TotalCost"] / row["PTP_total"] if row["PTP_total"] > 0 else 0, axis=1
                            )
                            df_cpa_post = df_post.groupby("Adname").agg({"TotalCost": "sum", "PTP_total": "sum"}).reset_index()
                            df_cpa_post["CPA_post"] = df_cpa_post.apply(
                                lambda row: row["TotalCost"] / row["PTP_total"] if row["PTP_total"] > 0 else 0, axis=1
                            )
                            df_cpam = pd.merge(
                                df_cpa_pre[["Adname", "CPA_pre"]],
                                df_cpa_post[["Adname", "CPA_post"]],
                                on="Adname", how="outer"
                            ).fillna(0)
                            df_cpam["Delta_cpa"] = df_cpam["CPA_post"] - df_cpam["CPA_pre"]

                            apagados_html += "\n- **TOP 3 mayor subida de CPA**:\n"
                            df_cpam.sort_values("Delta_cpa", ascending=False, inplace=True)
                            cpa_up = df_cpam[df_cpam["Delta_cpa"] > 0].head(3)
                            if cpa_up.empty:
                                apagados_html += "   - No se detectó aumento de CPA.\n"
                            else:
                                for _, rr in cpa_up.iterrows():
                                    apagados_html += (f"   - {rr['Adname']}: "
                                                    f"{rr['CPA_pre']:.2f} → {rr['CPA_post']:.2f} (Δ={rr['Delta_cpa']:.2f}).\n")

                            apagados_html += "\n- **Top 3 mayor bajada de CPA**:\n"
                            df_cpam.sort_values("Delta_cpa", ascending=True, inplace=True)
                            cpa_down = df_cpam[df_cpam["Delta_cpa"] < 0].head(3)
                            if cpa_down.empty:
                                apagados_html += "   - No se detectó disminución de CPA.\n"
                            else:
                                for _, rr in cpa_down.iterrows():
                                    apagados_html += (f"   - {rr['Adname']}: "
                                                    f"{rr['CPA_pre']:.2f} → {rr['CPA_post']:.2f} (Δ={rr['Delta_cpa']:.2f}).\n")
                        else:
                            apagados_html += "\n  (No hubo datos para comparar 7 días pre/post)\n"
                    else:
                        apagados_html += "Se detectó un apagado importante, pero la fecha no fue válida.\n"
                else:
                    apagados_html += "No se detectaron anuncios apagados relevantes en los últimos 20 días.\n"

            except Exception as e:
                resumen = f"Ocurrió un error al generar el resumen: {e}"    

        except Exception as e:
            resumen = f"Ocurrió un error al generar el resumen: {e}"

        # Devolvemos las 2 cadenas por separado
        return resumen, apagados_html
    # ===============================
    # 6. CONSULTA EN LENGUAJE NATURAL
    # ===============================
    def consulta_lenguaje_natural(pregunta, datos):
        """Realiza una consulta a Gemini con el DataFrame como contexto."""
        try:
            datos_csv = data

            prompt = f"""
            Actúas como un analista experto en marketing digital, especializado en meta ads. Los datos relevantes se encuentran en formato CSV y tienen las siguientes columnas:
            {', '.join(datos.columns)}.

            Los datos son los siguientes:

    csv
            {datos_csv}

            - Siempre debes consultar hasta una fecha máxima de los 28 dias anteriores a la fecha actual.
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
    st.title("Análisis General de Anuncios")
    st.markdown("<h3 style='color: #1877f2; font-weight: 700;'>Resumen Automatizado (IA)</h3>", unsafe_allow_html=True)
    resumen = generar_resumen_ultimas_2_semanas(data)
    resumen, apagados_html = generar_resumen_ultimas_2_semanas(data)
    st.markdown(resumen)  # Esto muestra el texto principal
    mostrar_apagados = st.checkbox("Mostrar Anuncios Nuevos y Apagados (últimos 20 días)")
    if mostrar_apagados:
        st.markdown(f"##### Anuncios Nuevos y Apagados (últimos 20 días)\n\n{apagados_html}")


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

    # 1) Un solo checkbox para los 3 gráficos Diarios por Conjunto
    mostrar_conjunto = st.checkbox("Mostrar Gráficos Diarios por Conjunto")
    if mostrar_conjunto:
        fig_cpa_adset = generar_grafico_cpa_diario_adset(df_segmentado, selected_adset)
        st.plotly_chart(fig_cpa_adset, use_container_width=True)

        fig_ptp = generar_grafico_ptp_diario(df_segmentado, selected_adset)
        st.plotly_chart(fig_ptp, use_container_width=True)

        fig_tc_adset = generar_grafico_tc_diario_adset(df_segmentado, selected_adset)
        st.plotly_chart(fig_tc_adset, use_container_width=True)

    # 2) Un solo checkbox para los 3 gráficos Diarios por Anuncio
    mostrar_anuncios = st.checkbox("Mostrar Gráficos Diarios por Anuncio")
    if mostrar_anuncios:
        fig_cpa = generar_grafico_cpa_diario(df_segmentado, selected_adset)
        st.plotly_chart(fig_cpa, use_container_width=True)

        fig_ptp_ads = generar_grafico_ptp_diario_ads(df_segmentado, selected_adset)
        st.plotly_chart(fig_ptp_ads, use_container_width=True)

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
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go

    st.title("Análisis de Engagement")

    # =============================================================================
    # (A) CARGAMOS df_meng ANTES DEL RESTO DE SECCIONES
    #     Para que el Diagnóstico IA pueda usarlo de inmediato.
    # =============================================================================
    df_meng = pd.read_csv("df_meng.csv")

    # =============================================================================
    # (B) DIAGNÓSTICO IA - Ubicado al comienzo (justo después del título)
    # =============================================================================
    with st.expander("Diagnóstico IA según Métricas de Abandono y Retención"):
        st.write("""
        En esta sección, la IA puede evaluar posibles problemas en el Anuncio o en la Landing,
        considerando el abandono en primeros segundos, retención al 100%, tiempo medio de reproducción,
        CTR y Tasa_de_Conversion, etc.  
        """)
        
        def consulta_lenguaje_natural_tendencias(prompt):
            try:
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
                    parts = response.json().get("candidates")[0].get("content").get("parts", [])
                    if parts:
                        return parts[0].get("text","")
                    else:
                        return "No se pudo interpretar la respuesta."
                else:
                    return f"Error: {response.status_code} - {response.text}"
            except Exception as e:
                return f"Error procesando la consulta: {e}"

        diagnosticar = st.button("Analizar con IA (Abandono, Retención, CTR, Conversión)")
        if diagnosticar:
            # Convertimos df_meng a CSV para que la IA vea los datos reales y los diagnósticos sean coherentes.
            df_meng_csv = df_meng.to_csv(index=False)

            prompt_ia = f"""
Actúa como un experto en marketing digital para Meta Ads.
Tienes las métricas detalladas en un DataFrame CSV (df_meng) con columnas como:
- Abandono_0_25 (porcentaje de usuarios que abandonan antes del 25% del video)
- Retention_25, Retention_50, Retention_75, Retention_100
- Videoaveragewatchtime (tiempo medio de reproducción)
- CTR (Click Through Rate)
- Tasa_de_Conversion (Conversion Rate), calculada como leads (PTP_total) / Landingpageviews
- Otras métricas relevantes (Reach, Frequency, etc.)
- Nombre de cada anuncio (Adname).

A continuación tienes el contenido de 'df_meng' en formato CSV:
-------------------------------------
{df_meng_csv}
-------------------------------------

1. Identifica 3 anuncios con posibles problemas centrados en el anuncio mismo (ej: mal gancho, abandono alto en los primeros segundos, etc.).
2. Identifica otros 3 anuncios con problemas potencialmente en la landing (ej: CTR alto pero baja conversión, retención decente pero leads muy bajos, etc.).
3. Identifica los 3 anuncios con mejor retención, considerando todas las métricas.

Ofrece un diagnóstico conciso y profesional, destacando si el problema está más en el anuncio (bajo engagement) o en la landing (fallas en conversión). 
"""

            resp_ia = consulta_lenguaje_natural_tendencias(prompt_ia)
            st.markdown("### Respuesta IA:")
            st.markdown(resp_ia)

            # -------------------------------------------------------------------------
            # SNIPPET EXTRA: Análisis adicional de Hooks y Embudos, si se desea mostrar
            # -------------------------------------------------------------------------

            # 1) TOP HOOKS: Retención al 25% y Tiempo Medio de Reproducción
            st.markdown("## Análisis de Hooks (Retención 25%)")
            df_hooks_25 = df_meng.copy()

            # Ordenamos de mayor a menor retención al 25%
            df_hooks_25.sort_values("Retention_25", ascending=False, inplace=True)

            # Mostramos sólo el Top10
            df_top10_hooks = df_hooks_25.head(10)

            st.markdown("### Top 10 Anuncios con Mejor 'Hook' (Retención al 25%)")
            st.dataframe(
                df_top10_hooks[["Adname","Retention_25","Videoaveragewatchtime","CTR","Tasa_de_Conversion"]],
                use_container_width=True
            )

            # Gráfico de barras con Retention_25
            fig_hooks = px.bar(
                df_top10_hooks,
                x="Adname",
                y="Retention_25",
                color="Adname",
                hover_data=["Videoaveragewatchtime","CTR","Tasa_de_Conversion"],
                title="Top 10 Anuncios (Retención al 25%)"
            )
            fig_hooks.update_layout(
                showlegend=False,
                xaxis_title="Anuncio",
                yaxis_title="Retención 25% (%)",
                xaxis=dict(tickangle=45)
            )
            st.plotly_chart(fig_hooks, use_container_width=True)

            # 2) Gráfico comparativo Retención vs. Tiempo de Reproducción (scatter)
            st.markdown("### Relación entre Retención al 25% y el Tiempo Medio de Reproducción")
            fig_scatter_hooks = px.scatter(
                df_top10_hooks,
                x="Retention_25",
                y="Videoaveragewatchtime",
                color="Adname",
                size="Retention_25",
                hover_data=["CTR","Tasa_de_Conversion"],
                title="Retención 25% vs. Avg. Watch Time"
            )
            fig_scatter_hooks.update_layout(
                xaxis_title="Retención 25% (%)",
                yaxis_title="Tiempo Medio de Reproducción (seg)"
            )
            st.plotly_chart(fig_scatter_hooks, use_container_width=True)

            # 3) (Opcional) Gráfico de Embudo de Retenciones
            st.markdown("### Embudo de Retención por Anuncio")
            adname_list_meng = df_meng["Adname"].unique().tolist()
            selected_ad_for_funnel = st.selectbox("Selecciona un anuncio para ver su embudo de retención", adname_list_meng)

            df_ad_funnel = df_meng[df_meng["Adname"] == selected_ad_for_funnel].copy()
            if not df_ad_funnel.empty:
                ad_row = df_ad_funnel.iloc[0]
                retention_data = [
                    {"stage":"Inicio (0%)", "value":100},
                    {"stage":"Retención 25%", "value": ad_row.get("Retention_25", 0)},
                    {"stage":"Retención 50%", "value": ad_row.get("Retention_50", 0)},
                    {"stage":"Retención 75%", "value": ad_row.get("Retention_75", 0)},
                    {"stage":"Retención 100%", "value": ad_row.get("Retention_100", 0)}
                ]
                funnel_df = pd.DataFrame(retention_data)

                fig_funnel = px.funnel(
                    funnel_df,
                    x="value",
                    y="stage",
                    title=f"Embudo de Retención para {selected_ad_for_funnel}",
                    labels={"value":"% de usuarios", "stage":"Etapa"}
                )
                st.plotly_chart(fig_funnel, use_container_width=True)
            else:
                st.info("No se encontró información de retención para el anuncio seleccionado.")

    # =============================================================================
    # (C) FILTROS PRINCIPALES (SIDEBAR)
    # =============================================================================
    with st.sidebar:
        st.markdown("<h3 style='color: #ffffff; font-weight: 700;'>Filtros de Datos</h3>", unsafe_allow_html=True)

        # 1.1) Rango de Fechas
        start_date = st.date_input("Fecha de inicio", value=data["Date"].min())
        end_date = st.date_input("Fecha de fin", value=data["Date"].max())

        # 1.2) Campaignname
        campaign_list = sorted(data["Campaignname"].dropna().unique())
        selected_campaign = st.selectbox("Selecciona la Campaña", ["Todas"] + campaign_list)

        # 1.3) Conjunto de Anuncios (AdSetname)
        adset_list = sorted(data["AdSetname"].dropna().unique())
        selected_adset = st.selectbox("Selecciona el Conjunto de Anuncios", ["Todos"] + adset_list)

    # =============================================================================
    # (D) APLICAR FILTROS PRINCIPALES
    # =============================================================================
    filtered_data = data.copy()
    filtered_data = filtered_data[
        (filtered_data["Date"] >= pd.to_datetime(start_date)) &
        (filtered_data["Date"] <= pd.to_datetime(end_date))
    ]
    if selected_campaign != "Todas":
        filtered_data = filtered_data[filtered_data["Campaignname"] == selected_campaign]
    if selected_adset != "Todos":
        filtered_data = filtered_data[filtered_data["AdSetname"] == selected_adset]

    # 2.1) Filtro de Link URL
    link_list = sorted(filtered_data["Linkurl"].dropna().unique())
    selected_links = st.sidebar.multiselect("Selecciona Link URL", link_list, default=link_list)
    filtered_data = filtered_data[filtered_data["Linkurl"].isin(selected_links)]

    # 2.2) Filtro de Anuncios
    adname_list = sorted(filtered_data["Adname"].dropna().unique())
    selected_adnames = st.sidebar.multiselect("Selecciona los Anuncios", adname_list, default=adname_list)
    filtered_data = filtered_data[filtered_data["Adname"].isin(selected_adnames)]

    # =============================================================================
    # (E) GRÁFICO AWT vs CTR TENDENCIA
    # =============================================================================
    st.markdown("<h3 style='color: #1877f2; font-weight: 700;'>Gráfico de Engagement con Tendencias</h3>", unsafe_allow_html=True)
    if not filtered_data.empty:
        daily_engagement = filtered_data.groupby("Date", as_index=False).agg({
            "Videoaveragewatchtime": "mean",
            "Clicks": "sum",
            "Impressions": "sum"
        })
        daily_engagement["CTR"] = (daily_engagement["Clicks"] / daily_engagement["Impressions"]) * 100
        daily_engagement = daily_engagement.fillna(0).sort_values("Date")

        # Suavizado CTR
        daily_engagement["CTR_Smoothed"] = daily_engagement["CTR"].rolling(window=3, min_periods=1).mean()

        # Tendencias (lineal) para AWT y CTR
        x = np.arange(len(daily_engagement))
        slope_awt, intercept_awt = np.polyfit(x, daily_engagement["Videoaveragewatchtime"], 1)
        daily_engagement["AWT_Trend"] = slope_awt * x + intercept_awt

        slope_ctr, intercept_ctr = np.polyfit(x, daily_engagement["CTR_Smoothed"], 1)
        daily_engagement["CTR_Trend"] = slope_ctr * x + intercept_ctr

        fig = px.line(
            daily_engagement,
            x="Date",
            y=["Videoaveragewatchtime", "AWT_Trend"],
            title="Engagement Diario con Tendencias",
            labels={"value": "Valores", "variable": "Métrica"},
            markers=True
        )
        # La línea de tendencia AWT en color rojo (dash) y CTR en color darkcyan (y2 axis)
        fig.update_traces(
            line=dict(color="red", dash="solid"),
            marker=dict(symbol="circle", size=8),
            selector=dict(name="AWT_Trend"),
        )
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
            yaxis=dict(title="Tiempo Promedio (segundos)"),
            yaxis2=dict(title="CTR (%)", overlaying="y", side="right"),
            xaxis_title="Fecha",
            legend_title="Métricas",
            height=600,
            width=900
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Tendencia AWT**: y = {slope_awt:.2f}x + {intercept_awt:.2f}")
        with col2:
            st.markdown(f"**Tendencia CTR**: y = {slope_ctr:.4f}x + {intercept_ctr:.4f}")

        # ============================================================================
        # (F) GRÁFICO => Landingpageviews vs. PTP_total (Conversión de Landings)
        # ============================================================================
        daily_data2 = filtered_data.groupby("Date", as_index=False).agg({
            "Landingpageviews": "sum",
            "PTP_total": "sum"
        }).sort_values("Date")
        daily_data2["ratio_pct"] = daily_data2.apply(
            lambda row: (row["PTP_total"] / row["Landingpageviews"])*100 if row["Landingpageviews"]>0 else 0,
            axis=1
        )

        fig2 = go.Figure()
        fig2.add_trace(
            go.Bar(
                x=daily_data2["Date"],
                y=daily_data2["Landingpageviews"],
                name="Landingpageviews",
                marker_color="cornflowerblue"
            )
        )
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
            yaxis2=dict(title="PTP_total / Ratio (%)", overlaying="y", side="right"),
            legend_title="Métricas",
            hovermode="x unified",
            width=900,
            height=600
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ============================================================================
        # (G) TABLA SÓLO FILTRADA POR FECHA Y CONJUNTO (RESUMEN DE MÉTRICAS)
        # ============================================================================
        st.markdown("## Resumen de Métricas (sólo filtrado por Fecha y Conjunto)")
        df_table = data.copy()
        df_table = df_table[
            (df_table["Date"] >= pd.to_datetime(start_date)) & 
            (df_table["Date"] <= pd.to_datetime(end_date))
        ]
        if selected_adset != "Todos":
            df_table = df_table[df_table["AdSetname"] == selected_adset]

        df_agg = df_table.groupby(["AdSetname","Adname"], as_index=False).agg({
            "Impressions":"sum",
            "Reach":"sum",
            "Landingpageviews":"sum",
            "TotalCost":"sum",
            "Postshares":"sum",
            "Postsaves":"sum",
            "Frequency":"mean",
            "PTP_total":"sum"
        })
        df_agg["CPA"] = df_agg.apply(
            lambda row: row["TotalCost"]/row["PTP_total"] if row["PTP_total"]>0 else 0, axis=1
        )
        df_agg["CPM"] = df_agg.apply(
            lambda row: (row["TotalCost"]/row["Impressions"]*1000) if row["Impressions"]>0 else 0, axis=1
        )
        df_agg["CPV"] = df_agg.apply(
            lambda row: row["TotalCost"]/row["Landingpageviews"] if row["Landingpageviews"]>0 else 0, axis=1
        )
        df_agg["CTR_c"] = df_agg.apply(
            lambda row: (row["Landingpageviews"]/row["Impressions"]*100) if row["Impressions"]>0 else 0, axis=1
        )
        df_agg["RatioConversion"] = df_agg.apply(
            lambda row: row["PTP_total"]/row["Landingpageviews"] if row["Landingpageviews"]>0 else 0, axis=1
        )

        df_agg = df_agg[[
            "AdSetname","Adname","Impressions","Reach","Landingpageviews",
            "TotalCost","Postshares","Postsaves","Frequency","PTP_total",
            "CPA","CPM","CPV","CTR_c","RatioConversion"
        ]]

        # Fila Totales
        sum_impressions = df_agg["Impressions"].sum()
        sum_reach = df_agg["Reach"].sum()
        sum_lpviews = df_agg["Landingpageviews"].sum()
        sum_cost = df_agg["TotalCost"].sum()
        sum_postshares = df_agg["Postshares"].sum()
        sum_postsaves = df_agg["Postsaves"].sum()
        sum_ptp = df_agg["PTP_total"].sum()
        avg_freq = df_agg["Frequency"].mean()

        tot_cpa = sum_cost/sum_ptp if sum_ptp>0 else 0
        tot_cpm = (sum_cost/sum_impressions*1000) if sum_impressions>0 else 0
        tot_cpv = (sum_cost/sum_lpviews) if sum_lpviews>0 else 0
        tot_ctr_c = (sum_lpviews/sum_impressions*100) if sum_impressions>0 else 0
        tot_ratio = (sum_ptp/sum_lpviews) if sum_lpviews>0 else 0

        df_tot_row = pd.DataFrame([{
            "AdSetname":"TOTAL",
            "Adname":"",
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
        }])
        df_agg_final = pd.concat([df_agg, df_tot_row], ignore_index=True)
        st.dataframe(df_agg_final, use_container_width=True)

    else:
        st.warning("No hay datos para mostrar con los filtros seleccionados.")

    # ============================================================================
    # (H) ANÁLISIS DE ABANDONO, RETENCIÓN Y TIEMPO REPRODUCCIÓN (df_meng)
    #     - Sección de Top10 por Abandono, Retención al 100%, y Video Time.
    # ============================================================================
    st.markdown("## Análisis de Abandono, Retención y T. Reproducción")

    # 1) Anuncios con mayor abandono los primeros segundos (Top10)
    st.markdown("### 1) Anuncios con mayor abandono en los primeros segundos (Top10)")
    top10_abandono = df_meng.sort_values("Abandono_0_25", ascending=False).head(10)
    st.dataframe(top10_abandono, use_container_width=True)

    # 2) Anuncios con mayor retención (100%) (Top10)
    st.markdown("### 2) Anuncios con mayor retención al 100% (Top10)")
    top10_retencion = df_meng.sort_values("Retention_100", ascending=False).head(10)
    st.dataframe(top10_retencion, use_container_width=True)

    # 3) Anuncios con mayor tiempo medio de reproducción (Top10)
    st.markdown("### 3) Anuncios con mayor tiempo medio de reproducción (Top10)")
    top10_videotime = df_meng.sort_values("Videoaveragewatchtime", ascending=False).head(10)
    st.dataframe(top10_videotime, use_container_width=True)

elif selected_page == "Tendencias":

    # ------------------------------------------------------
    # PÁGINA: TENDENCIAS (Código existente, NO ALTERADO)
    # ------------------------------------------------------
    st.title("Tendencias de Métricas")
    st.markdown("""
    Esta sección muestra la tendencia (regresión lineal) de distintas métricas en períodos 
    Mensual, Bisemanal y Semanal, además de opciones de IA para sugerencias.
    """)

    with st.expander("Sugerencias IA - Engagement y Conversión", expanded=False):
        st.write("""
        A continuación, puedes generar sugerencias específicas para:
        1) **Engagement** (Tiempo Medio de Reproducción y CTR),
        2) **Conversión** (CPA, Leads y Tasa de Conversión).

        Se seleccionarán los 3 anuncios con las tendencias más críticas en cada objetivo, 
        y se brindarán recomendaciones específicas.
        """)

        def consulta_lenguaje_natural_tendencias(prompt):
            try:
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
                    parts = response.json().get("candidates")[0].get("content").get("parts", [])
                    if parts:
                        return parts[0].get("text","")
                    else:
                        return "No se pudo interpretar la respuesta."
                else:
                    return f"Error: {response.status_code} - {response.text}"
            except Exception as e:
                return f"Error procesando la consulta: {e}"

        def calcular_metrica_ponderada(df_segment, metric):
            lpv   = df_segment["Landingpageviews"].sum()
            imp   = df_segment["Impressions"].sum()
            cost  = df_segment["TotalCost"].sum()
            leads = df_segment["PTP_total"].sum()

            if metric == "Videoaveragewatchtime":
                if imp>0:
                    vw_weighted = (df_segment["Videoaveragewatchtime"] * df_segment["Impressions"]).sum()
                    return vw_weighted / imp
                return 0

            if metric == "CTR":
                return (lpv / imp * 100) if imp>0 else 0
            elif metric == "CPA":
                return cost if leads == 0 else (cost / leads)
            elif metric == "Tasa de Conversión":
                return (leads / imp * 100) if imp>0 else 0
            elif metric == "Leads":
                return leads
            else:
                return 0

        def detectar_peores_anuncios_para_metricas(df_all, list_metrics, days=28):
            df_ok = df_all.copy()
            df_ok["Date"] = pd.to_datetime(df_ok["Date"], errors="coerce")
            df_ok.dropna(subset=["Date"], inplace=True)

            last_d = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
            start_d = last_d - pd.Timedelta(days=days)
            df_ok = df_ok[(df_ok["Date"] >= start_d) & (df_ok["Date"] <= last_d)].copy()
            if df_ok.empty:
                return []

            anuncios_data = defaultdict(lambda: {"score":0, "details":{}})

            for ad_i in df_ok["Adname"].unique():
                df_sub = df_ok[df_ok["Adname"] == ad_i].copy()
                if df_sub.empty:
                    continue
                df_sub.set_index("Date", drop=False, inplace=True)

                for metric in list_metrics:
                    daily_list = []
                    for date_g, dfg in df_sub.resample('D'):
                        if not dfg.empty:
                            val = calcular_metrica_ponderada(dfg, metric)
                            daily_list.append((date_g, val))
                    df_g = pd.DataFrame(daily_list, columns=["Periodo","Val"])
                    df_g.dropna(subset=["Periodo"], inplace=True)
                    df_g.sort_values("Periodo", inplace=True)
                    if len(df_g)<2:
                        continue

                    x_vals = np.arange(len(df_g)).reshape(-1,1)
                    y_vals = df_g["Val"].values.reshape(-1,1)
                    model = LinearRegression().fit(x_vals, y_vals)
                    slope = model.coef_[0][0]

                    if metric == "CPA":
                        worst = slope
                    else:
                        worst = -slope

                    anuncios_data[ad_i]["details"][metric] = slope
                    anuncios_data[ad_i]["score"] += worst
            
            result_list = []
            for adname_i, info in anuncios_data.items():
                result_list.append((adname_i, info["score"], info["details"]))
            if not result_list:
                return []
            
            result_list.sort(key=lambda x: x[1], reverse=True)
            return result_list[:3]

        col_eng, col_conv = st.columns(2)

        with col_eng:
            if st.button("Generar Sugerencias IA - Engagement"):
                top_3_eng = detectar_peores_anuncios_para_metricas(data, ["Videoaveragewatchtime","CTR"], days=28)
                if not top_3_eng:
                    st.warning("No se detectó ningún anuncio con problemas de Engagement o no hay datos.")
                else:
                    details_str = ""
                    for adname_i, score_i, detail_dict in top_3_eng:
                        details_str += f"- **{adname_i}** (score={score_i:.2f}):\n"
                        for mt, slp in detail_dict.items():
                            if mt=="CPA":
                                trend_str = "al alza" if slp>0 else "a la baja"
                            else:
                                trend_str = "a la baja" if slp<0 else "al alza"
                            details_str += f"   - {mt}: slope={slp:.4f} => {trend_str}\n"
                    
                    prompt_eng = (
                        "Top 3 anuncios con peor tendencia en Engagement (Videoaveragewatchtime, CTR) "
                        f"en los últimos 28 días:\n\n"
                        f"{details_str}\n\n"
                        "Actúa como experto en marketing digital para Meta Ads. "
                        "Ofrece un resumen y recomendaciones específicas para mejorar el Engagement "
                        "de estos anuncios (tiempo de reproducción, CTR, etc.)."
                    )
                    resp_eng = consulta_lenguaje_natural_tendencias(prompt_eng)
                    st.markdown("**Sugerencias IA - Engagement:**")
                    st.markdown(resp_eng)

        with col_conv:
            if st.button("Generar Sugerencias IA - Conversión"):
                top_3_conv = detectar_peores_anuncios_para_metricas(data, ["CPA","Leads","Tasa de Conversión"], days=28)
                if not top_3_conv:
                    st.warning("No se detectó ningún anuncio con problemas de Conversión o no hay datos.")
                else:
                    details_str = ""
                    for adname_i, score_i, detail_dict in top_3_conv:
                        details_str += f"- **{adname_i}** (score={score_i:.2f}):\n"
                        for mt, slp in detail_dict.items():
                            if mt=="CPA":
                                trend_str = "al alza" if slp>0 else "a la baja"
                            else:
                                trend_str = "a la baja" if slp<0 else "al alza"
                            details_str += f"   - {mt}: slope={slp:.4f} => {trend_str}\n"
                    prompt_conv = (
                        "Top 3 anuncios con peor tendencia en Conversión (CPA, Leads, Tasa de Conversión) "
                        f"en los últimos 28 días:\n\n"
                        f"{details_str}\n\n"
                        "Actúa como experto en marketing digital para Meta Ads. "
                        "Ofrece un resumen y consejos prácticos para optimizar la conversión "
                        "(CPA, Leads, Tasa de Conversión) de estos anuncios."
                    )
                    resp_conv = consulta_lenguaje_natural_tendencias(prompt_conv)
                    st.markdown("**Sugerencias IA - Conversión:**")
                    st.markdown(resp_conv)

    all_ads = sorted(data["Adname"].dropna().unique().tolist())
    selected_ad = st.selectbox("Seleccione el Anuncio", all_ads)
    metric_list = ["CTR", "CPA", "Tasa de Conversión", "Leads", "Videoaveragewatchtime"]
    selected_metric = st.selectbox("Seleccione la Métrica", metric_list)

    with st.expander("Resumen General (IA) - Top 3 Peores Tendencias", expanded=False):
        st.write("Detecta los 3 anuncios con peor tendencia (mayor relevancia negativa) en la métrica seleccionada, y ofrece sugerencias.")

        def calcular_metrica_ia(df_segment, metric):
            lpv   = df_segment["Landingpageviews"].sum()
            imp   = df_segment["Impressions"].sum()
            cost  = df_segment["TotalCost"].sum()
            leads = df_segment["PTP_total"].sum()

            if metric == "Videoaveragewatchtime":
                vw_weighted = (df_segment["Videoaveragewatchtime"] * df_segment["Impressions"]).sum()
                if imp > 0:
                    return vw_weighted / imp
                return 0

            if metric == "CTR":
                return (lpv / imp * 100) if imp > 0 else 0
            elif metric == "CPA":
                return cost if leads == 0 else (cost / leads)
            elif metric == "Tasa de Conversión":
                return (leads / imp * 100) if imp > 0 else 0
            elif metric == "Leads":
                return leads
            else:
                return 0

        def detectar_anuncios_peor_tendencia(df_all, metric, days=28):
            df_ok = df_all.copy()
            df_ok["Date"] = pd.to_datetime(df_ok["Date"], errors="coerce")
            df_ok.dropna(subset=["Date"], inplace=True)
            last_d = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
            start_d = last_d - pd.Timedelta(days=days)
            df_ok = df_ok[(df_ok["Date"] >= start_d) & (df_ok["Date"] <= last_d)].copy()
            if df_ok.empty:
                return []

            invert_slope = True if metric == "CPA" else False
            slopes = []
            for ad_i in df_ok["Adname"].unique():
                df_sub = df_ok[df_ok["Adname"]==ad_i].copy()
                if df_sub.empty:
                    continue
                df_sub.set_index("Date", drop=False, inplace=True)
                daily_list = []
                for date_g, dfg in df_sub.resample('D'):
                    if not dfg.empty:
                        val = calcular_metrica_ia(dfg, metric)
                        daily_list.append((date_g, val))
                df_g = pd.DataFrame(daily_list, columns=["Periodo","Val"])
                df_g.dropna(subset=["Periodo"], inplace=True)
                df_g.sort_values("Periodo", inplace=True)
                if len(df_g)<2:
                    continue

                x_vals = np.arange(len(df_g)).reshape(-1,1)
                y_vals = df_g["Val"].values.reshape(-1,1)
                model = LinearRegression().fit(x_vals, y_vals)
                slope = model.coef_[0][0]

                if invert_slope:
                    worst = slope
                else:
                    worst = -slope
                slopes.append((ad_i, slope, worst))

            if not slopes:
                return []
            slopes.sort(key=lambda x: x[2], reverse=True)
            return slopes[:3]

        generar_resumen_IA = st.button("Generar Resumen IA (Top 3 negativos)")
        if generar_resumen_IA:
            top_3 = detectar_anuncios_peor_tendencia(data, selected_metric, days=28)
            if not top_3:
                st.warning("No se detectó ningún anuncio con tendencia negativa o no hay datos suficientes.")
            else:
                intro = (f"Top 3 anuncios con peor tendencia en la métrica '{selected_metric}' en los últimos 28 días:\n\n")
                detalle = ""
                for ad, slope, worst in top_3:
                    if selected_metric == "CPA":
                        trend_str = "al alza" if slope>0 else "a la baja"
                    else:
                        trend_str = "a la baja" if slope<0 else "al alza"
                    detalle += f"- Anuncio '{ad}', slope={slope:.4f}, tendencia {trend_str}\n"
                prompt = (
                    f"{intro}{detalle}\n\n"
                    "Actúa como un experto en marketing digital y ofrece un resumen y sugerencias para mejorar el rendimiento "
                    "de estos anuncios con tendencia negativa o poco favorable."
                )
                resp_ia = consulta_lenguaje_natural_tendencias(prompt)
                st.markdown("### Sugerencias IA:")
                st.markdown(resp_ia)

    # ------------------------------------------------------
    # SECCIÓN EXISTENTE DE GRÁFICOS Y TABLAS (NO ALTERADA)
    # ------------------------------------------------------
    df_page = data[data["Adname"] == selected_ad].copy()
    df_page["Date"] = pd.to_datetime(df_page["Date"], errors="coerce")
    df_page.dropna(subset=["Date"], inplace=True)

    today = pd.Timestamp.now().normalize()
    last_day = today - pd.Timedelta(days=1)
    df_page = df_page[df_page["Date"] <= last_day].copy()
    if df_page.empty:
        st.warning("No hay datos para el anuncio seleccionado antes de HOY-1.")
        st.stop()

    df_global = data.copy()
    df_global["Date"] = pd.to_datetime(df_global["Date"], errors="coerce")
    df_global.dropna(subset=["Date"], inplace=True)

    adset_of_ad = df_page["AdSetname"].dropna().unique()
    selected_adset = adset_of_ad[0] if len(adset_of_ad)>0 else None
    if selected_adset is not None:
        df_adset = df_global[df_global["AdSetname"]==selected_adset].copy()
    else:
        df_adset = pd.DataFrame(columns=df_global.columns)

    def calcular_metrica(df_segment, metric):
        lpv   = df_segment["Landingpageviews"].sum()
        imp   = df_segment["Impressions"].sum()
        cost  = df_segment["TotalCost"].sum()
        leads = df_segment["PTP_total"].sum()

        if metric == "Videoaveragewatchtime":
            sum_wt = (df_segment["Videoaveragewatchtime"] * df_segment["Impressions"]).sum()
            if imp>0:
                return sum_wt / imp
            return 0

        if metric == "CTR":
            return (lpv / imp * 100) if imp > 0 else 0
        elif metric == "CPA":
            return cost if leads == 0 else (cost / leads)
        elif metric == "Tasa de Conversión":
            return (leads / imp * 100) if imp > 0 else 0
        elif metric == "Leads":
            return leads
        else:
            return 0

    def posicionar_leyenda_sup_izq(fig, titulo):
        fig.update_layout(
            title={
                'text': titulo,
                'x': 0.0,
                'xanchor': 'left'
            },
            legend=dict(
                orientation="h",
                yanchor="bottom",
                xanchor="left",
                y=1.03,
                x=0
            ),
            margin=dict(l=50, r=30, t=60, b=30)
        )
        return fig

    def plot_regresion_lineal(df_grouped, graph_title):
        fig = go.Figure()
        slope, intercept = None, None

        if df_grouped.empty:
            fig.add_annotation(text="Sin datos para graficar", showarrow=False)
            fig.update_layout(xaxis_title="Periodo", yaxis_title=selected_metric)
            fig = posicionar_leyenda_sup_izq(fig, graph_title)
            return fig, slope, intercept

        x_vals = np.arange(len(df_grouped)).reshape(-1, 1)
        y_vals = df_grouped["MetricValue"].values.reshape(-1, 1)

        if len(df_grouped) >= 2:
            model = LinearRegression().fit(x_vals, y_vals)
            slope = model.coef_[0][0]
            intercept = model.intercept_[0]
            y_pred = model.predict(x_vals)
        else:
            y_pred = y_vals

        fig.add_trace(go.Scatter(
            x=df_grouped["Periodo"],
            y=df_grouped["MetricValue"],
            mode="markers+lines",
            name="Observado",
            line=dict(color="blue")
        ))
        if len(df_grouped) >= 2:
            fig.add_trace(go.Scatter(
                x=df_grouped["Periodo"],
                y=y_pred.flatten(),
                mode="lines",
                name="Tendencia",
                line=dict(color="red", dash="dash")
            ))

        fig.update_layout(
            xaxis_title="Periodo",
            yaxis_title=selected_metric
        )
        fig = posicionar_leyenda_sup_izq(fig, graph_title)
        return fig, slope, intercept

    def filtrar_y_agrup_diario(df_orig, start_date, end_date):
        subf = df_orig[(df_orig["Date"] >= start_date) & (df_orig["Date"] <= end_date)].copy()
        if subf.empty:
            return pd.DataFrame(columns=["Periodo","MetricValue"])
        subf.set_index("Date", drop=False, inplace=True)

        points = []
        for date_g, dd in subf.resample('D'):
            if not dd.empty:
                val = calcular_metrica(dd, selected_metric)
                points.append((date_g, val))
        dfg = pd.DataFrame(points, columns=["Periodo","MetricValue"])
        dfg.dropna(subset=["Periodo"], inplace=True)
        dfg.sort_values("Periodo", inplace=True)
        return dfg

    def filtrar_y_agrup_semanal(df_orig, start_date, end_date):
        end_date = pd.Timestamp(end_date).normalize()
        start_date = end_date - pd.Timedelta(days=27)
        
        subf = df_orig[(df_orig["Date"] >= start_date) & (df_orig["Date"] <= end_date)].copy()
        if subf.empty:
            return pd.DataFrame(columns=["Periodo","MetricValue"])
        
        points = []
        dates = pd.date_range(start=start_date, end=end_date, freq='7D')
        
        if end_date not in dates:
            dates = dates.append(pd.DatetimeIndex([end_date]))
        
        for i in range(len(dates)-1):
            period_start = dates[i]
            period_end = min(dates[i+1] - pd.Timedelta(days=1), end_date)
            week_data = subf[(subf["Date"] >= period_start) & (subf["Date"] <= period_end)]
            if not week_data.empty:
                val = calcular_metrica(week_data, selected_metric)
                points.append((period_start, val))
        
        last_start = dates[-1]
        if last_start <= end_date:
            last_week = subf[(subf["Date"] >= last_start) & (subf["Date"] <= end_date)]
            if not last_week.empty:
                val = calcular_metrica(last_week, selected_metric)
                points.append((last_start, val))
        
        dfg = pd.DataFrame(points, columns=["Periodo","MetricValue"])
        return dfg

    st.markdown("<h3 style='color: #1877f2; font-weight: 700;'>Mensual (4 semanas)</h3>", unsafe_allow_html=True)
    end_mensual = last_day
    df_mensual = filtrar_y_agrup_semanal(df_page, None, end_mensual)
    fig_mensual, _, _ = plot_regresion_lineal(df_mensual,"")
    st.plotly_chart(fig_mensual, use_container_width=True)

    st.markdown("<h3 style='color: #1877f2; font-weight: 700;'>Bi-semanal</h3>", unsafe_allow_html=True)
    end_bisemanal = last_day
    start_bisemanal = end_bisemanal - pd.Timedelta(days=27)
    mid_bisemanal = start_bisemanal + pd.Timedelta(days=14) - pd.Timedelta(days=1)

    col_left, col_right = st.columns(2)
    with col_left:
        df_bi_1 = filtrar_y_agrup_diario(df_page, start_bisemanal, mid_bisemanal)
        fig_bi_1, _, _ = plot_regresion_lineal(df_bi_1, "Primeras 2 Semanas")
        st.plotly_chart(fig_bi_1, use_container_width=True)
    with col_right:
        df_bi_2 = filtrar_y_agrup_diario(df_page, mid_bisemanal + pd.Timedelta(days=1), end_bisemanal)
        fig_bi_2, _, _ = plot_regresion_lineal(df_bi_2, "Últimas 2 Semanas")
        st.plotly_chart(fig_bi_2, use_container_width=True)

    st.markdown("<h3 style='color: #1877f2; font-weight: 700;'>Semanal</h3>", unsafe_allow_html=True)
    sem1_start = start_bisemanal
    sem1_end = sem1_start + pd.Timedelta(days=7) - pd.Timedelta(days=1)
    sem2_start = sem1_end + pd.Timedelta(days=1)
    sem2_end = sem2_start + pd.Timedelta(days=7) - pd.Timedelta(days=1)
    sem3_start = sem2_end + pd.Timedelta(days=1)
    sem3_end = sem3_start + pd.Timedelta(days=7) - pd.Timedelta(days=1)
    sem4_start = sem3_end + pd.Timedelta(days=1)
    sem4_end = end_bisemanal

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    fig_w1, slope1, intercept1 = plot_regresion_lineal(filtrar_y_agrup_diario(df_page, sem1_start, sem1_end), "Semana 1")
    with col_s1:
        st.plotly_chart(fig_w1, use_container_width=True)
        if slope1 is not None and intercept1 is not None:
            st.markdown(f"**Ecuación**: y = {slope1:.4f}x + {intercept1:.4f}")

    fig_w2, slope2, intercept2 = plot_regresion_lineal(filtrar_y_agrup_diario(df_page, sem2_start, sem2_end), "Semana 2")
    with col_s2:
        st.plotly_chart(fig_w2, use_container_width=True)
        if slope2 is not None and intercept2 is not None:
            st.markdown(f"**Ecuación**: y = {slope2:.4f}x + {intercept2:.4f}")

    fig_w3, slope3, intercept3 = plot_regresion_lineal(filtrar_y_agrup_diario(df_page, sem3_start, sem3_end), "Semana 3")
    with col_s3:
        st.plotly_chart(fig_w3, use_container_width=True)
        if slope3 is not None and intercept3 is not None:
            st.markdown(f"**Ecuación**: y = {slope3:.4f}x + {intercept3:.4f}")

    fig_w4, slope4, intercept4 = plot_regresion_lineal(filtrar_y_agrup_diario(df_page, sem4_start, sem4_end), "Semana 4")
    with col_s4:
        st.plotly_chart(fig_w4, use_container_width=True)
        if slope4 is not None and intercept4 is not None:
            st.markdown(f"**Ecuación**: y = {slope4:.4f}x + {intercept4:.4f}")

    def calcular_resumen_semanal(df_metric, start_date, end_date):
        if df_metric.empty:
            mean_val = 0
            std_val  = 0
            max_val  = 0
            min_val  = 0
        else:
            vals    = df_metric["MetricValue"]
            mean_val= vals.mean()
            std_val = vals.std()
            max_val = vals.max()
            min_val = vals.min()

        df_filter_anuncio = df_page[(df_page["Date"]>=start_date) & (df_page["Date"]<=end_date)]
        cost_ad = df_filter_anuncio["TotalCost"].sum()

        df_filter_adset = df_adset[(df_adset["Date"]>=start_date) & (df_adset["Date"]<=end_date)]
        cost_adset = df_filter_adset["TotalCost"].sum() if not df_filter_adset.empty else 0

        df_filter_global = df_global[(df_global["Date"]>=start_date) & (df_global["Date"]<=end_date)]
        cost_global = df_filter_global["TotalCost"].sum() if not df_filter_global.empty else 0

        mean_adset  = calcular_metrica(df_filter_adset, selected_metric) if not df_filter_adset.empty else 0
        mean_global = calcular_metrica(df_filter_global, selected_metric) if not df_filter_global.empty else 0

        cost_ratio_conjunto = (cost_ad / cost_adset * 100) if cost_adset>0 else 0
        cost_ratio = (cost_ad / cost_global * 100) if cost_global>0 else 0

        return {
            "mean_anuncio": mean_val,
            "mean_adset": mean_adset,
            "mean_global": mean_global,
            "std": std_val,
            "max": max_val,
            "min": min_val,
            "cost": cost_ad,
            "cost_ratio_conjunto": cost_ratio_conjunto,
            "cost_ratio": cost_ratio
        }

    df_w1 = filtrar_y_agrup_diario(df_page, sem1_start, sem1_end)
    df_w2 = filtrar_y_agrup_diario(df_page, sem2_start, sem2_end)
    df_w3 = filtrar_y_agrup_diario(df_page, sem3_start, sem3_end)
    df_w4 = filtrar_y_agrup_diario(df_page, sem4_start, sem4_end)

    r1 = calcular_resumen_semanal(df_w1, sem1_start, sem1_end)
    r2 = calcular_resumen_semanal(df_w2, sem2_start, sem2_end)
    r3 = calcular_resumen_semanal(df_w3, sem3_start, sem3_end)
    r4 = calcular_resumen_semanal(df_w4, sem4_start, sem4_end)

    rows = {
        "Prom. Métrica (Anuncio)": [r1["mean_anuncio"], r2["mean_anuncio"], r3["mean_anuncio"], r4["mean_anuncio"]],
        "Prom. Métrica (Conjunto)": [r1["mean_adset"], r2["mean_adset"], r3["mean_adset"], r4["mean_adset"]],
        "Prom. Métrica (Global)": [r1["mean_global"], r2["mean_global"], r3["mean_global"], r4["mean_global"]],
        "Desviación Estándar": [r1["std"], r2["std"], r3["std"], r4["std"]],
        "Valor Máximo": [r1["max"], r2["max"], r3["max"], r4["max"]],
        "Valor Mínimo": [r1["min"], r2["min"], r3["min"], r4["min"]],
        "Inversión Total (Cost)": [r1["cost"], r2["cost"], r3["cost"], r4["cost"]],
        "% Inversión vs Conjunto": [
            r1["cost_ratio_conjunto"], r2["cost_ratio_conjunto"],
            r3["cost_ratio_conjunto"], r4["cost_ratio_conjunto"]
        ],
        "% Inversión vs Total": [
            r1["cost_ratio"], r2["cost_ratio"], r3["cost_ratio"], r4["cost_ratio"]
        ]
    }
    df_summary = pd.DataFrame(rows, index=["Semana 1","Semana 2","Semana 3","Semana 4"]).T

    st.markdown("<h3 style='color: #1877f2; font-weight: 700;'>Resumen Comparativo Semanal</h3>", unsafe_allow_html=True)
    st.dataframe(df_summary, use_container_width=True)

    st.markdown("----")
    st.markdown("<h3 style='color: #1877f2; font-weight: 700;'>Tasa de Cambio e Índice de Estabilidad</h3>", unsafe_allow_html=True)

    def tasa_de_cambio(prev_val, curr_val):
        if abs(prev_val)>0:
            return (curr_val - prev_val)/abs(prev_val)*100
        else:
            return 0

    tc12 = tasa_de_cambio(r1["mean_anuncio"], r2["mean_anuncio"])
    tc23 = tasa_de_cambio(r2["mean_anuncio"], r3["mean_anuncio"])
    tc34 = tasa_de_cambio(r3["mean_anuncio"], r4["mean_anuncio"])

    st.write(f"Tasa de cambio S1→S2: {tc12:.2f}% | S2→S3: {tc23:.2f}% | S3→S4: {tc34:.2f}%")

    slopes_week = [abs(x) for x in (slope1, slope2, slope3, slope4) if x is not None]
    if slopes_week:
        avg_slope = sum(slopes_week)/len(slopes_week)
        indice_estabilidad = 1/(1+avg_slope)
    else:
        indice_estabilidad = 1.0

    st.write(f"Índice de Estabilidad (0..1): {indice_estabilidad:.3f}")
    st.markdown("""
    - **Tasa de cambio**: variación % de la métrica promedio semanal de una semana a la siguiente.
    - **Índice de Estabilidad**: 1/(1+pendiente promedio). A mayor pendiente, menor estabilidad.
    """)

    st.markdown("<h3 style='color: #1877f2; font-weight: 700;'>Sugerencias IA para el Anuncio Actual</h3>", unsafe_allow_html=True)
    generar_ia_anuncio = st.button("Generar Sugerencias IA (Anuncio Actual)")
    if generar_ia_anuncio:
        summary_text = ""
        for i, week_label in enumerate(["Semana 1","Semana 2","Semana 3","Semana 4"]):
            wdict = [r1,r2,r3,r4][i]
            summary_text += (f"{week_label}:\n"
                f"  - Promedio Métrica (Anuncio): {wdict['mean_anuncio']:.2f}\n"
                f"  - Promedio Métrica (AdSet): {wdict['mean_adset']:.2f}\n"
                f"  - Promedio Métrica (Global): {wdict['mean_global']:.2f}\n"
                f"  - Desv. Estándar: {wdict['std']:.2f}\n"
                f"  - Máx: {wdict['max']:.2f}, Mín: {wdict['min']:.2f}\n"
                f"  - Inversión: {wdict['cost']:.2f}, % vs Total: {wdict['cost_ratio']:.2f}%\n\n"
            )

        prompt_anuncio = f"""
        El anuncio seleccionado: {selected_ad}.
        Métrica en análisis: {selected_metric}.

        Resumen semanal (últimas 4 semanas):
        {summary_text}

        Tasa de cambio:
         - S1→S2: {tc12:.2f}%
         - S2→S3: {tc23:.2f}%
         - S3→S4: {tc34:.2f}%

        Índice de Estabilidad: {indice_estabilidad:.3f} (cercano a 1 => estable, cercano a 0 => muy inestable).

        Actúa como un experto en marketing digital para Meta Ads. 
        Ofrece sugerencias específicas para mejorar el rendimiento de este anuncio, 
        teniendo en cuenta sus resultados en las últimas 4 semanas, 
        la tasa de cambio y la estabilidad, así como la comparación con el conjunto y el global.
        """

        resp_anuncio = consulta_lenguaje_natural_tendencias(prompt_anuncio)
        st.markdown("#### Respuesta IA para el Anuncio Actual:")
        st.markdown(resp_anuncio)


    # ------------------------------------------------------
    # (NUEVO) ANÁLISIS INTEGRAL (NUEVO) SOBRE df_esp
    # ------------------------------------------------------
    with st.expander("Análisis Integral (NUEVO)", expanded=False):
        """
        - Se filtra df_esp con las mismas condiciones (Adname == selected_ad, últimos 28 días).
        - Se generan las 4 semanas + correlaciones.
        - Se arma el prompt en 6 secciones (1..5 + Correlaciones).
        - La IA puede añadir "Acciones Recomendadas" sin alterar las demás partes.
        """
        st.write("**Análisis Integral basado en df_esp** (usa el mismo Anuncio seleccionado y 28 días)")

        # 1) Filtrar df_esp
        # Asumimos df_esp ya está en memoria global
        df_esp_copy = df_esp.copy()
        df_esp_copy["Date"] = pd.to_datetime(df_esp_copy["Date"], errors="coerce")
        df_esp_copy.dropna(subset=["Date"], inplace=True)

        # Ultimos 28 dias
        last_d_esp = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
        start_d_esp = last_d_esp - pd.Timedelta(days=28)
        df_esp_copy = df_esp_copy[(df_esp_copy["Date"] >= start_d_esp) & (df_esp_copy["Date"] <= last_d_esp)]

        # Filtrar por "Adname" == selected_ad
        df_esp_copy = df_esp_copy[df_esp_copy["Adname"] == selected_ad].copy()

        if df_esp_copy.empty:
            st.warning("df_esp no tiene datos para este anuncio en los últimos 28 días.")
        else:
            # A) Resumir por semana (similar a tu ejemplo)
            def resumir_por_semana_esp(df_esp_f):
                if df_esp_f.empty:
                    return {}

                df_esp_f["WeekNum"] = df_esp_f["Date"].dt.isocalendar().week
                weeks_unicas = sorted(df_esp_f["WeekNum"].unique())
                if len(weeks_unicas) > 4:
                    weeks_unicas = weeks_unicas[-4:]

                info_s = {
                    "tiempo_reproduccion": {},
                    "ctr": {},
                    "tasa_conversion": {},
                    "cpa": {},
                    "retencion_50": {},
                }
                for i, w in enumerate(weeks_unicas, start=1):
                    dfw = df_esp_f[df_esp_f["WeekNum"] == w].copy()

                    imp = dfw["Impressions"].sum() if "Impressions" in dfw.columns else 0
                    if imp>0 and "Videoaveragewatchtime" in dfw.columns:
                        sum_wt = (dfw["Videoaveragewatchtime"] * dfw["Impressions"]).sum()
                        tiempo_rep = sum_wt / imp
                    else:
                        tiempo_rep = 0

                    ctr_ = dfw["CTRall"].mean() if "CTRall" in dfw.columns and not dfw.empty else 0

                    lpv = dfw["Landingpageviews"].sum() if "Landingpageviews" in dfw.columns else 0
                    ptp = dfw["PTP_total"].sum() if "PTP_total" in dfw.columns else 0
                    t_conv = (ptp / lpv * 100) if lpv>0 else 0

                    if "CPA" in dfw.columns and not dfw["CPA"].isna().all():
                        cpa_ = dfw["CPA"].mean()
                    else:
                        cpa_ = 0

                    if "Videowatchesat50Percent" in dfw.columns and imp>0:
                        vw50 = dfw["Videowatchesat50Percent"].sum()
                        ret_50 = (vw50 / imp)*100
                    else:
                        ret_50 = 0

                    lbl = f"sem{i}"
                    info_s["tiempo_reproduccion"][lbl] = tiempo_rep
                    info_s["ctr"][lbl] = ctr_
                    info_s["tasa_conversion"][lbl] = t_conv
                    info_s["cpa"][lbl] = cpa_
                    info_s["retencion_50"][lbl] = ret_50

                return info_s

            # B) Correlaciones
            def calcular_correlaciones_esp(df_sub):
                corrs = {
                    "ctr_vs_tiempo": None,
                    "tiempo_vs_abandono": None,
                    "ctr_vs_conversion": None
                }
                # Creamos TasaConversionPostLanding
                if "Landingpageviews" in df_sub.columns and "PTP_total" in df_sub.columns:
                    df_sub = df_sub.copy()
                    df_sub["TasaConversionPostLanding"] = df_sub.apply(
                        lambda row: (row["PTP_total"]/row["Landingpageviews"]*100) 
                                    if row["Landingpageviews"]>0 else np.nan,
                        axis=1
                    )

                # 1) CTRall vs Videoaveragewatchtime
                if "CTRall" in df_sub.columns and "Videoaveragewatchtime" in df_sub.columns:
                    sub1 = df_sub[["CTRall","Videoaveragewatchtime"]].dropna()
                    if len(sub1)>1:
                        corrs["ctr_vs_tiempo"] = sub1.corr().iloc[0,1]

                # 2) Tiempo vs TasaAbandono (si la tuvieras)
                if "Videoaveragewatchtime" in df_sub.columns and "TasaAbandono" in df_sub.columns:
                    sub2 = df_sub[["Videoaveragewatchtime","TasaAbandono"]].dropna()
                    if len(sub2)>1:
                        corrs["tiempo_vs_abandono"] = sub2.corr().iloc[0,1]

                # 3) CTRall vs TasaConversionPostLanding
                if "CTRall" in df_sub.columns and "TasaConversionPostLanding" in df_sub.columns:
                    sub3 = df_sub[["CTRall","TasaConversionPostLanding"]].dropna()
                    if len(sub3)>1:
                        corrs["ctr_vs_conversion"] = sub3.corr().iloc[0,1]

                return corrs

            # C) Generar prompt con las 6 secciones (1..5 + correlaciones)
            def generar_prompt_integral_correlaciones_esp(anuncio, info_s, corrs):
                sems = sorted(info_s["tiempo_reproduccion"].keys())
                if not sems:
                    return "No hay datos semanales."

                s1 = "1) TENDENCIAS CLARAS EN EL ENGAGEMENT Y RETENCIÓN\n"
                s1 += "- Tiempo Medio de Reproducción:\n"
                for s in sems:
                    s1 += f"  {s}: {info_s['tiempo_reproduccion'][s]:.2f}s\n"
                s1 += "\n- CTR (CTRall):\n"
                for s in sems:
                    s1 += f"  {s}: {info_s['ctr'][s]:.2f}%\n"
                s1 += "\n- Retención al 50%:\n"
                for s in sems:
                    s1 += f"  {s}: {info_s['retencion_50'][s]:.2f}%\n"

                s2 = "\n2) CÓMO ESTAS TENDENCIAS IMPACTAN EN LA CONVERSIÓN\n"
                s2 += "- Tasa de Conversión Post-Landing:\n"
                for s in sems:
                    s2 += f"  {s}: {info_s['tasa_conversion'][s]:.2f}%\n"
                s2 += "\n- CPA:\n"
                for s in sems:
                    s2 += f"  {s}: {info_s['cpa'][s]:.2f}\n"

                s3 = "\n3) EVIDENCIA DE SATURACIÓN BASADA EN LOS DATOS\n"

                s4 = "\n4) RECOMENDACIÓN BASADA ÚNICAMENTE EN ESTAS MÉTRICAS\n"

                # Comparar semana inicial vs final
                s_ini, s_fin = sems[0], sems[-1]
                ctr_ini = info_s["ctr"][s_ini]
                ctr_fin = info_s["ctr"][s_fin]
                conv_ini = info_s["tasa_conversion"][s_ini]
                conv_fin = info_s["tasa_conversion"][s_fin]

                s5 = f"""
5) CONCLUSIÓN FINAL: ¿PROBLEMA EN EL LANDING O EN EL ANUNCIO?
- CTR (S1 vs S4): {ctr_ini:.2f}% -> {ctr_fin:.2f}%
- Tasa Conversión Post-Landing: {conv_ini:.2f}% -> {conv_fin:.2f}%
Si CTR cae más, falla el anuncio.
Si post-landing cae más, falla la landing.
"""

                # Correlaciones
                def fnum(x):
                    return f"{x:.2f}" if x is not None else "N/A"
                c_tiempo = fnum(corrs["ctr_vs_tiempo"])
                c_aband = fnum(corrs["tiempo_vs_abandono"])
                c_conv  = fnum(corrs["ctr_vs_conversion"])

                s6 = f"""
6) CORRELACIONES CLAVE:
   - CTR vs. Tiempo Reproducción: {c_tiempo}
   - Tiempo de Reproducción vs. Tasa de Abandono: {c_aband}
   - CTR vs. Tasa de Conversión Post-Landing: {c_conv}
"""

                final_txt = f"""
ANÁLISIS INTEGRAL DEL ANUNCIO: {anuncio}

{s1}{s2}{s3}{s4}{s5}{s6}
"""
                return final_txt

            info_esp = resumir_por_semana_esp(df_esp_copy)
            if not info_esp:
                st.warning("No se pudo resumir las 4 semanas en df_esp (datos insuficientes).")
            else:
                corr_esp = calcular_correlaciones_esp(df_esp_copy)
                prompt_nuevo = generar_prompt_integral_correlaciones_esp(selected_ad, info_esp, corr_esp)

                if st.button("Generar Análisis Integral (NUEVO) con Correlaciones"):
                    resp_esp = consulta_lenguaje_natural_tendencias(prompt_nuevo)
                    st.markdown("### Respuesta IA - Análisis Integral (df_esp)")
                    st.write(resp_esp)



elif selected_page == "Ranking de Anuncios":
    
    
    # -----------------------------------------------------
    # ENCABEZADO Y DESCRIPCIÓN
    # -----------------------------------------------------

    st.title("Ranking de Anuncios")
    st.markdown("""
    En esta sección se enlistan los **mejores anuncios** en cada métrica, combinando aspectos de **Conversión** (CPA, CTR, Tasa de Conversión) 
    y de **Engagement** (tasa de reacciones, guardados, compartidos, comentarios).

    ### Ecuación de cálculo propuesta
    """, unsafe_allow_html=True)

    # Mostrar imagen centrada con Streamlit
    from PIL import Image

    image_path = "ecuacion2.png"  # Asegúrate de que la imagen está en la misma carpeta
    try:
        image = Image.open(image_path)
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.image(image, caption="Fórmula de cálculo de puntuación", width=1200)
        st.markdown("</div>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("No se pudo encontrar la imagen. Asegúrate de que 'ecuacion2.png' esté en la ubicación correcta.")

    st.markdown("""
    - **Engagement**: 0.2*(Score_Reacciones) + 0.3*(Score_Guardados) + 0.3*(Score_Compartidos) + 0.2*(Score_Comentarios)
    - **Conversión**: Score_CPA + Score_CTR + Score_Tasa de Conversión
    """)
    # -----------------------------------------------------
    # 1) FILTROS PRINCIPALES (Campaña, Conjunto, Fechas)
    # -----------------------------------------------------
    # Supongamos que df_base ya está disponible en tu entorno.
    # O puedes cargarlo aquí con "pd.read_csv(...)"
    # ...
    df_base = pd.read_csv("df_activos.csv")  # ajusta la ruta

    # Filas únicas de campaña, conjunto:
    campaigns = ["General"] + sorted(df_base["Campaignname"].dropna().unique().tolist())
    adsets = ["General"] + sorted(df_base["AdSetname"].dropna().unique().tolist())

    selected_campaign = st.sidebar.selectbox("Campaña:", campaigns)
    selected_adset = st.sidebar.selectbox("Conjunto de Anuncios:", adsets)

    # Filtro de Fechas
    min_date = pd.to_datetime(df_base["Date"]).min()
    max_date = pd.to_datetime(df_base["Date"]).max()
    start_date = st.sidebar.date_input("Fecha inicio:", value=min_date)
    end_date = st.sidebar.date_input("Fecha fin:", value=max_date)

    # Aplica filtros
    df_base["Date"] = pd.to_datetime(df_base["Date"])
    df_filter = df_base[
        (df_base["Date"] >= pd.to_datetime(start_date)) &
        (df_base["Date"] <= pd.to_datetime(end_date))
    ].copy()

    if selected_campaign != "General":
        df_filter = df_filter[df_filter["Campaignname"] == selected_campaign]
    if selected_adset != "General":
        df_filter = df_filter[df_filter["AdSetname"] == selected_adset]

    if df_filter.empty:
        st.warning("No hay datos para mostrar con los filtros seleccionados.")
        st.stop()

    # -----------------------------------------------------
    # 2) AJUSTES DE α Y PESOS w_i
    # -----------------------------------------------------
    st.markdown("### Parámetros de Ponderación")
    alpha = st.number_input("Valor de α (alpha)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
    w_reactions = st.number_input("w Reacciones (Likes)", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    w_saves = st.number_input("w Guardados (Saves)", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
    w_shares = st.number_input("w Compartidos (Shares)", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
    w_comments = st.number_input("w Comentarios (Comments)", min_value=0.0, max_value=1.0, value=0.2, step=0.1)

    # -----------------------------------------------------
    # 3) CÁLCULO DE TOPS PARA CONVERSIÓN (CPA, CTR, TasaConv)
    # -----------------------------------------------------
    # Filtro: Ej. Adid con >30 leads, etc. (puedes ajustar)
    df_filtered = df_filter.groupby('Adid').filter(lambda x: x['PTP_total'].sum() > 30)

    # Agregar datos conversion
    df_conv = df_filtered.groupby(['Adid','TIPO DE ANUNCIO','Adname']).agg(
        TotalCost=('TotalCost','sum'),
        PTP_total=('PTP_total','sum'),
        Landingpageviews=('Landingpageviews','sum'),
        Impressions=('Impressions','sum')
    ).reset_index()

    df_conv["CPA"] = df_conv["TotalCost"]/df_conv["PTP_total"]
    df_conv["CTR"] = (df_conv["Landingpageviews"]/df_conv["Impressions"])*100
    df_conv["TasaConversion"] = (df_conv["PTP_total"]/df_conv["Impressions"])*1000

    top_cpa = df_conv.nsmallest(10,"CPA").reset_index(drop=True)
    top_ctr = df_conv.nlargest(10,"CTR").reset_index(drop=True)
    top_tc  = df_conv.nlargest(10,"TasaConversion").reset_index(drop=True)

    # -----------------------------------------------------
    # 4) CÁLCULO DE TOPS PARA ENGAGEMENT (reaction, saves, shares, comments)
    # -----------------------------------------------------
    df_eng = df_filtered.groupby(['Adid','TIPO DE ANUNCIO','Adname']).agg(
        Postreactions=('Postreactions','sum'),
        Postsaves=('Postsaves','sum'),
        Postshares=('Postshares','sum'),
        Postcomments=('Postcomments','sum'),
        Reach=('Reach','sum')
    ).reset_index()

    df_eng["ReactionRate"] = (df_eng["Postreactions"]/df_eng["Reach"])*100
    df_eng["SaveRate"] = (df_eng["Postsaves"]/df_eng["Reach"])*100
    df_eng["ShareRate"] = (df_eng["Postshares"]/df_eng["Reach"])*100
    df_eng["CommentRate"] = (df_eng["Postcomments"]/df_eng["Reach"])*100

    top_reactions = df_eng.nlargest(10,"ReactionRate").reset_index(drop=True)
    top_saves     = df_eng.nlargest(10,"SaveRate").reset_index(drop=True)
    top_shares    = df_eng.nlargest(10,"ShareRate").reset_index(drop=True)
    top_comments  = df_eng.nlargest(10,"CommentRate").reset_index(drop=True)

    # -----------------------------------------------------
    # 5) CALCULAR EL RANKING TOTAL (Score) 
    #    UNIFICANDO CONVERSIÓN Y ENGAGEMENT
    # -----------------------------------------------------
    # Helper para posición en top 10 => rank 1..10; si no está => rank=11
    def get_position(adname, df_metric, col_ad="Adname"):
        if adname in df_metric[col_ad].values:
            return df_metric[df_metric[col_ad] == adname].index[0] + 1
        return 11

    def calculate_score(position):
        """score(pos) = 1 / [1 + 1/(11-pos)] si pos<=10, sino 0."""
        if position <= 10:
            return 1/(1+1/(11-position))
        else:
            return 0

    # Listado total de Adname que aparecen en alguno de los top 10
    adnames_total = pd.concat([
        top_cpa["Adname"], top_ctr["Adname"], top_tc["Adname"],
        top_reactions["Adname"], top_saves["Adname"], top_shares["Adname"], top_comments["Adname"]
    ]).unique()

    # Creamos df_rank con Adname, TIPO DE ANUNCIO
    def get_tipo(adname):
        # Buscamos en cualquiera de los df top
        for df_ in [top_cpa, top_ctr, top_tc, top_reactions, top_saves, top_shares, top_comments]:
            if adname in df_["Adname"].values:
                return df_.loc[df_["Adname"]==adname,"TIPO DE ANUNCIO"].values[0]
        return "Desconocido"

    df_rank = pd.DataFrame({"Adname": adnames_total})
    df_rank["TIPO DE ANUNCIO"] = df_rank["Adname"].apply(get_tipo)

    # Ranks conversión
    df_rank["CPA_rank"] = df_rank["Adname"].apply(lambda x: get_position(x, top_cpa))
    df_rank["CTR_rank"] = df_rank["Adname"].apply(lambda x: get_position(x, top_ctr))
    df_rank["TasaC_rank"] = df_rank["Adname"].apply(lambda x: get_position(x, top_tc))

    # Ranks engagement
    df_rank["React_rank"] = df_rank["Adname"].apply(lambda x: get_position(x, top_reactions))
    df_rank["Save_rank"]  = df_rank["Adname"].apply(lambda x: get_position(x, top_saves))
    df_rank["Share_rank"] = df_rank["Adname"].apply(lambda x: get_position(x, top_shares))
    df_rank["Comm_rank"]  = df_rank["Adname"].apply(lambda x: get_position(x, top_comments))

    # Score para métricas de conversión: sum(3)
    df_rank["score_conv"] = (
        df_rank["CPA_rank"].apply(calculate_score) + 
        df_rank["CTR_rank"].apply(calculate_score) + 
        df_rank["TasaC_rank"].apply(calculate_score)
    )

    # Score para métricas sociales (c/ peso w)
    df_rank["score_social"] = (
        df_rank["React_rank"].apply(calculate_score)*w_reactions +
        df_rank["Save_rank"].apply(calculate_score)*w_saves +
        df_rank["Share_rank"].apply(calculate_score)*w_shares +
        df_rank["Comm_rank"].apply(calculate_score)*w_comments
    )

    # Score total = score_conv + alpha*score_social
    df_rank["Score"] = df_rank["score_conv"] + alpha*df_rank["score_social"]

    df_rank.sort_values("Score", ascending=False, inplace=True)
    df_rank.reset_index(drop=True, inplace=True)

    st.markdown("### 1) Ranking General (Score Total)")
    st.dataframe(df_rank[["TIPO DE ANUNCIO","Adname","Score"]].head(20), use_container_width=True)

    # -----------------------------------------------------
    # 6) RANKING DE CONVERSIÓN
    # -----------------------------------------------------
    st.markdown("### 2) Ranking de Conversión")
    st.write("""
    A continuación se listan los **Top 10** de cada métrica de conversión 
    (**CPA**, **CTR**, **Tasa de Conversión**). 
    Cuanto mejor sea la posición, mayor puntaje en el ranking total.
    """)

    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        st.markdown("**Top 10 CPA (menor es mejor)**")
        st.dataframe(top_cpa[["TIPO DE ANUNCIO","Adname","CPA"]], use_container_width=True)
    with col_c2:
        st.markdown("**Top 10 CTR (mayor es mejor)**")
        st.dataframe(top_ctr[["TIPO DE ANUNCIO","Adname","CTR"]], use_container_width=True)
    with col_c3:
        st.markdown("**Top 10 Tasa de Conversión**")
        st.dataframe(top_tc[["TIPO DE ANUNCIO","Adname","TasaConversion"]], use_container_width=True)

    # -----------------------------------------------------
    # 7) RANKING DE ENGAGEMENT
    # -----------------------------------------------------
    st.markdown("### 3) Ranking de Engagement")
    st.write("""
    Se muestran los **Top 10** de cada métrica de interacción 
    (Reacciones, Guardados, Compartidos y Comentarios).
    """)

    # Primera fila: guardados & compartidos
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown("**Top 10 Guardados** (SaveRate)")
        st.dataframe(top_saves[["TIPO DE ANUNCIO","Adname","SaveRate"]], use_container_width=True)
    with col_e2:
        st.markdown("**Top 10 Compartidos** (ShareRate)")
        st.dataframe(top_shares[["TIPO DE ANUNCIO","Adname","ShareRate"]], use_container_width=True)

    # Segunda fila: reacciones & comentarios
    col_e3, col_e4 = st.columns(2)
    with col_e3:
        st.markdown("**Top 10 Reacciones** (ReactionRate)")
        st.dataframe(top_reactions[["TIPO DE ANUNCIO","Adname","ReactionRate"]], use_container_width=True)
    with col_e4:
        st.markdown("**Top 10 Comentarios** (CommentRate)")
        st.dataframe(top_comments[["TIPO DE ANUNCIO","Adname","CommentRate"]], use_container_width=True)

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