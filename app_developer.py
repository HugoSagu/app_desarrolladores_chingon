# -*- coding: utf-8 -*-
"""HBU-NAVIGATOR | Terminal Institucional de Prefactibilidad para Desarrolladores (Con Comparables de Mercado $/M²)"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from streamlit_js_eval import get_geolocation
import pydeck as pdk
import os
import requests
import datetime

try:
    import geopandas as gpd
    from shapely.geometry import shape
except ImportError:
    st.error("Librerías geoespaciales no encontradas. Verifica requirements.txt.")

try:
    import h3
except ImportError:
    st.error("Módulo geoespacial 'h3' no encontrado. Verifica requirements.txt.")

try:
    from groq import Groq
except ImportError:
    st.error("Motor de Inteligencia Artificial no inicializado. Verifica requirements.txt.")

try:
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter
except ImportError:
    st.error("Módulo de geocodificación no encontrado. Verifica requirements.txt.")

# --- 1. CONFIGURACIÓN Y ESTILOS CORPORATIVOS ---
st.set_page_config(page_title="HBU-NAVIGATOR | Developer", layout="wide", initial_sidebar_state="collapsed")

if "location" not in st.session_state:
    st.session_state.location = None

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700;800&display=swap');

    .stApp { background-color: #FFFFFF; color: #1E293B; font-family: 'Inter', sans-serif; }

    .hud-panel {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    .predio-badge {
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        color: #1D4ED8;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 700;
        display: inline-block;
        margin-bottom: 8px;
    }

    .memo-panel {
        background: #F0FDF4;
        border: 1px solid #86EFAC;
        padding: 18px;
        border-radius: 8px;
        margin-bottom: 15px;
        box-shadow: 0 2px 6px rgba(16, 185, 129, 0.1);
    }

    .dial-container { display: flex; justify-content: center; align-items: center; padding: 5px 0; }

    .central-dial {
        width: 230px; height: 230px;
        border: 4px solid #0A2540;
        border-radius: 50%;
        display: flex; flex-direction: column;
        justify-content: center; align-items: center;
        background: #FFFFFF;
        box-shadow: 0 4px 20px rgba(10, 37, 64, 0.12);
    }

    .dial-header { color: #64748B; font-size: 0.68rem; letter-spacing: 2px; text-transform: uppercase; font-weight: 700; }
    .dial-value { color: #0A2540; font-size: 2.3rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; }
    .dial-sub { color: #2563EB; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; text-align: center; padding: 0 8px; }

    .stButton>button {
        background: #0A2540 !important;
        color: #FFFFFF !important;
        border: none !important;
        width: 100%; height: 46px;
        font-weight: 700; font-size: 0.92rem;
        letter-spacing: 1px; border-radius: 6px;
        text-transform: uppercase;
        box-shadow: 0 2px 8px rgba(10, 37, 64, 0.15);
    }

    .stButton>button:hover { background: #1E3A8A !important; color: #FFFFFF !important; }

    .status-pill {
        background: #DCFCE7; color: #166534; padding: 2px 8px;
        border-radius: 4px; font-weight: 700; font-size: 0.65rem;
        text-transform: uppercase; border: 1px solid #BBF7D0;
    }

    .archetype-pill {
        display: inline-block; padding: 3px 10px; border-radius: 4px;
        font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 1px;
    }

    [data-testid="stMetricValue"] { color: #0A2540 !important; font-family: 'JetBrains Mono', monospace; font-size: 1.35rem !important; }
    [data-testid="stMetricLabel"] { color: #64748B !important; font-weight: 600; text-transform: uppercase; font-size: 0.72rem; }
    </style>
""", unsafe_allow_html=True)

# --- 2. SISTEMA DE TELEMETRÍA ---
def registrar_telemetria(accion, data_mza, sup_lote, m2_const, residual_m2):
    try:
        log_file = "telemetria_uso_scout.csv"
        registro = {
            "fecha_hora": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "modulo": "DEVELOPER_HBU_MARKET",
            "accion": accion,
            "municipio": str(data_mza.get("NOM_MUN", "S/D")),
            "cvegeo_mza": str(data_mza.get("cvegeo_mza", "S/D")),
            "tipo_uso": str(data_mza.get("tipo_uso", "S/D")),
            "cus": float(data_mza.get("cus", 0.0)),
            "superficie_lote_m2": float(sup_lote),
            "m2_construccion_max": float(m2_const),
            "valor_residual_m2_tierra": float(residual_m2),
            "nse_score": float(data_mza.get("nse_score", 0.0))
        }
        df_log = pd.DataFrame([registro])
        if not os.path.exists(log_file):
            df_log.to_csv(log_file, index=False, mode="w")
        else:
            df_log.to_csv(log_file, index=False, mode="a", header=False)
    except Exception:
        pass

# --- 3. CONSULTA BAJO DEMANDA DE PREDIO CATASTRAL (ZAPOPAN WFS) ---
@st.cache_data(ttl=3600)
def consultar_predio_catastral(lat, lon, margen_metros=45.0):
    delta_deg = margen_metros / 111139.0
    bbox_str = f"{lon - delta_deg},{lat - delta_deg},{lon + delta_deg},{lat + delta_deg}"
    url_wfs = "https://mapa.zapopan.gob.mx:8000/geoserver/geomatica/ows"
    params = {
        "service": "WFS",
        "version": "1.0.0",
        "request": "GetFeature",
        "typeName": "geomatica:zpn_predios",
        "outputFormat": "application/json",
        "srsName": "EPSG:4326",
        "bbox": bbox_str
    }
    try:
        r = requests.get(url_wfs, params=params, timeout=4)
        if r.status_code == 200:
            data_geojson = r.json()
            features = data_geojson.get("features", [])
            if features:
                gdf_p = gpd.GeoDataFrame.from_features(data_geojson, crs="EPSG:4326")
                return gdf_p
    except Exception:
        pass
    return None

# --- 4. FUNCIONES AUXILIARES GEOGRÁFICAS ---
def latlng_to_h3_cell(lat, lon, res=9):
    try: return h3.latlng_to_cell(lat, lon, res)
    except AttributeError: return h3.geo_to_h3(lat, lon, res)

def decimal_to_dms(deg, is_lat=True):
    direction = ("N" if deg >= 0 else "S") if is_lat else ("E" if deg >= 0 else "W")
    deg = abs(deg)
    d = int(deg)
    m = int((deg - d) * 60)
    s = round((deg - d - m/60) * 3600, 2)
    return f"{direction} {d}°{m:02d}'{s:05.2f}\""

PARAMETRICOS_CMIC = {
    "Torre Departamentos Residencial Plus (10-15+ niveles)": {
        "costo_obra": 24500.0, "precio_venta_default": 58000.0, "soft_costs_pct": 0.16
    },
    "Torre Departamentos Media (4-8 niveles)": {
        "costo_obra": 18500.0, "precio_venta_default": 42000.0, "soft_costs_pct": 0.14
    },
    "Plaza Comercial / Strip Mall con Ancla (1-2 niveles)": {
        "costo_obra": 14500.0, "precio_venta_default": 38000.0, "soft_costs_pct": 0.12
    },
    "Torre de Oficinas Corporativas": {
        "costo_obra": 22500.0, "precio_venta_default": 52000.0, "soft_costs_pct": 0.15
    },
    "Nave Industrial / Bodega Logística de Última Milla": {
        "costo_obra": 9500.0,  "precio_venta_default": 23000.0, "soft_costs_pct": 0.10
    },
    "Personalizado (Ajuste Manual)": {
        "costo_obra": 19000.0, "precio_venta_default": 45000.0, "soft_costs_pct": 0.15
    }
}

# --- 5. CARGA DE MOTORES ESPACIALES (MANZANAS, DENUE Y COMPARABLES) ---
@st.cache_resource
def load_engines():
    base_dir = os.path.dirname(__file__)
    
    # 1. Base Manzanas
    path_mza = os.path.join(base_dir, 'data', 'master_jalisco_hbu_scout.parquet')
    if not os.path.exists(path_mza): path_mza = 'master_jalisco_hbu_scout.parquet'
    df_mza = pd.read_parquet(path_mza)
    tree_mza = KDTree(df_mza[['lat_centroide', 'lon_centroide']].values)

    # 2. Base DENUE
    path_poi = os.path.join(base_dir, 'data', 'jalisco_denue_scout.parquet')
    if not os.path.exists(path_poi): path_poi = 'jalisco_denue_scout.parquet'
    df_poi = pd.read_parquet(path_poi)

    # 3. Base Comparables de Mercado ($/m²)
    path_comp = os.path.join(base_dir, 'data', 'mercado_comparables_jalisco.parquet')
    if not os.path.exists(path_comp): path_comp = 'mercado_comparables_jalisco.parquet'
    df_comp = pd.read_parquet(path_comp) if os.path.exists(path_comp) else pd.DataFrame()
    tree_comp = KDTree(df_comp[['latitud', 'longitud']].values) if not df_comp.empty else None

    return df_mza, tree_mza, df_poi, df_comp, tree_comp

COLOR_MAP = {
    'TRANSPORTE_PUBLICO':      [2, 132, 199, 240],
    'RETAIL_ANCLA':            [234, 88, 12, 240],
    'SALUD_HOSPITALARIA':       [220, 38, 38, 240],
    'FARMACIA_SALUD_RETAIL':    [239, 68, 68, 240],
    'EDUCACION':                [37, 99, 235, 240],
    'FINANCIERO_SEGUROS':       [202, 138, 4, 240],
    'GASTRONOMIA':              [16, 185, 129, 240],
    'ENTRETENIMIENTO_FITNESS':  [5, 150, 105, 240],
    'HOTELERIA_TURISMO':        [147, 51, 234, 240],
    'CORPORATIVO_SERVICIOS':    [79, 70, 229, 240],
    'MEJORAMIENTO_HOGAR':       [120, 53, 15, 240],
    'AUTOMOTRIZ_COMBUSTIBLE':   [219, 39, 119, 240],
    'RETAIL_CONVENIENCIA':      [100, 116, 139, 200],
    'RETAIL_ESPECIALIZADO':     [190, 24, 93, 220],
    'SERVICIOS_PERSONALES':     [71, 85, 105, 180],
    'GOBIERNO_INSTITUCIONAL':   [30, 41, 59, 220],
    'LOGISTICA_LAST_MILE':      [37, 99, 235, 220]
}

# --- 6. GENERADOR DE MEMO DE COMITÉ HBU CON INTELIGENCIA ARTIFICIAL ---
def generar_memo_inversionista(data_mza, nearby_df, sup_lote, m2_desplante, m2_const, viv_max, pob_diu, arquetipo, gdv, costo_obra, residual_tierra, residual_m2, prototipo_nombre, es_predio_oficial, p_venta_usado, p_renta_mediana):
    groq_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", None))
    if not groq_key:
        return "⚠️ Motor de Inteligencia Artificial no autenticado. Configura la clave en Secrets."

    client = Groq(api_key=groq_key)

    top_pois = ""
    if not nearby_df.empty:
        top_pois = ", ".join([f"{r['nom_estab']} ({r['cat_hbu']} a {r['dist_m']:.0f}m)" for _, r in nearby_df.head(6).iterrows()])

    dot_status = str(data_mza.get('dot_acceso', 'SIN_COBERTURA'))
    pct_airbnb = float(pd.to_numeric(data_mza.get('pct_uso_temp'), errors='coerce') or 0.0) * 100
    niveles = int(data_mza.get('altura_max')) if pd.notna(data_mza.get('altura_max')) else 3
    origen_sup = "LEVANTAMIENTO CATASTRAL OFICIAL (POLÍGONO VECTORIAL)" if es_predio_oficial else "ESTIMACIÓN PARAMÉTRICA"

    contexto = f"""
    UBICACIÓN: {data_mza.get('NOM_MUN', 'Jalisco')} | CLAVE CATASTRAL: {data_mza.get('cvegeo_mza', 'S/D')}
    SUPERFICIE PREDIO: {sup_lote:,.2f} m² [{origen_sup}].
    PROTOTIPO PARAMÉTRICO CMIC: {prototipo_nombre}
    BENCHMARK DE MERCADO REAL EN ZONA:
    - Precio Venta Referencia Mercado: ${p_venta_usado:,.2f} MXN/m²
    - Renta Mensual Estimada Mercado: ${p_renta_mediana:,.2f} MXN/m²/mes
    NORMA LEGAL: Uso {data_mza.get('tipo_uso', 'S/D')}, CUS {data_mza.get('cus', 0)}, COS {data_mza.get('cos', 0)*100:.0f}%, Niveles máx: {niveles} pisos.
    VOLUMETRÍA CALCULADA: Huella PB: {m2_desplante:,.2f} m² | Construcción Total: {m2_const:,.2f} m² | Capacidad: {viv_max} viv.
    FINANZAS PRO-FORMA RESIDUALES:
    - Ventas Brutas Proyectadas (GDV): ${gdv:,.2f} MXN
    - Costo de Obra Directo + Soft Costs (CMIC): ${costo_obra:,.2f} MXN
    - Margen de Utilidad Objetivo Desarrollador: 20%
    - VALOR MÁXIMO DE OFERTA POR LA TIERRA: ${residual_tierra:,.2f} MXN (${residual_m2:,.2f} MXN/m² de terreno).
    CONECTIVIDAD DOT: {dot_status} (Transporte masivo a pie a < 500m).
    DEMOGRAFÍA: NSE {data_mza.get('nse_score', 0):.1f}/100, Población flotante diurna: {pob_diu:,} pers ({arquetipo}).
    MERCADO RENTAS CORTAS: {pct_airbnb:.1f}% de vivienda temporal.
    COMPETENCIA Y ANCLAS EN 800M: {top_pois if top_pois else 'Sin anclas identificadas'}.
    """

    prompt = f"""
    Eres el Chief Investment Officer (CIO) de un fondo institucional de desarrollo inmobiliario en México.
    Emite un MEMORÁNDUM DE PREFACTIBILIDAD Y MAYOR Y MEJOR USO (HBU) con base en la siguiente corrida territorial y de mercado:
    {contexto}

    Estructura tu dictamen en 3 secciones ejecutivas estrictas:
    1. 🎯 **Mayor y Mejor Uso (HBU) y Mix de Producto**: Evalúa el prototipo seleccionado ({prototipo_nombre}) y define la mezcla óptima (% m² locales comerciales en PB vs. % m² departamentos u oficinas arriba). Justifica con base en el precio de venta (${p_venta_usado:,.0f}/m²) y renta (${p_renta_mediana:,.0f}/m²) si conviene venta tradicional o pool de rentas.
    2. 💰 **Viabilidad Financiera y Precio Máximo de Oferta**: Explica por qué el fondo no debe pagar más de ${residual_m2:,.0f} MXN/m² de tierra para salvaguardar su margen del 20%, tomando como base las ventas proyectadas (${gdv:,.0f} MXN) y los costos paramétricos oficiales (${costo_obra:,.0f} MXN).
    3. ⚖️ **Riesgos Críticos y Estrategia de Licenciamiento**: Señala el impacto de la norma (CUS/COS), incentivo de transporte DOT y velocidad de absorción del mercado.
    Sé numéricamente implacable, sofisticado y conciso.
    """

    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.25,
            max_tokens=1000
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error al generar el memorándum: {e}"

# --- 7. INTERFAZ PRINCIPAL DEVELOPER ---
def main():
    st.markdown("<p style='text-align:center; color:#0A2540; letter-spacing:4px; margin-bottom:0; font-weight:800; font-size:1.35rem;'>HBU-NAVIGATOR | DEVELOPER & PE</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#64748B; font-size:0.8rem; margin-top:2px;'>PREFACTIBILIDAD CATASTRAL, MODELO RESIDUAL Y BENCHMARK $/M²</p>", unsafe_allow_html=True)
    st.markdown("<hr style='margin:10px auto; width:30%; opacity:0.2;'>", unsafe_allow_html=True)

    # --- PANEL DE BÚSQUEDA TRIPLE ---
    st.markdown('<div class="hud-panel">', unsafe_allow_html=True)
    st.markdown("<p style='color:#0A2540; font-weight:700; font-size:0.85rem; letter-spacing:1px; margin-bottom:8px;'>🔍 LOCALIZADOR DE PREDIO (DIRECCIÓN / CÓDIGO POSTAL / GPS)</p>", unsafe_allow_html=True)

    tab_dir, tab_gps, tab_coord = st.tabs(["📍 Búsqueda por Dirección / C.P.", "📡 Sensor GPS en Vivo", "⌨️ Coordenadas Manuales"])

    with tab_dir:
        col_txt, col_btn = st.columns([3, 1])
        with col_txt:
            direccion_input = st.text_input("Dirección o C.P.:", placeholder="ej. Circuito Madrigal 4295, Zapopan o 45110", label_visibility="collapsed")
        with col_btn:
            buscar_btn = st.button("BUSCAR PREDIO")

        if buscar_btn and direccion_input:
            with st.spinner("Localizando en cartografía urbana oficial..."):
                try:
                    geolocator = Nominatim(user_agent="propscan_developer_hbu")
                    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
                    query = direccion_input if "mexico" in direccion_input.lower() else f"{direccion_input}, Mexico"
                    res_loc = geocode(query)

                    if res_loc:
                        st.session_state.location = {"lat": res_loc.latitude, "lon": res_loc.longitude, "acc": 0.0}
                        st.success(f"📍 Predio localizado: {res_loc.address}")
                        st.rerun()
                    else:
                        st.error("❌ Dirección no localizada. Especifica municipio o código postal.")
                except Exception as e:
                    st.error(f"Error al geocodificar: {e}")

    with tab_gps:
        if st.button("🔄 SINCRONIZAR SENSOR GPS DEL CELULAR"):
            st.session_state.location = None
            st.rerun()

        if st.session_state.location is None:
            loc = get_geolocation()
            if loc and 'coords' in loc:
                st.session_state.location = {"lat": loc['coords']['latitude'], "lon": loc['coords']['longitude'], "acc": loc['coords'].get('accuracy', 1.0)}
                st.rerun()

    with tab_coord:
        c1, c2, c3 = st.columns([1.5, 1.5, 1])
        m_lat = c1.number_input("Latitud", value=20.7145, format="%.6f")
        m_lon = c2.number_input("Longitud", value=-103.4365, format="%.6f")
        if c3.button("FIJAR COORDENADAS"):
            st.session_state.location = {"lat": m_lat, "lon": m_lon, "acc": 0.0}
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.location:
        lat_now = st.session_state.location['lat']
        lon_now = st.session_state.location['lon']
        acc_now = st.session_state.location.get('acc', 1.0)

        df_mza, tree_mza, df_poi, df_comp, tree_comp = load_engines()

        # 1. Búsqueda de Manzana más cercana
        dist_deg, idx_mza = tree_mza.query(np.array([lat_now, lon_now]))
        data = df_mza.iloc[idx_mza]
        dist_m = dist_deg * 111139

        # 2. CONSULTA BAJO DEMANDA DEL PREDIO CATASTRAL EN VIVO (ZAPOPAN WFS)
        gdf_predio = consultar_predio_catastral(lat_now, lon_now)
        es_predio_oficial = gdf_predio is not None and not gdf_predio.empty

        sup_oficial_m2 = None
        if es_predio_oficial:
            try: sup_oficial_m2 = float(gdf_predio.iloc[0].get('shape_area', 0.0))
            except Exception: sup_oficial_m2 = None

        # 3. Consulta de Comercios en Anillo de 800m
        radio_800m_deg = 800.0 / 111139.0
        tree_poi = KDTree(df_poi[['latitud', 'longitud']].values)
        indices_800m = tree_poi.query_ball_point([lat_now, lon_now], r=radio_800m_deg)
        nearby_pois = df_poi.iloc[indices_800m].copy()

        if not nearby_pois.empty:
            d_lats = nearby_pois['latitud'].values - lat_now
            d_lons = (nearby_pois['longitud'].values - lon_now) * np.cos(np.radians(lat_now))
            nearby_pois['dist_m'] = np.sqrt(d_lats**2 + d_lons**2) * 111139
            nearby_pois = nearby_pois.sort_values(by='dist_m')
            nearby_pois['color'] = [COLOR_MAP.get(str(c), [100, 116, 139, 200]) for c in nearby_pois['cat_hbu']]
            nearby_pois['radius'] = 9

        # 4. CONSULTA DE COMPARABLES DE MERCADO ($/M²) EN 800M
        nearby_comp = pd.DataFrame()
        mediana_venta_m2 = 48000.0
        mediana_renta_m2 = 250.0
        dias_absorcion_med = 60

        if tree_comp is not None and not df_comp.empty:
            indices_comp = tree_comp.query_ball_point([lat_now, lon_now], r=radio_800m_deg)
            if indices_comp:
                nearby_comp = df_comp.iloc[indices_comp].copy()
                d_lats_c = nearby_comp['latitud'].values - lat_now
                d_lons_c = (nearby_comp['longitud'].values - lon_now) * np.cos(np.radians(lat_now))
                nearby_comp['dist_m'] = np.sqrt(d_lats_c**2 + d_lons_c**2) * 111139
                nearby_comp = nearby_comp.sort_values(by='dist_m')

                # Calcular medianas de la zona
                deptos_vta = nearby_comp[(nearby_comp['operacion'] == 'Venta') & (nearby_comp['tipo_propiedad'].isin(['Departamento', 'Casa']))]
                if not deptos_vta.empty:
                    mediana_venta_m2 = float(deptos_vta['precio_m2_mxn'].median())
                    dias_absorcion_med = int(deptos_vta['dias_mercado'].median())

                rentas_loc = nearby_comp[nearby_comp['operacion'] == 'Renta']
                if not rentas_loc.empty:
                    mediana_renta_m2 = float(rentas_loc['precio_m2_mxn'].median())

        # --- SECCIÓN SUPERIOR: TELEMETRÍA Y DIAL HBU ---
        col_izq, col_cen, col_der = st.columns([1, 1.2, 1])

        with col_izq:
            dot_label = str(data.get('dot_acceso', 'SIN_COBERTURA'))
            dot_color = "#166534" if "APTO" in dot_label else "#991B1B"
            dot_bg    = "#DCFCE7" if "APTO" in dot_label else "#FEE2E2"
            st.markdown(f"""
                <div class="hud-panel">
                    <p style="color:#64748B; font-size:0.75rem; margin:0; font-weight:600;">AUDITORÍA CATASTRAL Y LEGAL</p>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-top:4px;">
                        <span style="color:#0A2540; font-weight:800; font-size:0.9rem;">{'POLÍGONO PREDIO OFICIAL' if es_predio_oficial else 'PREFACTIBILIDAD'}</span>
                        <span class="status-pill">{'CATASTRO OK' if es_predio_oficial else 'MANZANA OK'}</span>
                    </div>
                    <hr style="opacity:0.15; margin:8px 0;">
                    <p style="color:#64748B; font-size:0.75rem; margin:0; font-weight:600;">INCENTIVO DOT</p>
                    <span style="display:inline-block; margin-top:3px; padding:2px 8px; border-radius:4px; font-family:'JetBrains Mono'; font-size:0.85rem; color:{dot_color}; background:{dot_bg}; font-weight:700;">{dot_label}</span>
                    <p style="color:#64748B; font-size:0.75rem; margin-top:8px; font-weight:600;">COMPARABLES EN 800M</p>
                    <p style="font-family:'JetBrains Mono'; margin:0; font-size:1.05rem; color:#0A2540; font-weight:700;">{len(nearby_comp)} Anuncios Activos</p>
                </div>
            """, unsafe_allow_html=True)

        with col_cen:
            opp_score = float(data.get('hbu_opportunity_score', 0.0))
            hbu_label = data.get('hbu_label', 'EVALUANDO')
            st.markdown(f"""
                <div class="dial-container">
                    <div class="central-dial">
                        <span class="dial-header">HBU OPPORTUNITY</span>
                        <span class="dial-value">{opp_score:.1f}</span>
                        <span class="dial-sub">{hbu_label}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with col_der:
            st.markdown(f"""
                <div class="hud-panel">
                    <p style="color:#64748B; font-size:0.75rem; margin:0; font-weight:600;">EXPEDIENTE DEL PREDIO</p>
                    <p style="color:#0A2540; font-weight:800; margin:2px 0 0 0; font-size:0.9rem;">{data.get('NOM_MUN', 'JALISCO')}</p>
                    <hr style="opacity:0.15; margin:8px 0;">
                    <p style="color:#64748B; font-size:0.75rem; margin:0; font-weight:600;">CLAVE CATASTRAL MANZANA</p>
                    <p style="font-family:'JetBrains Mono'; margin:0; font-size:0.85rem; color:#2563EB; font-weight:700;">{data.get('cvegeo_mza', 'S/D')}</p>
                    <p style="color:#64748B; font-size:0.75rem; margin-top:8px; font-weight:600;">COORDENADAS DE ANÁLISIS</p>
                    <p style="font-family:'JetBrains Mono'; margin:0; font-size:0.75rem; color:#475569;">{decimal_to_dms(lat_now, True)}</p>
                    <p style="font-family:'JetBrains Mono'; margin:0; font-size:0.75rem; color:#475569;">{decimal_to_dms(lon_now, False)}</p>
                </div>
            """, unsafe_allow_html=True)

        # --- SECCIÓN: INGRESO DE SUPERFICIE REAL Y MODELO FINANCIERO RESIDUAL ---
        st.markdown('<div class="hud-panel">', unsafe_allow_html=True)
        st.markdown("<p style='color:#0A2540; font-weight:800; font-size:0.9rem; letter-spacing:1px; margin-bottom:8px;'>💰 MODELO FINANCIERO RESIDUAL DE TIERRA (CALIBRADO CON MERCADO VIVO)</p>", unsafe_allow_html=True)

        if es_predio_oficial and sup_oficial_m2:
            st.markdown(f"""
                <div class="predio-badge">
                    🏛️ Polígono Catastral Oficial Detectado: Superficie Registrada = {sup_oficial_m2:,.2f} m²
                </div>
            """, unsafe_allow_html=True)

        prototipo_seleccionado = st.selectbox(
            "🏗️ Selecciona el Prototipo de Construcción a Evaluar (Costos CMIC):",
            list(PARAMETRICOS_CMIC.keys()),
            index=0
        )

        datos_prototipo = PARAMETRICOS_CMIC[prototipo_seleccionado]

        col_f1, col_f2, col_f3 = st.columns(3)

        with col_f1:
            if es_predio_oficial and sup_oficial_m2 and sup_oficial_m2 > 10:
                sup_default = sup_oficial_m2
            else:
                sup_default = float(data.get('superficie_terreno_m2', 1000.0))
                if sup_default > 40000 or sup_default <= 0: sup_default = 1000.0
            
            sup_lote = st.number_input(
                "Superficie del Terreno (m²):",
                min_value=50.0, max_value=1000000.0, value=round(sup_default, 2), step=50.0
            )

        cus_val = float(data.get('cus', 1.8))
        cos_val = float(data.get('cos', 0.6))
        densidad_ha = float(data.get('densidad_ha', 60.0))

        m2_desplante_real = sup_lote * cos_val
        m2_construccion_real = sup_lote * cus_val
        viv_max_real = int(round((sup_lote / 10000.0) * densidad_ha)) if densidad_ha > 0 else 0

        with col_f2:
            # Pre-llenado inteligente con la mediana de comparables de mercado si existe
            precio_sugerido = mediana_venta_m2 if mediana_venta_m2 > 10000 else datos_prototipo["precio_venta_default"]
            precio_venta_m2 = st.number_input(
                "Precio Venta Proyectado ($/m² const):",
                min_value=10000.0, max_value=250000.0, value=round(precio_sugerido, -2), step=1000.0,
                help="Pre-llenado automáticamente con la mediana de anuncios de mercado en un radio de 800m."
            )

        with col_f3:
            costo_obra_m2 = st.number_input(
                "Costo de Obra Directo ($/m² const):",
                min_value=6000.0, max_value=120000.0, value=datos_prototipo["costo_obra"], step=500.0,
                help="Costo paramétrico directo de edificación según tabulador CMIC."
            )

        # CÁLCULOS RESIDUALES FINANCIEROS
        soft_costs_factor = datos_prototipo["soft_costs_pct"]
        costo_total_obra_directa = m2_construccion_real * costo_obra_m2
        costo_soft_costs = costo_total_obra_directa * soft_costs_factor
        costo_total_desarrollo = costo_total_obra_directa + costo_soft_costs

        gdv_total = m2_construccion_real * 0.85 * precio_venta_m2
        utilidad_desarrollador_20pct = gdv_total * 0.20

        residual_tierra = max(0.0, gdv_total - costo_total_desarrollo - utilidad_desarrollador_20pct)
        residual_m2_tierra = residual_tierra / max(sup_lote, 1.0)

        st.markdown(f"""
            <div style="background: #FFFFFF; padding: 14px; border-radius: 6px; margin-top: 10px; border: 1px solid #CBD5E1; border-left: 5px solid #0A2540;">
                <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
                    <div>
                        <span style="color: #64748B; font-size: 0.72rem; font-weight:600;">VENTAS TOTALES (GDV)</span><br>
                        <span style="font-family:'JetBrains Mono'; font-weight:800; font-size: 1.15rem; color:#0A2540;">${gdv_total:,.0f} MXN</span>
                    </div>
                    <div>
                        <span style="color: #64748B; font-size: 0.72rem; font-weight:600;">OBRA + SOFT COSTS ({soft_costs_factor*100:.0f}%)</span><br>
                        <span style="font-family:'JetBrains Mono'; font-weight:800; font-size: 1.15rem; color:#DC2626;">${costo_total_desarrollo:,.0f} MXN</span>
                    </div>
                    <div>
                        <span style="color: #64748B; font-size: 0.72rem; font-weight:600;">VALOR MÁXIMO OFERTA (TIERRA)</span><br>
                        <span style="font-family:'JetBrains Mono'; font-weight:800; font-size: 1.25rem; color:#16A34A;">${residual_tierra:,.0f} MXN</span>
                    </div>
                    <div>
                        <span style="color: #64748B; font-size: 0.72rem; font-weight:600;">VALOR RESIDUAL $/M² TERRENO</span><br>
                        <span style="font-family:'JetBrains Mono'; font-weight:800; font-size: 1.25rem; color:#16A34A;">${residual_m2_tierra:,.0f} MXN/m²</span>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        registrar_telemetria("CALCULO_RESIDUAL_DEVELOPER", data, sup_lote, m2_construccion_real, residual_m2_tierra)

        # --- SECCIÓN: BENCHMARK DE PRECIOS Y COMPARABLES ACTIVOS ---
        st.markdown('<div class="hud-panel">', unsafe_allow_html=True)
        st.markdown("<p style='color:#0A2540; font-weight:800; font-size:0.85rem; letter-spacing:1px; margin-bottom:8px;'>📊 BENCHMARK DE PRECIOS Y TESTIGOS DE MERCADO (< 800M)</p>", unsafe_allow_html=True)

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Mediana Venta Departamentos", f"${mediana_venta_m2:,.0f} / m²", "Mercado Real")
        b2.metric("Mediana Renta Mensual", f"${mediana_renta_m2:,.0f} / m²", "Pool / Rentas")
        b3.metric("Oferta Activa en Radio", f"{len(nearby_comp)} Propiedades", "Muestra 800m")
        b4.metric("Absorción Estimada", f"{dias_absorcion_med} Días", "Velocidad Venta")

        if not nearby_comp.empty:
            st.markdown("<p style='color:#64748B; font-size:0.75rem; font-weight:600; margin-top:8px;'>PROPIEDADES TESTIGO MÁS CERCANAS EN EL ENTORNO:</p>", unsafe_allow_html=True)
            cols_t = ['titulo', 'tipo_propiedad', 'operacion', 'precio_mxn', 'precio_m2_mxn', 'dist_m', 'dias_mercado']
            df_tabla_comp = nearby_comp[cols_t].head(4).copy()
            df_tabla_comp.columns = ['Propiedad Testigo', 'Tipo', 'Operación', 'Precio Total', 'Precio $/m²', 'Distancia', 'Días']
            df_tabla_comp['Precio Total'] = df_tabla_comp['Precio Total'].apply(lambda x: f"${x:,.0f} MXN")
            df_tabla_comp['Precio $/m²'] = df_tabla_comp['Precio $/m²'].apply(lambda x: f"${x:,.0f}/m²")
            df_tabla_comp['Distancia'] = df_tabla_comp['Distancia'].apply(lambda x: f"{x:.0f} m")
            df_tabla_comp['Días'] = df_tabla_comp['Días'].apply(lambda x: f"{x} d")
            st.dataframe(df_tabla_comp, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # --- ESTIMACIÓN DE POBLACIÓN FLOTANTE ---
        pob_res = int(data.get('POBTOT', 0)) if pd.notna(data.get('POBTOT')) else 0
        EMPLEO_POR_GIRO = {
            'RETAIL_ANCLA': 120.0, 'SALUD_HOSPITALARIA': 80.0, 'CORPORATIVO_SERVICIOS': 25.0,
            'EDUCACION': 35.0, 'HOTELERIA_TURISMO': 25.0, 'FINANCIERO_SEGUROS': 12.0,
            'FARMACIA_SALUD_RETAIL': 8.0, 'GASTRONOMIA': 8.0, 'MEJORAMIENTO_HOGAR': 10.0,
            'AUTOMOTRIZ_COMBUSTIBLE': 6.0, 'LOGISTICA_LAST_MILE': 15.0, 'ENTRETENIMIENTO_FITNESS': 8.0,
            'RETAIL_CONVENIENCIA': 3.0, 'RETAIL_ESPECIALIZADO': 3.5, 'SERVICIOS_PERSONALES': 2.5,
            'GOBIERNO_INSTITUCIONAL': 20.0, 'TRANSPORTE_PUBLICO': 0.0
        }
        VISITANTES_POR_GIRO = {
            'RETAIL_ANCLA': 450.0, 'GASTRONOMIA': 45.0, 'FARMACIA_SALUD_RETAIL': 35.0,
            'FINANCIERO_SEGUROS': 50.0, 'SALUD_HOSPITALARIA': 180.0, 'ENTRETENIMIENTO_FITNESS': 40.0,
            'RETAIL_CONVENIENCIA': 25.0, 'RETAIL_ESPECIALIZADO': 15.0
        }

        if not nearby_pois.empty:
            emp_h3 = sum([EMPLEO_POR_GIRO.get(str(c), 3.0) for c in nearby_pois['cat_hbu']])
            vis_h3 = sum([VISITANTES_POR_GIRO.get(str(c), 0.0) for c in nearby_pois['cat_hbu']])
        else:
            emp_h3, vis_h3 = 0.0, 0.0

        pob_diu = int((pob_res * 0.45) + emp_h3 + vis_h3)
        ratio_f = pob_diu / max(pob_res, 1)

        if ratio_f > 2.0 or (pob_diu > 400 and ratio_f > 1.3):
            arquetipo = 'POLO DE EMPLEO / COMERCIAL'
            badge_color = "#16A34A"
            badge_bg = "#DCFCE7"
        elif ratio_f >= 0.8:
            arquetipo = 'ZONA MIXTA 24/7'
            badge_color = "#2563EB"
            badge_bg = "#DBEAFE"
        else:
            arquetipo = 'ZONA DORMITORIO'
            badge_color = "#D97706"
            badge_bg = "#FEF3C7"

        # --- BOTÓN DE MEMORÁNDUM CON IA ---
        if st.button("🏛️ GENERAR MEMORÁNDUM DE COMITÉ DE INVERSIÓN (IA GENERATIVA)"):
            registrar_telemetria("CLICK_MEMO_IA", data, sup_lote, m2_construccion_real, residual_m2_tierra)
            with st.spinner("Compilando corrida pro-forma y memorándum de inversión con Inteligencia Artificial..."):
                memo = generar_memo_inversionista(
                    data, nearby_pois, sup_lote, m2_desplante_real, m2_construccion_real,
                    viv_max_real, pob_diu, arquetipo, gdv_total, costo_total_desarrollo, residual_tierra, residual_m2_tierra, prototipo_seleccionado, es_predio_oficial, precio_venta_m2, mediana_renta_m2
                )
                st.markdown(f"""
                    <div class="memo-panel">
                        <p style="color:#166534; font-weight:800; letter-spacing:1px; margin-bottom:8px; font-size:0.95rem;">🏛️ MEMORÁNDUM DE PREFACTIBILIDAD Y MAYOR Y MEJOR USO (HBU)</p>
                        <div style="font-size:0.92rem; line-height:1.6; color:#1E293B;">
                            {memo}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        # --- PANELES DE APOYO TÉCNICO ---
        r_col1, r_col2 = st.columns(2)

        with r_col1:
            st.markdown('<div class="hud-panel">', unsafe_allow_html=True)
            st.markdown("<p style='color:#0A2540; font-weight:700; font-size:0.85rem; letter-spacing:1px; border-bottom:1px solid #E2E8F0; padding-bottom:5px; margin-bottom:8px;'>⚖️ CAPACIDAD NORMATIVA AUTORIZADA</p>", unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m1.metric("Zonificación", f"{data.get('tipo_uso', 'S/D')}")
            alt_val = data.get('altura_max')
            niveles = int(alt_val) if pd.notna(alt_val) else 3
            m2.metric("Límite de Niveles", f"{niveles} Pisos")

            m3, m4 = st.columns(2)
            m3.metric("Factor CUS", f"{cus_val:.2f}x")
            m4.metric("Ocupación COS", f"{cos_val*100:.0f}%")

            m5, m6 = st.columns(2)
            m5.metric("M² Construcción Máx", f"{m2_construccion_real:,.0f} m²")
            m6.metric("Capacidad Máx Viv", f"{viv_max_real} Unidades")
            st.markdown('</div>', unsafe_allow_html=True)

        with r_col2:
            st.markdown('<div class="hud-panel">', unsafe_allow_html=True)
            st.markdown("<p style='color:#0A2540; font-weight:700; font-size:0.85rem; letter-spacing:1px; border-bottom:1px solid #E2E8F0; padding-bottom:5px; margin-bottom:8px;'>👥 MERCADO Y DINÁMICA DE ENTORNO</p>", unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            c1.metric("Población Residente", f"{pob_res:,} hab")
            c2.metric("Población Diurna", f"{pob_diu:,} pers", f"{ratio_f:.1f}x Flujo")

            c3, c4 = st.columns(2)
            c3.metric("Poder Adquisitivo (NSE)", f"{data.get('nse_score', 0.0):.1f} / 100")
            pct_airbnb = float(pd.to_numeric(data.get('pct_uso_temp'), errors='coerce') or 0.0) * 100
            c4.metric("Tasa Renta Corta / Airbnb", f"{pct_airbnb:.1f}%")

            st.markdown(f"""
                <div style="margin-top: 8px; text-align: center;">
                    <span class="archetype-pill" style="background: {badge_bg}; border: 1px solid {badge_color}; color: {badge_color};">
                        ● ARQUETIPO: {arquetipo}
                    </span>
                </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # --- SECCIÓN INFERIOR: MAPA CON POLÍGONO DEL PREDIO CATASTRAL + ANILLOS 400M Y 800M ---
        st.markdown('<div class="hud-panel">', unsafe_allow_html=True)
        st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                <span style="color:#0A2540; font-weight:700; font-size:0.85rem; letter-spacing:1px;'>🗺️ CARTOGRAFÍA: POLÍGONO DEL PREDIO + ANILLOS CAMINABLES 400M Y 800M</span>
                <span style="color:#2563EB; font-size:0.75rem; font-family:'JetBrains Mono'; font-weight:600;">{len(nearby_pois)} NODOS EN PANTALLA</span>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="display: flex; flex-wrap: wrap; gap: 6px; margin: 6px 0 10px 0; justify-content: center;">
            <span style="background: #E0F2FE; border: 1px solid #0284C7; color: #0284C7; padding: 2px 7px; border-radius: 4px; font-size: 0.7rem; font-weight: 700;">● TRANSPORTE</span>
            <span style="background: #FFEDD5; border: 1px solid #EA580C; color: #EA580C; padding: 2px 7px; border-radius: 4px; font-size: 0.7rem; font-weight: 700;">● RETAIL ANCLA</span>
            <span style="background: #FEE2E2; border: 1px solid #DC2626; color: #DC2626; padding: 2px 7px; border-radius: 4px; font-size: 0.7rem; font-weight: 700;">● SALUD</span>
            <span style="background: #FEE2E2; border: 1px solid #EF4444; color: #EF4444; padding: 2px 7px; border-radius: 4px; font-size: 0.7rem; font-weight: 700;">● FARMACIAS</span>
            <span style="background: #D1FAE5; border: 1px solid #059669; color: #059669; padding: 2px 7px; border-radius: 4px; font-size: 0.7rem; font-weight: 700;">● PARQUES / FITNESS</span>
            <span style="background: #DBEAFE; border: 1px solid #2563EB; color: #2563EB; padding: 2px 7px; border-radius: 4px; font-size: 0.7rem; font-weight: 700;">● EDUCACIÓN</span>
            <span style="background: #FEF9C3; border: 1px solid #CA8A04; color: #CA8A04; padding: 2px 7px; border-radius: 4px; font-size: 0.7rem; font-weight: 700;">● FINANCIERO</span>
            <span style="background: #D1FAE5; border: 1px solid #10B981; color: #10B981; padding: 2px 7px; border-radius: 4px; font-size: 0.7rem; font-weight: 700;">● GASTRONOMÍA</span>
        </div>
        """, unsafe_allow_html=True)

        node_lat = float(data['lat_centroide'])
        node_lon = float(data['lon_centroide'])
        u_lat = float(lat_now)
        u_lon = float(lon_now)

        base_points = pd.DataFrame([
            {"name": "Ubicación Scout", "lat": u_lat, "lon": u_lon, "color": [37, 99, 235, 255], "radius": 13},
            {"name": "Predio Objetivo HBU", "lat": node_lat, "lon": node_lon, "color": [10, 37, 64, 255], "radius": 18}
        ])

        line_data = pd.DataFrame([{"start": [u_lon, u_lat], "end": [node_lon, node_lat]}])

        rings_data = pd.DataFrame([
            {"lon": node_lon, "lat": node_lat, "radius": 400, "color": [10, 37, 64, 20], "border": [10, 37, 64, 200]},
            {"lon": node_lon, "lat": node_lat, "radius": 800, "color": [37, 99, 235, 12], "border": [37, 99, 235, 160]}
        ])

        view_state = pdk.ViewState(
            latitude=(u_lat + node_lat) / 2,
            longitude=(u_lon + node_lon) / 2,
            zoom=15.0,
            pitch=25
        )

        layers = [
            pdk.Layer(
                "ScatterplotLayer",
                rings_data,
                get_position=["lon", "lat"],
                get_radius="radius",
                get_fill_color="color",
                get_line_color="border",
                line_width_min_pixels=2,
                stroked=True,
                filled=True,
                pickable=False
            ),
            pdk.Layer(
                "LineLayer",
                line_data,
                get_source_position="start",
                get_target_position="end",
                get_color=[10, 37, 64, 220],
                get_width=3
            ),
            pdk.Layer(
                "ScatterplotLayer",
                base_points,
                get_position=["lon", "lat"],
                get_color="color",
                get_radius="radius",
                pickable=True
            )
        ]

        # Polígono Catastral Oficial si existe
        if es_predio_oficial:
            try:
                poly_geom = gdf_predio.geometry.iloc[0]
                if poly_geom.geom_type == 'MultiPolygon':
                    poly_geom = list(poly_geom.geoms)[0]
                coords_predio = [[coord[0], coord[1]] for coord in poly_geom.exterior.coords]
                df_poly_predio = pd.DataFrame([{'polygon': coords_predio, 'name': 'LOTE CATASTRAL OFICIAL'}])

                layer_predio_polygon = pdk.Layer(
                    "PolygonLayer",
                    df_poly_predio,
                    get_polygon="polygon",
                    get_fill_color=[37, 99, 235, 45],
                    get_line_color=[10, 37, 64, 255],
                    get_line_width=3,
                    line_width_min_pixels=2,
                    filled=True,
                    stroked=True,
                    pickable=True
                )
                layers.insert(0, layer_predio_polygon)
            except Exception:
                pass

        # Capa de Comercios y Paradas
        if not nearby_pois.empty:
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                nearby_pois,
                get_position=["longitud", "latitud"],
                get_color="color",
                get_radius="radius",
                radius_min_pixels=5,
                radius_max_pixels=11,
                pickable=True
            ))

        st.pydeck_chart(
            pdk.Deck(
                layers=layers,
                initial_view_state=view_state,
                map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                tooltip={"html": "<b>{nom_estab}</b><br/>{cat_hbu}", "style": {"color": "#1E293B", "backgroundColor": "#FFFFFF", "border": "1px solid #CBD5E1"}}
            ),
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # --- SECCIÓN: RADAR DE PROXIMIDAD ---
        st.markdown('<div class="hud-panel">', unsafe_allow_html=True)
        st.markdown("<p style='color:#0A2540; font-weight:700; font-size:0.85rem; letter-spacing:1px; margin-bottom:8px;'>🏬 EQUIPAMIENTO Y COMPETENCIA DIRECTA (< 800M)</p>", unsafe_allow_html=True)

        if not nearby_pois.empty:
            cols_show = nearby_pois[['cat_hbu', 'nom_estab', 'dist_m']].head(12).copy()
            cols_show.columns = ['Categoría', 'Establecimiento / Parada', 'Distancia']
            cols_show['Distancia'] = cols_show['Distancia'].apply(lambda x: f"{x:.0f} m ({x/80:.1f} min)")
            st.dataframe(cols_show, use_container_width=True, hide_index=True)
        else:
            st.caption("No se detectaron comercios ancla registrados en el radio analizado.")
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
