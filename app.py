"""
Rotas & Acidentes — Streamlit App
==================================
O arquivo .parquet fica armazenado no servidor (pasta `data/`).
Qualquer visitante pode consultar rotas sem precisar fazer upload ou
configurar nada. O administrador coloca o arquivo lá uma única vez.

Estrutura esperada do parquet
------------------------------
Obrigatórias : latitude (float), longitude (float)
Opcional     : qualquer coluna com "tipo", "type", "acidente" ou "accident"
               qualquer coluna com "data", "date"
               qualquer coluna com "morto", "dead", "obito"
               qualquer coluna com "ferido", "injured", "wound"
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Configuração da página
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rotas & Acidentes",
    page_icon="🗺️",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────
OSRM_BASE    = "http://router.project-osrm.org/route/v1/driving"
NOMINATIM    = "https://nominatim.openstreetmap.org/search"
DATA_DIR     = Path("data/processed")
PARQUET_FILE = DATA_DIR / "acidentes.parquet"   # ← arquivo do servidor

ACCIDENT_COLORS = {
    "Colisão":        "red",
    "Atropelamento":  "orange",
    "Capotamento":    "purple",
    "Queda":          "blue",
    "Incêndio":       "darkred",
    "Saída de Pista": "cadetblue",
    "Engavetamento":  "darkblue",
    "Outros":         "gray",
}

# ─────────────────────────────────────────────────────────────────────────────
# Session state — inicializa uma única vez por sessão de usuário
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS = {
    "route_coords": None,
    "accidents_df": None,
    "has_result":   False,
    "error_msg":    None,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# Dados — garante que o parquet esteja disponível
# ─────────────────────────────────────────────────────────────────────────────

def _create_sample_parquet() -> None:
    """Cria um parquet de exemplo caso o arquivo real não exista."""
    DATA_DIR.mkdir(exist_ok=True)
    np.random.seed(42)
    n = 500
    tipos = list(ACCIDENT_COLORS.keys())
    df = pd.DataFrame({
        "latitude":      np.random.uniform(-23.70, -23.40, n),
        "longitude":     np.random.uniform(-46.85, -46.35, n),
        "tipo_acidente": np.random.choice(tipos, n),
        "data":          pd.date_range("2020-01-01", periods=n, freq="D"),
        "mortos":        np.random.randint(0, 5, n),
        "feridos":       np.random.randint(0, 20, n),
    })
    df.to_parquet(PARQUET_FILE, index=False)


@st.cache_resource(show_spinner=False)
def _load_parquet_to_duckdb() -> tuple[duckdb.DuckDBPyConnection, dict]:
    """
    Carrega o parquet em um banco DuckDB in-memory compartilhado
    entre todos os usuários (cache_resource = uma instância por processo).
    Retorna a conexão e o mapeamento de colunas detectado.
    """
    if not PARQUET_FILE.exists():
        _create_sample_parquet()

    con = duckdb.connect()
    con.execute(f"CREATE TABLE acidentes AS SELECT * FROM read_parquet('{PARQUET_FILE}')")

    schema = con.execute("DESCRIBE acidentes").df()
    cols   = {c.lower(): c for c in schema["column_name"].tolist()}

    col_map = {}
    col_map["lat"]  = next((orig for lo, orig in cols.items() if "lat"  in lo), None)
    col_map["lon"]  = next((orig for lo, orig in cols.items() if "lon"  in lo or "lng" in lo), None)
    col_map["tipo"] = next(
        (orig for lo, orig in cols.items()
         if "tipo" in lo or "type" in lo or "acidente" in lo or "accident" in lo),
        None,
    )
    col_map["data"]   = next((orig for lo, orig in cols.items() if "data" in lo or "date" in lo), None)
    col_map["mortos"] = next((orig for lo, orig in cols.items() if "morto" in lo or "dead" in lo or "obito" in lo), None)
    col_map["feridos"]= next((orig for lo, orig in cols.items() if "ferido" in lo or "injur" in lo or "wound" in lo), None)

    total = con.execute("SELECT COUNT(*) FROM acidentes").fetchone()[0]
    return con, col_map, total


# ─────────────────────────────────────────────────────────────────────────────
# Funções de consulta
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def geocode(place: str):
    params  = {"q": place, "format": "json", "limit": 1}
    headers = {"User-Agent": "streamlit-rotas-acidentes/2.0"}
    r = requests.get(NOMINATIM, params=params, headers=headers, timeout=10)
    r.raise_for_status()
    results = r.json()
    return (float(results[0]["lat"]), float(results[0]["lon"])) if results else None


@st.cache_data(show_spinner=False)
def get_route(origin_ll: tuple, dest_ll: tuple):
    url = (
        f"{OSRM_BASE}"
        f"/{origin_ll[1]},{origin_ll[0]}"
        f";{dest_ll[1]},{dest_ll[0]}"
        f"?overview=full&geometries=geojson"
    )
    r    = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()
    if data.get("code") != "Ok":
        return None, None

    coords_lonlat = data["routes"][0]["geometry"]["coordinates"]
    coords = [[c[1], c[0]] for c in coords_lonlat]
    lats   = [c[0] for c in coords]
    lons   = [c[1] for c in coords]
    bbox   = (min(lats), max(lats), min(lons), max(lons))
    return coords, bbox


def query_accidents(bbox: tuple, margin_km: float) -> pd.DataFrame:
    con, col_map, _ = _load_parquet_to_duckdb()

    if not col_map["lat"] or not col_map["lon"]:
        return pd.DataFrame()

    margin     = margin_km / 111.0
    min_lat    = bbox[0] - margin
    max_lat    = bbox[1] + margin
    min_lon    = bbox[2] - margin
    max_lon    = bbox[3] + margin

    lat  = col_map["lat"]
    lon  = col_map["lon"]
    tipo = col_map["tipo"]

    selects = [f'"{lat}" AS latitude', f'"{lon}" AS longitude']
    selects.append(f'"{tipo}" AS tipo_acidente' if tipo else "'Outros' AS tipo_acidente")

    for alias, key in [("data_acidente", "data"), ("mortos", "mortos"), ("feridos", "feridos")]:
        if col_map.get(key):
            selects.append(f'"{col_map[key]}" AS {alias}')

    sql = f"""
        SELECT {', '.join(selects)}
        FROM acidentes
        WHERE "{lat}" BETWEEN {min_lat} AND {max_lat}
          AND "{lon}" BETWEEN {min_lon} AND {max_lon}
    """
    return con.execute(sql).df()


# ─────────────────────────────────────────────────────────────────────────────
# Mapa
# ─────────────────────────────────────────────────────────────────────────────

def build_map(route_coords: list, accidents_df: pd.DataFrame) -> folium.Map:
    mid = len(route_coords) // 2
    m = folium.Map(location=route_coords[mid], zoom_start=12, tiles="CartoDB positron")

    # Rota
    folium.PolyLine(route_coords, color="#2563EB", weight=5,
                    opacity=0.85, tooltip="Rota calculada").add_to(m)
    folium.Marker(
        route_coords[0], tooltip="Origem",
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
    ).add_to(m)
    folium.Marker(
        route_coords[-1], tooltip="Destino",
        icon=folium.Icon(color="red", icon="flag", prefix="fa"),
    ).add_to(m)

    # Acidentes
    if not accidents_df.empty:
        grupos: dict[str, folium.FeatureGroup] = {}
        for tipo in sorted(accidents_df["tipo_acidente"].dropna().unique()):
            label = str(tipo)
            grupos[label] = folium.FeatureGroup(name=f"● {label}", show=True)

        for _, row in accidents_df.iterrows():
            tipo  = str(row.get("tipo_acidente", "Outros"))
            color = ACCIDENT_COLORS.get(tipo, "gray")

            html = f"<b>{tipo}</b><br>Lat {row['latitude']:.5f} | Lon {row['longitude']:.5f}"
            if "data_acidente" in row and pd.notna(row["data_acidente"]):
                html += f"<br>Data: {str(row['data_acidente'])[:10]}"
            if "mortos"  in row and pd.notna(row["mortos"]):
                html += f"<br>Mortos: {int(row['mortos'])}"
            if "feridos" in row and pd.notna(row["feridos"]):
                html += f"<br>Feridos: {int(row['feridos'])}"

            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=6, color=color, fill=True,
                fill_color=color, fill_opacity=0.75,
                popup=folium.Popup(html, max_width=230),
                tooltip=tipo,
            ).add_to(grupos.setdefault(tipo, folium.FeatureGroup(name=f"● {tipo}", show=True)))

        for fg in grupos.values():
            fg.add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)

    return m


# ─────────────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────────────

# Pré-carrega o banco (só executa uma vez por processo do servidor)
try:
    _con, _col_map, _total_records = _load_parquet_to_duckdb()
    _data_ok = True
except Exception as e:
    _data_ok = False
    _load_error = str(e)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🗂️ Base de dados")
    if _data_ok:
        st.success(f"✅ Parquet carregado\n\n**{_total_records:,}** registros disponíveis")
        tipos_disponiveis = (
            _con.execute("SELECT DISTINCT tipo_acidente FROM acidentes ORDER BY 1")
            .df()["tipo_acidente"]
            .dropna()
            .tolist()
            if _col_map.get("tipo") else []
        )
        if tipos_disponiveis:
            st.caption("Tipos de acidente presentes:")
            for t in tipos_disponiveis:
                cor = ACCIDENT_COLORS.get(str(t), "gray")
                st.markdown(f"<span style='color:{cor}'>●</span> {t}", unsafe_allow_html=True)
    else:
        st.error(f"Erro ao carregar dados:\n{_load_error}")

    st.markdown("---")
    margin_km = st.slider(
        "Margem do Bounding Box (km)", min_value=1, max_value=5, value=2,
        help="Expande a busca de acidentes em torno da rota calculada.",
    )

    st.markdown("---")
    st.markdown(
        "**Como usar**\n"
        "1. Digite origem e destino\n"
        "2. Clique em **Calcular Rota**\n"
        "3. Explore o mapa — use a legenda para filtrar tipos de acidente"
    )

    if st.session_state.has_result:
        st.markdown("---")
        if st.button("🔄 Nova consulta", use_container_width=True):
            for k, v in _DEFAULTS.items():
                st.session_state[k] = v
            st.rerun()

# ── Cabeçalho ─────────────────────────────────────────────────────────────────
st.title("🗺️ Rotas & Pontos de Acidente")
st.caption("Informe a origem e o destino para calcular a rota e visualizar os acidentes registrados ao longo do caminho.")

# ── Formulário de entrada ─────────────────────────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    origin_input = st.text_input("📍 Local de saída",
                                  placeholder="Ex: Aeroporto de Congonhas, São Paulo")
with c2:
    dest_input = st.text_input("🏁 Destino",
                                placeholder="Ex: Avenida Paulista, São Paulo")

calcular = st.button("🔍 Calcular Rota", type="primary",
                      use_container_width=True, disabled=not _data_ok)

# ── Lógica do botão ───────────────────────────────────────────────────────────
if calcular:
    st.session_state.error_msg = None

    if not origin_input.strip() or not dest_input.strip():
        st.warning("Preencha origem e destino antes de calcular.")
        st.stop()

    with st.spinner("Localizando endereços..."):
        origin_ll = geocode(origin_input)
        dest_ll   = geocode(dest_input)

    if not origin_ll:
        st.session_state.error_msg = f"Endereço não encontrado: **{origin_input}**"
        st.rerun()
    if not dest_ll:
        st.session_state.error_msg = f"Endereço não encontrado: **{dest_input}**"
        st.rerun()

    with st.spinner("Calculando rota..."):
        route_coords, bbox = get_route(origin_ll, dest_ll)

    if route_coords is None:
        st.session_state.error_msg = "Não foi possível traçar a rota. Verifique os endereços."
        st.rerun()

    with st.spinner("Consultando acidentes na rota..."):
        accidents_df = query_accidents(bbox, margin_km)

    st.session_state.route_coords = route_coords
    st.session_state.accidents_df = accidents_df
    st.session_state.has_result   = True
    st.rerun()

# ── Renderização persistente (lê do session_state) ────────────────────────────
if st.session_state.error_msg:
    st.error(st.session_state.error_msg)

if st.session_state.has_result:
    route_coords = st.session_state.route_coords
    accidents_df = st.session_state.accidents_df
    total_acc    = len(accidents_df)

    # Métricas
    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("📍 Pontos calculados na rota", f"{len(route_coords):,}")
    m2.metric("⚠️ Acidentes na região", f"{total_acc:,}")
    if total_acc > 0 and "tipo_acidente" in accidents_df.columns:
        top = accidents_df["tipo_acidente"].value_counts().idxmax()
        m3.metric("🔝 Tipo mais frequente", top)

    # Mapa
    st.markdown("### 🗺️ Mapa")
    mapa = build_map(route_coords, accidents_df)
    st_folium(mapa, use_container_width=True, height=600)

    # Tabela e gráfico
    if total_acc > 0:
        with st.expander(f"📋 Tabela de acidentes ({total_acc} registros)", expanded=False):
            st.dataframe(accidents_df, use_container_width=True)

        st.markdown("### 📊 Distribuição por Tipo")
        tc = accidents_df["tipo_acidente"].value_counts().reset_index()
        tc.columns = ["Tipo", "Quantidade"]
        st.bar_chart(tc.set_index("Tipo"))
    else:
        st.success("✅ Nenhum acidente registrado encontrado neste trecho.")

elif not st.session_state.error_msg:
    st.info("👆 Preencha os campos acima e clique em **Calcular Rota** para começar.")
    m = folium.Map(location=[-15.78, -47.93], zoom_start=4, tiles="CartoDB positron")
    st_folium(m, use_container_width=True, height=500)