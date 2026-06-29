"""
Rotas & Acidentes — Streamlit App
==================================
"""

import streamlit as st
import folium
from folium.plugins import MarkerCluster
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

# Paleta para os 17 tipos oficiais — hex direto, sem intermediário Leaflet
ACCIDENT_COLORS = {
    "Tombamento":                        "#e67e22",  # laranja
    "Colisão frontal":                   "#c0392b",  # vermelho escuro
    "Colisão traseira":                  "#e74c3c",  # vermelho
    "Saída de leito carroçável":         "#16a085",  # verde-teal
    "Incêndio":                          "#d35400",  # laranja queimado
    "Colisão com objeto":                "#8e44ad",  # roxo
    "Colisão lateral mesmo sentido":     "#f39c12",  # amarelo-âmbar
    "Colisão lateral sentido oposto":    "#e91e63",  # rosa
    "Queda de ocupante de veículo":      "#2980b9",  # azul médio
    "Engavetamento":                     "#1a5276",  # azul marinho
    "Derramamento de carga":             "#795548",  # marrom
    "Colisão transversal":               "#ff5722",  # laranja avermelhado
    "Atropelamento de Pedestre":         "#27ae60",  # verde
    "Capotamento":                       "#6c3483",  # violeta
    "Atropelamento de Animal":           "#117a65",  # verde escuro
    "Eventos atípicos":                  "#7f8c8d",  # cinza
    "Sinistro pessoal de trânsito":      "#566573",  # cinza escuro
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


# ─────────────────────────────────────────────────────────────────────────────
# Geometria — simplificação de rota e distância ponto→segmento
# ─────────────────────────────────────────────────────────────────────────────

_R_EARTH = 6_371_000.0  # raio médio da Terra em metros


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distância em metros entre dois pontos (graus decimais)."""
    rl1, rlo1, rl2, rlo2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = rl2 - rl1
    dlon = rlo2 - rlo1
    a = np.sin(dlat / 2) ** 2 + np.cos(rl1) * np.cos(rl2) * np.sin(dlon / 2) ** 2
    return 2 * _R_EARTH * np.arcsin(np.sqrt(a))


def _simplify_route(coords: list[list[float]], max_points: int = 300) -> np.ndarray:
    """
    Reduz a polilinha a no máximo max_points por subamostragem uniforme.
    Preserva sempre o primeiro e o último ponto.
    Retorna array (N, 2) com [lat, lon].
    """
    arr = np.array(coords, dtype=np.float64)   # (N, 2)
    if len(arr) <= max_points:
        return arr
    idx = np.round(np.linspace(0, len(arr) - 1, max_points)).astype(int)
    return arr[idx]


def _min_dist_to_polyline_m(
    pts_lat: np.ndarray,
    pts_lon: np.ndarray,
    route: np.ndarray,
) -> np.ndarray:
    """
    Calcula, para cada ponto (pts_lat[i], pts_lon[i]), a distância mínima
    em metros até qualquer segmento da polilinha `route` (array Nx2 [lat,lon]).

    Usa projeção no plano local (equiretangular) — válida para distâncias
    de até ~50 km, mais que suficiente para o buffer de 1–5 km.

    Retorna array 1-D com a distância mínima de cada ponto.
    """
    # Segmentos: A → B
    A = route[:-1]   # (M, 2)
    B = route[1:]    # (M, 2)

    # Converte lat/lon para metros no plano local centrado na rota
    lat0 = route[:, 0].mean()
    cos_lat = np.cos(np.radians(lat0))
    DEG_LAT = _R_EARTH * np.pi / 180.0
    DEG_LON = DEG_LAT * cos_lat

    # Coordenadas dos segmentos em metros
    ax = A[:, 1] * DEG_LON;  ay = A[:, 0] * DEG_LAT   # (M,)
    bx = B[:, 1] * DEG_LON;  by = B[:, 0] * DEG_LAT   # (M,)
    dx = bx - ax;             dy = by - ay              # (M,)
    seg_len2 = dx * dx + dy * dy                        # (M,) comprimento² de cada segmento

    # Coordenadas dos pontos candidatos em metros
    px = (pts_lon * DEG_LON)[:, None]   # (P, 1)
    py = (pts_lat * DEG_LAT)[:, None]   # (P, 1)

    # Parâmetro t ∈ [0,1] da projeção de cada ponto sobre cada segmento
    # t = ((P-A)·(B-A)) / |B-A|²
    t = ((px - ax) * dx + (py - ay) * dy) / np.where(seg_len2 > 0, seg_len2, 1.0)
    t = np.clip(t, 0.0, 1.0)           # (P, M)

    # Ponto mais próximo sobre o segmento
    cx = ax + t * dx    # (P, M)
    cy = ay + t * dy    # (P, M)

    # Distância euclidiana no plano local
    dist2 = (px - cx) ** 2 + (py - cy) ** 2   # (P, M)
    return np.sqrt(dist2.min(axis=1))          # (P,)


def query_accidents(route_coords: list, buffer_km: float) -> pd.DataFrame:
    """
    Filtragem em 2 estágios:

    1. Pré-filtro rápido no DuckDB com bbox ligeiramente expandida
       (descarta a grande maioria dos registros sem custo de memória).

    2. Filtro preciso em NumPy: distância ponto → segmento mais próximo
       da rota ≤ buffer_km. Funciona corretamente para rotas longas
       e curvas (ex.: SP→RJ), sem o falso-positivo do retângulo.
    """
    con, col_map, _ = _load_parquet_to_duckdb()
    if not col_map["lat"] or not col_map["lon"]:
        return pd.DataFrame()

    # ── Estágio 1: pré-filtro bbox no DuckDB ──────────────────────────────────
    arr = np.array(route_coords)
    pad = (buffer_km + 1.0) / 111.0          # +1 km de margem de segurança
    min_lat = arr[:, 0].min() - pad
    max_lat = arr[:, 0].max() + pad
    min_lon = arr[:, 1].min() - pad
    max_lon = arr[:, 1].max() + pad

    lat_col  = col_map["lat"]
    lon_col  = col_map["lon"]
    tipo_col = col_map["tipo"]

    selects = [f'"{lat_col}" AS latitude', f'"{lon_col}" AS longitude']
    selects.append(f'"{tipo_col}" AS tipo_acidente' if tipo_col else "'Outros' AS tipo_acidente")
    for alias, key in [("data_acidente", "data"), ("mortos", "mortos"), ("feridos", "feridos")]:
        if col_map.get(key):
            selects.append(f'"{col_map[key]}" AS {alias}')

    sql = f"""
        SELECT {', '.join(selects)}
        FROM acidentes
        WHERE "{lat_col}" BETWEEN {min_lat} AND {max_lat}
          AND "{lon_col}" BETWEEN {min_lon} AND {max_lon}
    """
    candidates = con.execute(sql).df()

    if candidates.empty:
        return candidates

    # ── Estágio 2: filtro preciso por distância ao segmento mais próximo ──────
    route_simplified = _simplify_route(route_coords, max_points=300)

    dists = _min_dist_to_polyline_m(
        candidates["latitude"].to_numpy(),
        candidates["longitude"].to_numpy(),
        route_simplified,
    )
    candidates["dist_rota_m"] = dists.round(0).astype(int)
    mask = dists <= buffer_km * 1_000.0
    return candidates[mask].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Mapa
# ─────────────────────────────────────────────────────────────────────────────

def _cluster_icon_js(color: str) -> str:
    """
    Retorna o JavaScript que customiza a aparência do cluster bubble
    para uma determinada cor. Usado no parâmetro icon_create_function
    do MarkerCluster.
    """
    return f"""
    function(cluster) {{
        var count = cluster.getChildCount();
        var size  = count < 10 ? 32 : count < 50 ? 40 : 48;
        return L.divIcon({{
            html: '<div style="'
                + 'background:{color};'
                + 'border:2px solid white;'
                + 'border-radius:50%;'
                + 'width:' + size + 'px;'
                + 'height:' + size + 'px;'
                + 'display:flex;'
                + 'align-items:center;'
                + 'justify-content:center;'
                + 'color:white;'
                + 'font-weight:bold;'
                + 'font-size:12px;'
                + 'box-shadow:0 1px 4px rgba(0,0,0,.4);'
                + '">' + count + '</div>',
            className: '',
            iconSize: [size, size],
        }});
    }}
    """


def build_map(route_coords: list, accidents_df: pd.DataFrame) -> folium.Map:
    """
    Constrói o mapa Folium com:
      • Rota em azul
      • Marcadores de origem / destino
      • Um MarkerCluster por tipo de acidente dentro de um FeatureGroup
        (agrupamento automático ao dar zoom out; expande ao dar zoom in)
    """
    mid = len(route_coords) // 2
    m = folium.Map(location=route_coords[mid], zoom_start=12, tiles="CartoDB positron")

    # ── Rota ──────────────────────────────────────────────────────────────────
    folium.PolyLine(
        route_coords, color="#2563EB", weight=5,
        opacity=0.85, tooltip="Rota calculada",
    ).add_to(m)

    folium.Marker(
        route_coords[0], tooltip="Origem",
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
    ).add_to(m)
    folium.Marker(
        route_coords[-1], tooltip="Destino",
        icon=folium.Icon(color="red", icon="flag", prefix="fa"),
    ).add_to(m)

    # ── Acidentes agrupados por tipo ──────────────────────────────────────────
    if accidents_df.empty:
        return m

    tipos = sorted(accidents_df["tipo_acidente"].dropna().unique())

    for tipo in tipos:
        tipo_str  = str(tipo)
        hex_color = ACCIDENT_COLORS.get(tipo_str, "#7f8c8d")

        # FeatureGroup → aparece como camada ligável/desligável na legenda
        fg = folium.FeatureGroup(name=f"● {tipo_str}", show=True)

        # MarkerCluster dentro do FeatureGroup com bubble colorida
        cluster = MarkerCluster(
            name=tipo_str,
            icon_create_function=_cluster_icon_js(hex_color),
            options={
                "maxClusterRadius": 60,       # raio (px) para agrupar
                "disableClusteringAtZoom": 16, # expande totalmente a partir deste zoom
                "spiderfyOnMaxZoom": True,
            },
        )

        subset = accidents_df[accidents_df["tipo_acidente"].astype(str) == tipo_str]

        for _, row in subset.iterrows():
            popup_html = (
                f"<b style='color:{hex_color}'>{tipo_str}</b><br>"
                f"<small>Lat {row['latitude']:.5f} | Lon {row['longitude']:.5f}</small>"
            )
            if "dist_rota_m" in row:
                popup_html += f"<br>📏 Dist. rota: {int(row['dist_rota_m'])} m"
            if "data_acidente" in row and pd.notna(row["data_acidente"]):
                popup_html += f"<br>📅 {str(row['data_acidente'])[:10]}"
            if "mortos" in row and pd.notna(row["mortos"]):
                popup_html += f"<br>💀 Mortos: {int(row['mortos'])}"
            if "feridos" in row and pd.notna(row["feridos"]):
                popup_html += f"<br>🤕 Feridos: {int(row['feridos'])}"

            # Ícone individual: círculo colorido via DivIcon
            div_icon = folium.DivIcon(
                html=(
                    f"<div style='"
                    f"width:12px;height:12px;"
                    f"border-radius:50%;"
                    f"background:{hex_color};"
                    f"border:1.5px solid white;"
                    f"box-shadow:0 1px 3px rgba(0,0,0,.35);"
                    f"'></div>"
                ),
                icon_size=(12, 12),
                icon_anchor=(6, 6),
            )

            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                icon=div_icon,
                popup=folium.Popup(popup_html, max_width=230),
                tooltip=tipo_str,
            ).add_to(cluster)

        cluster.add_to(fg)
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
                hex_cor = ACCIDENT_COLORS.get(str(t), "#7f8c8d")
                st.markdown(f"<span style='color:{hex_cor}'>●</span> {t}", unsafe_allow_html=True)
    else:
        st.error(f"Erro ao carregar dados:\n{_load_error}")

    st.markdown("---")
    buffer_km = st.slider(
        "Raio do buffer da rota (km)", min_value=1, max_value=20, value=1,
        help="Filtra apenas acidentes dentro deste raio em relação à linha da rota.",
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
        route_coords, _ = get_route(origin_ll, dest_ll)

    if route_coords is None:
        st.session_state.error_msg = "Não foi possível traçar a rota. Verifique os endereços."
        st.rerun()

    with st.spinner("Consultando acidentes na rota..."):
        accidents_df = query_accidents(route_coords, buffer_km)

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
    m2.metric("⚠️ Acidentes no buffer", f"{total_acc:,}")
    if total_acc > 0 and "tipo_acidente" in accidents_df.columns:
        top = accidents_df["tipo_acidente"].value_counts().idxmax()
        m3.metric("🔝 Tipo mais frequente", top)

    # Mapa
    st.markdown("### 🗺️ Mapa")
    mapa = build_map(route_coords, accidents_df)
    st_folium(mapa, use_container_width=True, height=600, returned_objects=[])

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
    st_folium(m, use_container_width=True, height=500, returned_objects=[])
