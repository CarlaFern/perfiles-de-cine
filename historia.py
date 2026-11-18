"""Datos y gráficos del relato de la app.

Todo número que aparece en un título o en un texto se calcula acá, a partir de
las salidas del pipeline (usuarios_generos.csv, ratings_selected_users.csv,
tmdb_movies_prepared.csv) y del modelo de modelado.py. La app sólo los muestra.
"""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

from modelado import (COLUMNAS_DECADAS, MIN_RATINGS, SEMILLA, SUAVIZADO,
                      COBERTURA, entrenar_desde_csv)

# ---------------------------------------------------------------------------
# Colores: un solo canal por dimensión para toda la app
# ---------------------------------------------------------------------------
GRIS = "#6b6b75"
GRIS_CLARO = "#a3a3ad"
ROJO = "#e50914"

GENEROS_GRUPOS = ["Animación, familia y aventura", "Drama, crimen e histórico"]
DECADAS_GRUPOS = ["Cine de 2000–2010", "Cine noventero y clásico"]
COLOR_GENERO = alt.Scale(domain=GENEROS_GRUPOS, range=["#f2b134", "#d64550"])
COLOR_DECADA = alt.Scale(domain=DECADAS_GRUPOS, range=["#4cc9f0", "#9b8cff"])

NOMBRE_EPOCA = {
    "prop_antes_1980": "Antes de 1980", "prop_1980s": "1980s",
    "prop_1990s": "1990s", "prop_2000s": "2000s", "prop_2010s": "2010s",
    "prop_2020_mas": "2020+",
}
ORDEN_EPOCAS = list(NOMBRE_EPOCA.values())

GENERO_ES = {
    "Action": "Acción", "Adventure": "Aventura", "Animation": "Animación",
    "Comedy": "Comedia", "Crime": "Crimen", "Documentary": "Documental",
    "Drama": "Drama", "Family": "Familia", "Fantasy": "Fantasía",
    "History": "Historia", "Horror": "Terror", "Music": "Música",
    "Mystery": "Misterio", "Romance": "Romance", "Science Fiction": "Ciencia ficción",
    "TV Movie": "Película de TV", "Thriller": "Suspenso", "War": "Bélico",
    "Western": "Western",
}


def fmt(n) -> str:
    """Entero con punto de miles: 3503 → '3.503'."""
    return f"{int(round(n)):,}".replace(",", ".")


def dec(x, d=1) -> str:
    """Decimal con coma: 0.175 → '0,175'."""
    return f"{x:.{d}f}".replace(".", ",")


def titulo(texto, subtitulo=None):
    """Título que afirma, con un subtítulo de qué mirar (como en la sección 7.6)."""
    # limit=0: sin límite de ancho, para que Streamlit no corte el título con «…»
    return alt.Title(texto, subtitle=subtitulo or "", anchor="start", fontSize=16, limit=0,
                     subtitleColor=GRIS_CLARO, subtitleFontSize=12, offset=12)


# ---------------------------------------------------------------------------
# Datos base
# ---------------------------------------------------------------------------
def cargar_todo(ruta_silver="usuarios_generos.csv",
                ruta_ratings="ratings_selected_users.csv",
                ruta_peliculas="tmdb_movies_prepared.csv"):
    silver = pd.read_csv(ruta_silver)
    modelos = entrenar_desde_csv(ruta_silver)
    ratings = pd.read_csv(ruta_ratings)
    peliculas = pd.read_csv(ruta_peliculas)
    return silver, modelos, ratings, peliculas


def tabla_usuarios(modelos, silver):
    """Una fila por usuario modelado: grupos, PCs, actividad, afinidades y épocas."""
    D = modelos.entrenamiento
    et = modelos.etiquetas.copy()
    prefs = modelos.generos["preparar"]["features"].transform(D)
    prefs.columns = [GENERO_ES.get(c.replace("pref_", ""), c) for c in prefs.columns]
    epocas = (D[COLUMNAS_DECADAS] * 100).rename(columns=NOMBRE_EPOCA)
    tabla = et.join(D[["cantidad_valoraciones", "cantidad_valoraciones_total",
                       "rating_promedio_general"]])
    tabla["genero_afin"] = prefs.idxmax(axis=1)
    tabla = tabla.join(prefs.add_prefix("g_")).join(epocas.add_prefix("e_"))
    return tabla.reset_index()


def espacios(modelos):
    D = modelos.entrenamiento
    zg = modelos.generos["preparar"].transform(D)
    zd = modelos.decadas["preparar"].transform(D)
    return D.index.to_numpy(), zg, zd


# ---------------------------------------------------------------------------
# 1 · Los datos
# ---------------------------------------------------------------------------
def resumen_actividad(silver):
    total = silver["cantidad_valoraciones_total"]
    return {
        "usuarios": len(silver),
        "minimo": int(total.min()),
        "mediana": int(total.median()),
        "maximo": int(total.max()),
        "con_tope": int((total > 500).sum()),
        "modelados": int((silver["cantidad_valoraciones"] >= MIN_RATINGS).sum()),
    }


def tabla_histograma(silver, bins=24):
    total = silver["cantidad_valoraciones_total"]
    # intervalos iguales en escala log, elegidos para que 150 sea un borde exacto
    paso = np.log10(MIN_RATINGS / 100) / 2
    bordes = 10 ** np.arange(2, 4 + paso, paso)
    cortes = pd.cut(total, bordes, right=False, include_lowest=True)
    tabla = cortes.value_counts(sort=False).rename_axis("intervalo").reset_index(name="usuarios")
    tabla["desde"] = [i.left for i in tabla["intervalo"]]
    tabla["hasta"] = [i.right for i in tabla["intervalo"]]
    tabla["tramo"] = np.where(tabla["desde"] >= MIN_RATINGS - 1, "Entra al modelo", "Queda afuera")
    return tabla[["desde", "hasta", "usuarios", "tramo"]]


def grafico_histograma(silver):
    r = resumen_actividad(silver)
    tabla = tabla_histograma(silver)
    x = alt.X("desde:Q", title="Ratings por usuario (historial completo, escala log)",
              scale=alt.Scale(type="log", domain=[100, 10000]),
              axis=alt.Axis(values=[100, 150, 200, 300, 500, 1000, 2000, 5000, 10000], format="~s"))
    barras = alt.Chart(tabla).mark_bar(stroke="#0f0f12", strokeWidth=0.5).encode(
        x=x, x2="hasta:Q",
        y=alt.Y("usuarios:Q", title="Usuarios"), y2=alt.datum(0),   # la barra va de 0 a su altura
        color=alt.Color("tramo:N", title=None, scale=alt.Scale(
            domain=["Entra al modelo", "Queda afuera"], range=[ROJO, GRIS]),
            legend=alt.Legend(orient="top-right")),
        tooltip=[alt.Tooltip("desde:Q", title="Desde", format=".0f"),
                 alt.Tooltip("hasta:Q", title="Hasta", format=".0f"),
                 alt.Tooltip("usuarios:Q", title="Usuarios")])
    reglas = pd.DataFrame({"x": [150, 500], "y": [tabla["usuarios"].max() * 1.02] * 2,
                           "t": ["corte del modelo: 150", "tope del pipeline: 500"]})
    regla = alt.Chart(reglas).mark_rule(strokeDash=[4, 3], color=GRIS_CLARO).encode(x="x:Q")
    texto = alt.Chart(reglas).mark_text(align="left", dx=5, dy=4, baseline="top",
                                        color=GRIS_CLARO, fontSize=11).encode(
        x="x:Q", y=alt.value(2), text="t:N")
    return (barras + regla + texto).properties(height=430, title=titulo(
        f"La mitad de los usuarios califica menos de {fmt(r['mediana'])} películas; "
        f"unos pocos superan las {fmt(r['maximo'] // 1000 * 1000)}",
        "Cada barra cuenta usuarios. El eje es logarítmico: cada marca multiplica la cantidad de ratings."))


# ---------------------------------------------------------------------------
# 2 · Lo que se pierde en el pipeline
# ---------------------------------------------------------------------------
def tabla_etapas(silver, ratings, peliculas):
    con_genero = peliculas.loc[peliculas["genres"].fillna("") != "", "movieId"]
    modelados = silver["cantidad_valoraciones"] >= MIN_RATINGS
    filas = [
        ("Historial completo de los 5.000", len(silver), silver["cantidad_valoraciones_total"].sum()),
        ("Tope de 500 ratings por usuario", len(silver), silver["cantidad_valoraciones"].sum()),
        ("Mínimo de 150 ratings", int(modelados.sum()), silver.loc[modelados, "cantidad_valoraciones"].sum()),
        ("Con género conocido en TMDB", ratings["userId"].nunique(),
         int(ratings["movieId"].isin(con_genero).sum())),
    ]
    tabla = pd.DataFrame(filas, columns=["etapa", "usuarios", "ratings"])
    tabla["orden"] = range(1, len(tabla) + 1)
    tabla["perdidos"] = (-tabla["ratings"].diff()).fillna(0).astype(int)
    tabla["pct_perdido"] = tabla["perdidos"] / tabla["ratings"].shift(1)
    return tabla


def grafico_etapas(tabla):
    tope = tabla.loc[1, "pct_perdido"] * 100
    tmdb = tabla.loc[3, "pct_perdido"] * 100
    tabla = tabla.assign(
        etiqueta=[f"{fmt(r)} ratings · {fmt(u)} usuarios" for r, u in zip(tabla["ratings"], tabla["usuarios"])],
        nota=["", f"−{dec(tope)} %", f"−{dec(tabla.loc[2, 'pct_perdido'] * 100)} %", f"−{dec(tmdb)} %"])
    y = alt.Y("etapa:N", sort=list(tabla["etapa"]), title=None,
              axis=alt.Axis(labelLimit=260))
    barras = alt.Chart(tabla).mark_bar(color=GRIS, cornerRadiusEnd=3).encode(
        x=alt.X("ratings:Q", title="Ratings (las barras arrancan en cero)",
                scale=alt.Scale(domain=[0, 1_800_000]), axis=alt.Axis(format="~s")),
        y=y, tooltip=["etapa:N", alt.Tooltip("ratings:Q", format=","), "usuarios:Q"])
    ultima = barras.transform_filter(alt.datum.orden == 4).mark_bar(color=ROJO, cornerRadiusEnd=3)
    texto = alt.Chart(tabla).mark_text(align="left", dx=6, color="#ececf1", fontSize=11).encode(
        x="ratings:Q", y=y, text="etiqueta:N")
    nota = alt.Chart(tabla).mark_text(align="right", dx=-6, color="#0f0f12", fontWeight="bold",
                                      fontSize=11).encode(x="ratings:Q", y=y, text="nota:N")
    return (barras + ultima + texto + nota).properties(height=340, title=titulo(
        f"El tope de 500 recorta el {dec(tope)} % de los ratings; TMDB, apenas el {dec(tmdb)} %",
        "Ningún usuario se pierde en la ingesta: sólo el corte de 150 deja usuarios afuera."))


# ---------------------------------------------------------------------------
# 3 · La trampa de los pocos ratings
# ---------------------------------------------------------------------------
TRAMOS_N = [(1, 1, "1"), (2, 2, "2"), (3, 4, "3–4"), (5, 9, "5–9"),
            (10, 19, "10–19"), (20, 49, "20–49"), (50, 10_000, "50 o más")]


def tabla_trampa(silver):
    generos = [c[2:] for c in silver.columns if c.startswith("n_")]
    general = silver["rating_promedio_general"].to_numpy()
    partes = []
    for g in generos:
        n = silver[f"n_{g}"].to_numpy()
        media = silver[g].to_numpy()
        ok = n > 0
        partes.append(pd.DataFrame({"n": n[ok], "dif": media[ok] - general[ok]}))
    largo = pd.concat(partes, ignore_index=True)
    largo["amortiguada"] = largo["n"] / (largo["n"] + SUAVIZADO) * largo["dif"]
    filas = []
    for lo, hi, nombre in TRAMOS_N:
        sub = largo[(largo["n"] >= lo) & (largo["n"] <= hi)]
        for version, col in [("Promedio crudo", "dif"), (f"Amortiguado (m = {SUAVIZADO})", "amortiguada")]:
            q = sub[col].quantile([.05, .25, .5, .75, .95]).to_numpy()
            filas.append({"tramo": nombre, "version": version, "pares": len(sub),
                          "p05": q[0], "p25": q[1], "mediana": q[2], "p75": q[3], "p95": q[4]})
    return pd.DataFrame(filas)


def grafico_trampa(tabla):
    orden = [t[2] for t in TRAMOS_N]
    crudo = tabla[tabla["version"] == "Promedio crudo"].set_index("tramo")
    ancho_1 = crudo.loc["1", "p95"] - crudo.loc["1", "p05"]
    ancho_50 = crudo.loc["50 o más", "p95"] - crudo.loc["50 o más", "p05"]
    x = alt.X("tramo:O", sort=orden, title="Películas del género que calificó el usuario",
              axis=alt.Axis(labelAngle=0))
    y = alt.Y("p05:Q", title="Promedio del género − promedio propio",
              scale=alt.Scale(domain=[-2.5, 2]))
    base = alt.Chart()   # sin datos: la tabla se pasa una sola vez, a la capa que se facetea
    banda = base.mark_area(opacity=0.25, color=ROJO).encode(x=x, y=y, y2="p95:Q")
    caja = base.mark_area(opacity=0.45, color=ROJO).encode(x=x, y="p25:Q", y2="p75:Q")
    linea = base.mark_line(color="#ececf1", point=True).encode(
        x=x, y="mediana:Q",
        tooltip=["tramo:O", alt.Tooltip("pares:Q", title="usuario × género", format=","),
                 alt.Tooltip("p05:Q", format=".2f"), alt.Tooltip("mediana:Q", format=".2f"),
                 alt.Tooltip("p95:Q", format=".2f")])
    # el cero usa la misma tabla que el resto (la faceta exige un solo origen de datos)
    cero = base.mark_rule(color=GRIS, strokeDash=[3, 3]).encode(y=alt.datum(0))
    capas = alt.layer(banda, caja, linea, cero, data=tabla).properties(width=440, height=260)
    return capas.facet(column=alt.Column("version:N", title=None, sort=[
        "Promedio crudo", f"Amortiguado (m = {SUAVIZADO})"],
        header=alt.Header(labelFontSize=13, labelColor="#ececf1", labelAnchor="start"))
    ).resolve_scale(y="shared").properties(title=titulo(
        f"Con una sola película, el «gusto» por un género varía {dec(ancho_1 / ancho_50, 0)} veces más "
        "que con cincuenta",
        ["Banda clara: 90 % central de los pares usuario × género; banda oscura: 50 % central; línea: mediana.",
         "La versión amortiguada achica la diferencia cuando hay pocas películas y la deja casi igual cuando hay muchas."]))


def tabla_umbral(silver):
    filas = []
    for x in [100, 150, 200, 250, 300, 400]:
        n = int((silver["cantidad_valoraciones"] >= x).sum())
        filas.append({"minimo": x, "usuarios": n, "pct": n / len(silver)})
    return pd.DataFrame(filas)


def grafico_umbral(tabla):
    fila = tabla.set_index("minimo").loc[MIN_RATINGS]
    tabla = tabla.assign(t=[f"{fmt(u)} · {dec(p * 100, 0)} %" for u, p in zip(tabla["usuarios"], tabla["pct"])])
    x = alt.X("minimo:O", title="Mínimo de ratings por usuario", axis=alt.Axis(labelAngle=0))
    barras = alt.Chart(tabla).mark_bar(cornerRadiusEnd=3).encode(
        x=x, y=alt.Y("usuarios:Q", title="Usuarios que quedan", scale=alt.Scale(domain=[0, 5600])),
        color=alt.condition(alt.datum.minimo == MIN_RATINGS, alt.value(ROJO), alt.value(GRIS)))
    texto = alt.Chart(tabla).mark_text(dy=-8, fontSize=11, color="#ececf1").encode(
        x=x, y="usuarios:Q", text="t:N")
    return (barras + texto).properties(height=370, title=titulo(
        f"Con {MIN_RATINGS} ratings como mínimo quedan {fmt(fila['usuarios'])} usuarios: "
        f"el {dec(fila['pct'] * 100, 0)} %",
        "Subir el mínimo da perfiles más confiables, pero cada escalón deja afuera a miles."))


# ---------------------------------------------------------------------------
# 4 · Qué entra al modelo
# ---------------------------------------------------------------------------
def tabla_variables(silver, modelos):
    D = modelos.entrenamiento
    generos = [c[2:] for c in silver.columns if c.startswith("n_")]
    cobertura = {g: (D[f"n_{g}"].fillna(0) > 0).mean() for g in generos}
    usados = modelos.generos["preparar"]["features"].generos_
    fuera = [g for g in generos if g not in usados]
    pct_fuera = ", ".join(f"{GENERO_ES[g]} {dec(cobertura[g] * 100, 0)} %" for g in fuera)
    con_2020 = (D["prop_2020_mas"] > 0).mean() * 100
    filas = [
        ("Afinidad por género", f"{len(usados)} columnas", True,
         f"Promedio del género menos el promedio propio, amortiguado con n/(n+{SUAVIZADO})."),
        ("Géneros poco vistos", ", ".join(GENERO_ES[g] for g in fuera), False,
         f"Cobertura menor al {dec(COBERTURA * 100, 0)} % de los usuarios ({pct_fuera})."),
        ("Proporción por época", "antes de 1980 … 2010s (5)", True,
         "Qué parte del historial es de cada época, transformada con arcsen(√p)."),
        ("Proporción 2020+", "prop_2020_mas", False,
         f"Sólo el {dec(con_2020, 0)} % de los usuarios calificó algo de 2020 en adelante."),
        ("Cantidad de ratings", "cantidad_valoraciones", False,
         "Es actividad, no gusto: sirve para el corte de 150 y para chequear que no arme los grupos."),
        ("Promedio general", "rating_promedio_general", False,
         "No entra sola: se usa para centrar cada género en la vara del propio usuario."),
        ("Género mejor valorado", "genero_mejor_valorado", False,
         "Lo suele decidir un género visto una o dos veces (hipótesis H4 de la Entrega 2)."),
        ("Identificador y fechas", "userId, primera / última valoración", False,
         "No describen gustos; userId sería fuga de información."),
        ("Otras temporales", "anio_estreno_std, desfase_temporal, rating_std", False,
         "El bloque de épocas ya describe el historial temporal."),
    ]
    tabla = pd.DataFrame(filas, columns=["Variable", "Columnas", "Entra", "Por qué"])
    tabla["Entra"] = tabla["Entra"].map({True: "✅ Entra", False: "— Sale"})
    return tabla


# ---------------------------------------------------------------------------
# 5 · PCA
# ---------------------------------------------------------------------------
def tabla_varianza(modelos):
    filas = []
    for bloque, modelo in [("Géneros", modelos.generos), ("Épocas", modelos.decadas)]:
        X = modelo["preparar"][:-1].transform(modelos.entrenamiento)
        completa = PCA(random_state=SEMILLA).fit(X)
        elegidas = modelo["preparar"]["pca"].n_components_
        for i, v in enumerate(np.cumsum(completa.explained_variance_ratio_), start=1):
            filas.append({"bloque": bloque, "componente": i, "acumulada": v, "elegida": i == elegidas})
    return pd.DataFrame(filas)


def grafico_varianza(tabla):
    el = tabla[tabla["elegida"]].set_index("bloque")
    color = alt.Color("bloque:N", title=None, scale=alt.Scale(
        domain=["Géneros", "Épocas"], range=["#f2b134", "#4cc9f0"]), legend=None)
    x = alt.X("componente:Q", title="Componentes conservadas", scale=alt.Scale(domain=[1, 17]),
              axis=alt.Axis(tickMinStep=1))
    y = alt.Y("acumulada:Q", title="Varianza explicada acumulada",
              scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"))
    lineas = alt.Chart(tabla).mark_line(point=alt.OverlayMarkDef(size=30)).encode(
        x=x, y=y, color=color, tooltip=["bloque:N", "componente:Q", alt.Tooltip("acumulada:Q", format=".1%")])
    marcas = alt.Chart(tabla[tabla["elegida"]]).mark_point(size=180, filled=False, strokeWidth=2).encode(
        x=x, y=y, color=color)
    rotulos = alt.Chart(tabla[tabla["elegida"]].assign(
        t=[f"{b}: {c} componentes, {dec(a * 100)} %" for b, c, a in
           zip(tabla.loc[tabla["elegida"], "bloque"], tabla.loc[tabla["elegida"], "componente"],
               tabla.loc[tabla["elegida"], "acumulada"])])).mark_text(
        align="left", dx=12, dy=16, fontSize=12, fontWeight="bold").encode(x=x, y=y, text="t:N", color=color)
    regla = alt.Chart(pd.DataFrame({"y": [0.8]})).mark_rule(strokeDash=[4, 3], color=GRIS_CLARO).encode(y="y:Q")
    return (regla + lineas + marcas + rotulos).properties(height=410, title=titulo(
        [f"{el.loc['Géneros', 'componente']} componentes alcanzan para los géneros;",
         f"{el.loc['Épocas', 'componente']}, para las épocas"],
        "La línea punteada es el 80 % que pedimos conservar."))


def tabla_cargas(modelos):
    pca = modelos.generos["preparar"]["pca"]
    nombres = [GENERO_ES[g] for g in modelos.generos["preparar"]["features"].generos_]
    t = pd.DataFrame({"genero": nombres, "carga": pca.components_[0]})
    signo_drama = np.sign(t.set_index("genero").loc["Drama", "carga"])
    t["lado"] = np.where(np.sign(t["carga"]) == signo_drama, GENEROS_GRUPOS[1], GENEROS_GRUPOS[0])
    return t.sort_values("carga")


def grafico_cargas(tabla):
    return alt.Chart(tabla).mark_bar(cornerRadiusEnd=2).encode(
        x=alt.X("carga:Q", title="Carga en la primera componente de géneros"),
        y=alt.Y("genero:N", sort=list(tabla["genero"]), title=None, axis=alt.Axis(labelOverlap=False)),
        color=alt.Color("lado:N", scale=COLOR_GENERO, title=None,
                        legend=alt.Legend(orient="bottom", labelLimit=320)),
        tooltip=["genero:N", alt.Tooltip("carga:Q", format=".2f")],
    ).properties(height=510, title=titulo(
        ["El primer eje de géneros opone drama y crimen", "a animación y familia"],
        "El signo es arbitrario: importa qué géneros quedan de cada lado."))


# ---------------------------------------------------------------------------
# 6 · Cuántos grupos
# ---------------------------------------------------------------------------
def evaluar_k(modelos):
    _, zg, zd = espacios(modelos)
    filas = []
    for bloque, Z in [("Géneros", zg), ("Épocas", zd)]:
        base = None
        for k in range(1, 7):
            km = KMeans(n_clusters=k, random_state=SEMILLA, n_init=10).fit(Z)
            base = km.inertia_ if k == 1 else base
            filas.append({"bloque": bloque, "k": k, "inercia": km.inertia_ / base,
                          "silhouette": silhouette_score(Z, km.labels_) if k > 1 else np.nan})
    evaluacion = pd.DataFrame(filas)
    # modelo conjunto: las dos PCA juntas, cada bloque con el mismo peso
    zc = np.hstack([zg / np.sqrt((zg ** 2).sum(1).mean()), zd / np.sqrt((zd ** 2).sum(1).mean())])
    conjunto = KMeans(n_clusters=2, random_state=SEMILLA, n_init=10).fit_predict(zc)
    et = modelos.etiquetas
    ari = pd.DataFrame({
        "comparacion": ["Conjunto vs. épocas", "Conjunto vs. géneros"],
        "ari": [adjusted_rand_score(conjunto, et["grupo_decada"]),
                adjusted_rand_score(conjunto, et["grupo_genero"])]})
    return evaluacion, ari, silhouette_score(zc, conjunto)


def grafico_codo(evaluacion):
    color = alt.Color("bloque:N", title=None, scale=alt.Scale(
        domain=["Géneros", "Épocas"], range=["#f2b134", "#4cc9f0"]), legend=alt.Legend(orient="top-right"))
    x = alt.X("k:O", title="Cantidad de grupos (k)", axis=alt.Axis(labelAngle=0))
    codo = alt.Chart(evaluacion).mark_line(point=True).encode(
        x=x, y=alt.Y("inercia:Q", title="Inercia (k = 1 vale 100 %)", axis=alt.Axis(format="%"),
                     scale=alt.Scale(domain=[0, 1])),
        color=color, tooltip=["bloque:N", "k:O", alt.Tooltip("inercia:Q", format=".0%")]
    ).properties(height=360, title=titulo("El codo de las épocas está en k = 2",
                                          "Cuánta dispersión queda dentro de los grupos (menos es mejor)."))
    sil = evaluacion.dropna()
    k2 = sil[sil["k"] == 2]
    lineas = alt.Chart(sil).mark_line(point=True).encode(
        x=x, y=alt.Y("silhouette:Q", title="Silhouette", scale=alt.Scale(domain=[0, 0.5])),
        color=color, tooltip=["bloque:N", "k:O", alt.Tooltip("silhouette:Q", format=".3f")])
    marca = alt.Chart(k2).mark_point(size=200, filled=False, strokeWidth=2).encode(
        x=x, y="silhouette:Q", color=color)
    texto = alt.Chart(k2.assign(t=[dec(v, 3) for v in k2["silhouette"]])).mark_text(
        dx=14, align="left", fontWeight="bold").encode(x=x, y="silhouette:Q", text="t:N", color=color)
    silueta = (lineas + marca + texto).properties(height=360, title=titulo(
        "Con k = 2 la separación es la más alta en los dos bloques",
        "Silhouette: qué tan cerca está cada usuario de su grupo y lejos del otro."))
    return codo, silueta


def grafico_ari(ari):
    t = ari.assign(texto=[dec(v, 3) for v in ari["ari"]])
    y = alt.Y("comparacion:N", title=None, sort=list(t["comparacion"]))
    barras = alt.Chart(t).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("ari:Q", title="Coincidencia entre agrupamientos (ARI, 1 = idénticos)",
                scale=alt.Scale(domain=[0, 1])), y=y,
        color=alt.condition(alt.datum.comparacion == "Conjunto vs. épocas", alt.value("#4cc9f0"), alt.value("#f2b134")))
    texto = alt.Chart(t).mark_text(align="left", dx=6, fontWeight="bold", color="#ececf1").encode(
        x=alt.X("ari:Q", scale=alt.Scale(domain=[0, 1])), y=y, text="texto:N")
    return (barras + texto).properties(height=240, title=titulo(
        "Un solo modelo con géneros y épocas juntos copia a las épocas y borra los géneros",
        "Por eso mantenemos dos modelos separados."))


# ---------------------------------------------------------------------------
# 7 · Los perfiles
# ---------------------------------------------------------------------------
def tabla_cruce(usuarios):
    cruce = pd.crosstab(usuarios["nombre_genero"], usuarios["nombre_decada"])
    chi2, p, _, _ = chi2_contingency(cruce)
    v = np.sqrt(chi2 / (cruce.to_numpy().sum() * (min(cruce.shape) - 1)))
    largo = cruce.stack().rename("usuarios").reset_index()
    largo["pct"] = largo["usuarios"] / largo["usuarios"].sum()
    return largo, p, v


def grafico_cruce(largo, p, v):
    largo = largo.assign(t=[f"{fmt(u)} · {dec(q * 100)} %" for u, q in zip(largo["usuarios"], largo["pct"])])
    base = alt.Chart(largo).encode(
        x=alt.X("nombre_decada:N", title="Grupo por época", sort=DECADAS_GRUPOS,
                axis=alt.Axis(labelAngle=0, orient="top")),
        y=alt.Y("nombre_genero:N", title="Grupo por género", sort=GENEROS_GRUPOS,
                axis=alt.Axis(labelLimit=220)))
    celdas = base.mark_rect(cornerRadius=6, stroke="#0f0f12", strokeWidth=3).encode(
        color=alt.Color("usuarios:Q", legend=None, scale=alt.Scale(range=["#3a1518", "#b3121b"])))
    texto = base.mark_text(fontSize=15, fontWeight="bold", color="#ffffff").encode(text="t:N")
    return (celdas + texto).properties(height=330, title=titulo(
        "Los cuatro perfiles: los dos grupos de género se reparten mitad y mitad entre las épocas",
        f"Test χ²: p = {dec(p, 2)}, V de Cramér = {dec(v, 3)}. Género y época son dimensiones distintas."))


def tabla_perfil_generos(usuarios):
    cols = [c for c in usuarios.columns if c.startswith("g_")]
    t = usuarios.groupby("nombre_genero")[cols].mean().T
    t.index = [c[2:] for c in t.index]
    t["diferencia"] = t[GENEROS_GRUPOS[1]] - t[GENEROS_GRUPOS[0]]
    largo = t.drop(columns="diferencia").stack().rename("afinidad").reset_index()
    largo.columns = ["genero", "grupo", "afinidad"]
    return largo, t["diferencia"].sort_values()


def grafico_perfil_generos(largo, orden):
    y = alt.Y("genero:N", sort=list(orden.index), title=None, axis=alt.Axis(labelOverlap=False))
    regla = alt.Chart(largo).mark_rule(color=GRIS, strokeWidth=2).encode(
        y=y, x="min(afinidad):Q", x2="max(afinidad):Q")
    puntos = alt.Chart(largo).mark_circle(size=110, opacity=1).encode(
        y=y, x=alt.X("afinidad:Q", title="Afinidad media (respecto del promedio propio)"),
        color=alt.Color("grupo:N", scale=COLOR_GENERO, title=None,
                        legend=alt.Legend(orient="bottom", labelLimit=320)),
        tooltip=["genero:N", "grupo:N", alt.Tooltip("afinidad:Q", format="+.3f")])
    cero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=GRIS, strokeDash=[3, 3]).encode(x="x:Q")
    return (cero + regla + puntos).properties(height=530, title=titulo(
        ["Drama, crimen y bélico de un lado;", "animación, familia y aventura del otro"],
        ["Cada punto es el promedio de un grupo.", "Ordenados por la distancia entre los dos grupos."]))


def tabla_perfil_epocas(usuarios):
    cols = [c for c in usuarios.columns if c.startswith("e_")]
    t = usuarios.groupby("nombre_decada")[cols].mean()
    t.columns = [c[2:] for c in t.columns]
    largo = t.stack().rename("pct").reset_index()
    largo.columns = ["grupo", "epoca", "pct"]
    return largo


def grafico_perfil_epocas(largo):
    x = alt.X("epoca:N", sort=ORDEN_EPOCAS, title="Época de estreno", axis=alt.Axis(labelAngle=0))
    return alt.Chart(largo).mark_bar(cornerRadiusEnd=2).encode(
        x=x, xOffset=alt.XOffset("grupo:N", sort=DECADAS_GRUPOS),
        y=alt.Y("pct:Q", title="% del historial (promedio del grupo)"),
        color=alt.Color("grupo:N", scale=COLOR_DECADA, title=None,
                        legend=alt.Legend(orient="bottom", labelLimit=320)),
        tooltip=["grupo:N", "epoca:N", alt.Tooltip("pct:Q", format=".1f")],
    ).properties(height=470, title=titulo(
        ["Un grupo ve sobre todo cine de los 2000 y 2010;", "el otro, de los 90 para atrás"],
        ["Mide qué hay en el historial,", "no qué época le gusta más a cada uno."]))


def pregunta_actividad(usuarios, modelos):
    """¿Los grupos separan a los que califican mucho de los que califican poco?"""
    _, zg, zd = espacios(modelos)
    act = usuarios["cantidad_valoraciones_total"].to_numpy()
    rho = {f"{b} PC{i + 1}": spearmanr(Z[:, i], act).statistic
           for b, Z in [("Géneros", zg), ("Épocas", zd)] for i in range(2)}
    medianas = pd.concat([
        usuarios.groupby("nombre_genero")["cantidad_valoraciones_total"].median().rename_axis("grupo"),
        usuarios.groupby("nombre_decada")["cantidad_valoraciones_total"].median().rename_axis("grupo")])
    return pd.Series(rho), medianas


def grafico_actividad(usuarios, rho, medianas):
    largo = pd.concat([
        usuarios[["cantidad_valoraciones_total", "nombre_genero"]].rename(
            columns={"nombre_genero": "grupo"}).assign(modelo="Por género"),
        usuarios[["cantidad_valoraciones_total", "nombre_decada"]].rename(
            columns={"nombre_decada": "grupo"}).assign(modelo="Por época")])
    escala = alt.Scale(domain=GENEROS_GRUPOS + DECADAS_GRUPOS,
                       range=COLOR_GENERO.range + COLOR_DECADA.range)
    cajas = alt.Chart(largo).mark_boxplot(size=26, median={"color": "#ffffff"},
                                          outliers={"size": 6, "opacity": 0.25}).encode(
        y=alt.Y("grupo:N", title=None, sort=GENEROS_GRUPOS + DECADAS_GRUPOS, axis=alt.Axis(labelLimit=220)),
        x=alt.X("cantidad_valoraciones_total:Q", title="Ratings por usuario (escala log)",
                scale=alt.Scale(type="log", domain=[150, 10000])),
        color=alt.Color("grupo:N", scale=escala, legend=None))
    return cajas.properties(height=340, title=titulo(
        "Los grupos no son «los que califican mucho» contra «los que califican poco»",
        f"Medianas entre {fmt(medianas.min())} y {fmt(medianas.max())} ratings; "
        f"la correlación de la actividad con los ejes del modelo no pasa de |ρ| = {dec(rho.abs().max(), 2)}."))


# ---------------------------------------------------------------------------
# 8 · La copa: explorar
# ---------------------------------------------------------------------------
def grafico_copa(usuarios, espacio, minimo):
    """Nube PC1 × PC2 con selección por leyenda y un brush que filtra las barras."""
    datos = usuarios[usuarios["cantidad_valoraciones_total"] >= minimo]
    if espacio == "Géneros":
        px, py, grupo, escala = "G_PC1", "G_PC2", "nombre_genero", COLOR_GENERO
        cols = [c for c in datos.columns if c.startswith("g_")]
        eje_barras, fmt_barras = "Afinidad media de los seleccionados", "+.3f"
    else:
        px, py, grupo, escala = "D_PC1", "D_PC2", "nombre_decada", COLOR_DECADA
        cols = [c for c in datos.columns if c.startswith("e_")]
        eje_barras, fmt_barras = "% medio del historial de los seleccionados", ".1f"
    datos = datos[["userId", px, py, grupo, "perfil", "cantidad_valoraciones_total", "genero_afin"] + cols]
    datos = datos.rename(columns={px: "x", py: "y", grupo: "grupo"})
    nombres = [c[2:] for c in cols]

    elegido = alt.selection_point(fields=["grupo"], bind="legend")
    brush = alt.selection_interval(name="brush")
    nube = alt.Chart(datos).mark_circle(size=22).encode(
        x=alt.X("x:Q", title=f"Eje 1 de {espacio.lower()}", scale=alt.Scale(zero=False)),
        y=alt.Y("y:Q", title=f"Eje 2 de {espacio.lower()}", scale=alt.Scale(zero=False)),
        color=alt.condition(brush, alt.Color("grupo:N", scale=escala, title="Grupo (clic para resaltar)",
                                             legend=alt.Legend(orient="top", labelLimit=320)), alt.value("#3a3a42")),
        opacity=alt.condition(elegido, alt.value(0.75), alt.value(0.06)),
        tooltip=[alt.Tooltip("userId:N", title="Usuario"), alt.Tooltip("perfil:N", title="Perfil"),
                 alt.Tooltip("cantidad_valoraciones_total:Q", title="Ratings", format=","),
                 alt.Tooltip("genero_afin:N", title="Género con más afinidad")],
    ).add_params(elegido, brush).properties(width=520, height=440)

    largo = alt.Chart(datos).transform_fold(cols, as_=["variable", "valor"]).transform_calculate(
        nombre="substring(datum.variable, 2)")
    orden = nombres if espacio == "Épocas" else "-x"
    barras = largo.transform_filter(brush).transform_filter(elegido).mark_bar(
        color=ROJO, opacity=0.85, cornerRadiusEnd=2).encode(
        y=alt.Y("nombre:N", title=None, sort=orden),
        x=alt.X("mean(valor):Q", title=eje_barras),
        tooltip=[alt.Tooltip("nombre:N", title="Variable"),
                 alt.Tooltip("mean(valor):Q", title="Seleccionados", format=fmt_barras),
                 alt.Tooltip("count():Q", title="Usuarios")])
    todos = largo.mark_tick(color="#ececf1", thickness=2, size=16).encode(
        y=alt.Y("nombre:N", sort=orden), x="mean(valor):Q")
    panel = (barras + todos).properties(width=300, height=440, title=alt.Title(
        "Perfil de la selección", subtitle="Barra: usuarios marcados · raya blanca: todos",
        anchor="start", fontSize=14, subtitleColor=GRIS_CLARO))
    return alt.hconcat(nube, panel, spacing=30), len(datos)


# ---------------------------------------------------------------------------
# 9 · Un usuario: vecinos y recomendaciones
# ---------------------------------------------------------------------------
def distancias(ids, zg, zd, usuario):
    i = int(np.where(ids == usuario)[0][0])
    dg = np.linalg.norm(zg - zg[i], axis=1) / np.sqrt(zg.shape[1])
    dd = np.linalg.norm(zd - zd[i], axis=1) / np.sqrt(zd.shape[1])
    sim = pd.DataFrame({"userId": ids, "distancia": (dg + dd) / 2})
    return sim[sim["userId"] != usuario].sort_values("distancia")


def recomendar(similitud, ratings, peliculas, usuario, n=6):
    vistas = set(ratings.loc[ratings["userId"] == usuario, "movieId"])
    con_titulo = set(peliculas["movieId"])
    for cantidad in [100, 200, 400, 800]:
        vecinos = similitud.head(cantidad)
        r = ratings[ratings["userId"].isin(vecinos["userId"])
                    & ~ratings["movieId"].isin(vistas)
                    & ratings["movieId"].isin(con_titulo)].merge(vecinos, on="userId")
        r["peso"] = 1 / (r["distancia"] + 0.001)
        r["ponderado"] = r["rating"] * r["peso"]
        rec = r.groupby("movieId").agg(rating_promedio=("rating", "mean"),
                                       cantidad_vecinos=("userId", "nunique"),
                                       suma=("ponderado", "sum"), pesos=("peso", "sum")).reset_index()
        rec["score"] = rec["suma"] / rec["pesos"]
        rec = rec[(rec["cantidad_vecinos"] >= 2) & (rec["rating_promedio"] >= 3.8)]
        if len(rec) >= n:
            break
    rec = rec.sort_values(["score", "cantidad_vecinos"], ascending=False).head(n)
    return rec.merge(peliculas, on="movieId", how="left"), cantidad


def grafico_usuario_en_espacio(usuarios, usuario, vecinos, espacio):
    px, py = ("G_PC1", "G_PC2") if espacio == "Géneros" else ("D_PC1", "D_PC2")
    d = usuarios[["userId", px, py]].rename(columns={px: "x", py: "y"}).copy()
    d["tipo"] = "Resto"
    d.loc[d["userId"].isin(vecinos), "tipo"] = "Los 5 más parecidos"
    d.loc[d["userId"] == usuario, "tipo"] = "Usuario elegido"
    d["orden"] = d["tipo"].map({"Resto": 0, "Los 5 más parecidos": 1, "Usuario elegido": 2})
    escala = alt.Scale(domain=["Resto", "Los 5 más parecidos", "Usuario elegido"],
                       range=["#3a3a42", "#f2b134", ROJO])
    return alt.Chart(d).mark_circle().encode(
        x=alt.X("x:Q", title=f"Eje 1 de {espacio.lower()}", scale=alt.Scale(zero=False)),
        y=alt.Y("y:Q", title=f"Eje 2 de {espacio.lower()}", scale=alt.Scale(zero=False)),
        color=alt.Color("tipo:N", scale=escala, title=None, legend=alt.Legend(orient="top")),
        size=alt.Size("tipo:N", scale=alt.Scale(domain=escala.domain, range=[14, 90, 200]), legend=None),
        order=alt.Order("orden:Q"),
        tooltip=[alt.Tooltip("userId:N", title="Usuario"), "tipo:N"],
    ).properties(height=360, title=alt.Title(f"Espacio de {espacio.lower()}", anchor="start", fontSize=14))


def grafico_afinidades_usuario(fila):
    cols = [c for c in fila.index if c.startswith("g_")]
    t = pd.DataFrame({"genero": [c[2:] for c in cols], "afinidad": fila[cols].astype(float).to_numpy()})
    return alt.Chart(t).mark_bar(cornerRadiusEnd=2).encode(
        y=alt.Y("genero:N", sort="-x", title=None),
        x=alt.X("afinidad:Q", title="Afinidad (respecto de su propio promedio)"),
        color=alt.condition(alt.datum.afinidad > 0, alt.value(ROJO), alt.value(GRIS)),
        tooltip=["genero:N", alt.Tooltip("afinidad:Q", format="+.3f")],
    ).properties(height=380, title=alt.Title("Sus afinidades por género", anchor="start", fontSize=14))


def grafico_epocas_usuario(fila):
    cols = [c for c in fila.index if c.startswith("e_")]
    t = pd.DataFrame({"epoca": [c[2:] for c in cols], "pct": fila[cols].astype(float).to_numpy()})
    return alt.Chart(t).mark_bar(color="#4cc9f0", cornerRadiusEnd=2).encode(
        x=alt.X("epoca:N", sort=ORDEN_EPOCAS, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("pct:Q", title="% de su historial"),
        tooltip=["epoca:N", alt.Tooltip("pct:Q", format=".1f")],
    ).properties(height=380, title=alt.Title("Su historial según época de estreno", anchor="start", fontSize=14))
