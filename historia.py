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
                      COBERTURA, entrenar_desde_csv, preparar_generos)

# el histograma de la muestra contra MovieLens usa 25.000 filas y Altair corta en 5.000
# (como en la sección 1.2 de la Unidad 4): cada gráfico recibe sólo las columnas que usa
alt.data_transformers.disable_max_rows()

# ---------------------------------------------------------------------------
# Colores: un solo canal por dimensión para toda la app
# ---------------------------------------------------------------------------
GRIS = "#6b6b75"
GRIS_CLARO = "#8a8a94"
TEXTO = "#26273d"
FONDO = "#ffffff"
APAGADO = "#dcdce2"
# Paleta de la app: burnt orange, chartreuse, dusty navy y electric blue
QUEMADO, CHARTREUSE, NAVY, ELECTRICO = "#8b3a26", "#c4a22c", "#3e3f66", "#2b30a3"
GRIS_VIA, GRIS_NOTA = "#e3e4e8", "#80818f"

GENEROS_GRUPOS = ["Géneros · Grupo 1","Géneros · Grupo 2","Géneros · Grupo 3","Géneros · Grupo 4"] #Aca falta asignar nombre de grupos
DECADAS_GRUPOS = ["Cine de 2000–2010", "Cine noventero y clásico"]
COLOR_GENERO = alt.Scale(domain=GENEROS_GRUPOS,range=[CHARTREUSE, QUEMADO, NAVY, ELECTRICO])
# los dos grupos de época: electric blue y lavanda. Validado con validate_palette.js de la skill
# dataviz (dusty navy junto a electric blue no se distinguían: diferencia 11,6, mínimo 15).
# El lavanda tiene poco contraste con el fondo blanco: va siempre con etiqueta o leyenda.
EPOCA_RECIENTE, EPOCA_CLASICA = "#3b47c9", "#9aa2ec"
COLOR_DECADA = alt.Scale(domain=DECADAS_GRUPOS, range=[EPOCA_RECIENTE, EPOCA_CLASICA])

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
# 1 · De dónde salen los datos (resúmenes del DAG, ver preparar_origen.py)
# ---------------------------------------------------------------------------
def cargar_origen():
    resumen = pd.read_csv("origen_resumen.csv").set_index("clave")["valor"]
    return {"resumen": resumen,
            "crudo": pd.read_csv("origen_crudo.csv"),
            "poblacion": pd.read_csv("origen_poblacion.csv"),
            "cobertura": pd.read_csv("origen_cobertura.csv")}


def tabla_cruda_con_tmdb(crudo, peliculas):
    """Las mismas filas crudas, después de cruzarlas con TMDB."""
    t = crudo.merge(peliculas[["movieId", "titulo", "fecha_estreno", "genres"]], on="movieId", how="left")
    t["genres"] = t["genres"].fillna("").map(
        lambda g: ", ".join(GENERO_ES.get(x, x) for x in g.split("|") if x))
    return t.rename(columns={"titulo": "título", "fecha_estreno": "estreno", "genres": "géneros"})


def tabla_poblacion(poblacion, silver):
    """Usuarios por intervalo de ratings: toda MovieLens y la muestra de 5.000."""
    # intervalos iguales en escala log, anclados en 100 y con 150 como borde exacto:
    # ninguna barra mezcla a los que entran a la muestra o al modelo con los que no
    paso = np.log10(MIN_RATINGS / 100) / 2
    bordes = 100 * 10 ** (np.arange(-23, 32) * paso)
    partes = []
    for nombre, serie in [("MovieLens", poblacion["cantidad"]),
                          ("Muestra", silver["cantidad_valoraciones_total"])]:
        cortes = pd.cut(serie, bordes, right=False)
        t = cortes.value_counts(sort=False).rename_axis("intervalo").reset_index(name="usuarios")
        t["desde"] = [i.left for i in t["intervalo"]]
        t["hasta"] = [i.right for i in t["intervalo"]]
        t["grupo"] = nombre
        partes.append(t.drop(columns="intervalo"))
    t = pd.concat(partes, ignore_index=True)
    t = t[t["usuarios"] > 0]
    t["tramo"] = np.select(
        [t["desde"] < 100, (t["grupo"] == "Muestra") & (t["desde"] < MIN_RATINGS)],
        ["Menos de 100: no entra a la muestra", f"De 100 a {MIN_RATINGS - 1}: no entra al modelo"],
        "Entra")
    return t


def grafico_muestra_vs_poblacion(poblacion, silver):
    """Histogramas superpuestos: MovieLens (>= 100 ratings) y la muestra, cada uno en % de su grupo.

    Un control cambia la cantidad de intervalos. Vega-Lite no deja que un parámetro mueva
    maxbins, así que hay una capa por cada valor posible y el control muestra sólo la suya.
    """
    pob = poblacion["cantidad"]
    elegibles = pob[pob >= 100]
    nombres = [f"MovieLens: los {fmt(len(elegibles))} con 100 ratings o más",
               f"Muestra: los {fmt(len(silver))} elegidos al azar"]
    largo = pd.concat([
        pd.DataFrame({"log": np.log10(elegibles.sample(20_000, random_state=SEMILLA)), "grupo": nombres[0]}),
        pd.DataFrame({"log": np.log10(silver["cantidad_valoraciones_total"]), "grupo": nombres[1]})])
    intervalos = alt.param(name="intervalos", value=30,
                           bind=alt.binding_range(min=10, max=60, step=10, name="Cantidad de intervalos "))
    grupo = alt.selection_point(fields=["grupo"], bind="legend")
    marcas = [2, np.log10(MIN_RATINGS), np.log10(500), 3, 4]
    capas = []
    for k in [10, 20, 30, 40, 50, 60]:
        capas.append(alt.Chart(largo).transform_filter(f"intervalos == {k}").transform_bin(
            "b", "log", bin=alt.Bin(maxbins=k, extent=[2, 4.5])).transform_aggregate(
            n="count()", groupby=["b", "b_end", "grupo"]).transform_joinaggregate(
            total="sum(n)", groupby=["grupo"]).transform_calculate(pct="datum.n / datum.total").mark_bar(
            binSpacing=0, stroke=FONDO, strokeWidth=0.4).encode(
            x=alt.X("b:Q", title="Ratings por usuario (escala log)", scale=alt.Scale(domain=[2, 4.5]),
                    axis=alt.Axis(values=marcas, labelExpr="format(pow(10, datum.value), ',.0f')")),
            x2="b_end:Q",
            y=alt.Y("pct:Q", title="% de usuarios del grupo", stack=None, axis=alt.Axis(format="%")),
            y2=alt.datum(0),
            color=alt.Color("grupo:N", title=None, scale=alt.Scale(domain=nombres, range=[NAVY, QUEMADO]),
                            legend=alt.Legend(orient="top", direction="vertical", labelLimit=0)),
            opacity=alt.condition(grupo, alt.value(0.55), alt.value(0.05)),
            tooltip=[alt.Tooltip("grupo:N", title="Grupo"),
                     alt.Tooltip("pct:Q", format=".1%", title="% del grupo")]))
    return alt.layer(*capas).add_params(intervalos, grupo).properties(height=360, title=titulo(
        "La muestra al azar tiene la misma forma que MovieLens",
        ["Cada histograma está en % de su grupo, para compararlos aunque uno sea 16 veces más grande.",
         "Mové el control para cambiar los intervalos; clic en la leyenda para dejar uno solo."]))


def grafico_cobertura(cobertura, resumen):
    n = int(resumen["peliculas_consultadas"])
    total = int(resumen["peliculas_muestra"])
    fila = cobertura.set_index("peliculas").loc[n]
    x = alt.X("peliculas:Q", title="Películas, de la más vista a la menos vista (escala log)",
              scale=alt.Scale(type="log", domain=[1, total]),
              axis=alt.Axis(values=[1, 10, 100, 1000, 10000], format="~s"))
    y = alt.Y("cubierto:Q", title="Ratings de la muestra que cubren", scale=alt.Scale(domain=[0, 1]),
              axis=alt.Axis(format="%"))
    curva = alt.Chart(cobertura).mark_area(
        color=GRIS, opacity=0.2, line={"color": GRIS, "strokeWidth": 2}).encode(
        x=x, y=y, tooltip=[alt.Tooltip("peliculas:Q", title="Películas", format=","),
                           alt.Tooltip("cubierto:Q", title="Ratings cubiertos", format=".1%")])
    consultadas = alt.Chart(cobertura[cobertura["peliculas"] <= n]).mark_area(color=QUEMADO, opacity=0.35).encode(
        x=x, y=y)
    marca = pd.DataFrame({"peliculas": [n], "cubierto": [fila["cubierto"]],
                          "t": [f"{fmt(n)} películas → {dec(fila['cubierto'] * 100)} % de los ratings"]})
    punto = alt.Chart(marca).mark_circle(color=QUEMADO, size=110, opacity=1).encode(x=x, y=y)
    texto = alt.Chart(marca).mark_text(align="right", dx=-10, dy=14, color=QUEMADO, fontWeight="bold",
                                       fontSize=12).encode(x=x, y=y, text="t:N")
    return (curva + consultadas + punto + texto).properties(height=400, title=titulo(
        [f"Las {fmt(n)} películas más vistas cubren", f"el {dec(fila['cubierto'] * 100, 0)} % de los ratings de la muestra"],
        f"La muestra tiene {fmt(total)} películas distintas; la mayoría tiene muy pocos ratings."))


TRAMOS_RECORRIDO = [("Datos", 1, 4, "De los votos a una tabla de usuarios", QUEMADO),
                    ("Modelo", 5, 7, "Del gusto a los grupos", CHARTREUSE),
                    ("Resultado", 8, 9, "De los grupos a recomendar", ELECTRICO)]


def tabla_recorrido(resumen):
    """Las nueve estaciones del proyecto, con la decisión de cada una."""
    filas = [
        ("Votos crudos", f"MovieLens 32M", f"{resumen['ratings_total'] / 1e6:.0f} M de votos", "Datos", "paso-crudo"),
        ("Selección", "≥ 100 ratings,", f"{fmt(resumen['usuarios_muestra'])} al azar", "Datos", "paso-muestra"),
        ("TMDB", f"{fmt(resumen['peliculas_consultadas'])} películas:", "género y fecha", "Datos", "paso-tmdb"),
        ("Tabla silver", "una fila", "por usuario", "Datos", "paso-silver"),
        ("Afinidad", f"≥ {MIN_RATINGS} ratings,", "promedio amortiguado", "Modelo", "trampa"),
        ("PCA", "géneros y épocas", "por separado", "Modelo", "pca"),
        ("K-means", "k = 2", "en cada bloque", "Modelo", "grupos"),
        ("Perfiles", "2 × 2 =", "4 perfiles", "Resultado", "perfiles"),
        ("Recomendación", "películas de", "usuarios parecidos", "Resultado", "recomendador"),
    ]
    t = pd.DataFrame(filas, columns=["nombre", "linea1", "linea2", "tramo", "ancla"])
    t.insert(0, "n", range(1, len(t) + 1))
    return t


def grafico_recorrido(estaciones):
    """De la tabla de votos a la recomendación: una línea gris con un tinte por tramo."""
    dominio_x = alt.Scale(domain=[0.4, 9.6], nice=False)
    dominio_y = alt.Scale(domain=[0, 1], nice=False)
    x = lambda campo="n:Q": alt.X(campo, scale=dominio_x, axis=None)
    y = lambda campo="y:Q": alt.Y(campo, scale=dominio_y, axis=None)
    color = alt.Color("tramo:N", legend=None, scale=alt.Scale(
        domain=[t[0] for t in TRAMOS_RECORRIDO], range=[t[4] for t in TRAMOS_RECORRIDO]))
    alto_via = 0.55
    est = estaciones.assign(y=alto_via)
    tramos = pd.DataFrame(TRAMOS_RECORRIDO, columns=["tramo", "desde", "hasta", "titulo", "color"])

    via = alt.Chart(pd.DataFrame({"n": [1, 9], "y": [alto_via] * 2})).mark_line(
        color=GRIS_VIA, strokeWidth=1.5).encode(x=x(), y=y())
    segmentos = pd.DataFrame([{"tramo": t, "n": n, "y": alto_via}
                              for t, a, b, *_ in TRAMOS_RECORRIDO for n in (a, b)])
    tramo_via = alt.Chart(segmentos).mark_line(strokeWidth=2.5, strokeCap="round").encode(
        x=x(), y=y(), color=color, detail="tramo:N")
    halo = alt.Chart(est).mark_circle(size=260, opacity=0.18).encode(x=x(), y=y(), color=color)
    punto = alt.Chart(est).mark_circle(size=55, opacity=1).encode(
        x=x(), y=y(), color=color, tooltip=[alt.Tooltip("nombre:N", title="Paso"),
                                            alt.Tooltip("linea1:N", title="Decisión"),
                                            alt.Tooltip("linea2:N", title=" ")])
    base = alt.Chart(est)
    numero = base.mark_text(dy=-20, fontSize=10, fontWeight=500, color=GRIS_NOTA).encode(
        x=x(), y=y(), text=alt.Text("n:Q", format="02d"))
    nombre = base.mark_text(dy=22, fontSize=13, fontWeight=600, color=NAVY).encode(x=x(), y=y(), text="nombre:N")
    linea1 = base.mark_text(dy=40, fontSize=11, color=GRIS_NOTA).encode(x=x(), y=y(), text="linea1:N")
    linea2 = base.mark_text(dy=55, fontSize=11, color=GRIS_NOTA).encode(x=x(), y=y(), text="linea2:N")

    tramos = tramos.assign(a=tramos["desde"] - 0.3, b=tramos["hasta"] + 0.3, y=0.97,
                           m=(tramos["desde"] + tramos["hasta"]) / 2, etiqueta=tramos["titulo"].str.upper())
    raya = alt.Chart(tramos).mark_rule(strokeWidth=1).encode(x=x("a:Q"), x2="b:Q", y=y(), color=color)
    titulo_tramo = alt.Chart(tramos).mark_text(dy=-8, fontSize=10, fontWeight=600).encode(
        x=x("m:Q"), y=y(), text="etiqueta:N", color=color)
    return (raya + titulo_tramo + via + tramo_via + halo + punto + numero + nombre + linea1 + linea2).properties(
        height=175).configure_view(stroke=None)


# ---------------------------------------------------------------------------
# Actividad de la muestra
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


# ---------------------------------------------------------------------------
# 2 · Lo que se pierde en el pipeline
# ---------------------------------------------------------------------------


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
                          "n_mediana": float(sub["n"].median()),
                          "p05": q[0], "p25": q[1], "mediana": q[2], "p75": q[3], "p95": q[4]})
    return pd.DataFrame(filas)


def grafico_trampa(tabla):
    orden = [t[2] for t in TRAMOS_N]
    amort = f"Amortiguado (m = {SUAVIZADO})"
    crudo = tabla[tabla["version"] == "Promedio crudo"].set_index("tramo")
    suave = tabla[tabla["version"] == amort].set_index("tramo")
    ancho_1 = crudo.loc["1", "p95"] - crudo.loc["1", "p05"]
    ancho_50 = crudo.loc["50 o más", "p95"] - crudo.loc["50 o más", "p05"]
    # zoom del panel amortiguado: simétrico y con algo de aire sobre el 90 % central
    tope = float(np.ceil(max(suave["p95"].max(), -suave["p05"].min()) * 10 + 1) / 10)
    zoom = (2.5 + 2) / (2 * tope)
    # etiquetas en dos líneas en el panel amortiguado: el tramo y su factor n/(n+m)
    # (para los tramos con varios valores de n, el factor de la cantidad típica: la mediana)
    mapa = {t: [t, f"×{dec(n / (n + SUAVIZADO), 2)}"] for t, n in suave["n_mediana"].items()}
    expr_factor = f"{mapa}[datum.value]".replace("'", '"')

    def panel(version, dominio, cabecera, label_expr=None, titulo_y=None):
        d = tabla[tabla["version"] == version]
        x = alt.X("tramo:O", sort=orden, title="Películas del género que calificó el usuario",
                  axis=alt.Axis(labelAngle=0, labelExpr=label_expr) if label_expr
                  else alt.Axis(labelAngle=0))
        y = alt.Y("p05:Q", title=titulo_y, scale=alt.Scale(domain=dominio, nice=False))
        base = alt.Chart(d)
        banda = base.mark_area(opacity=0.25, color=QUEMADO).encode(x=x, y=y, y2="p95:Q")
        caja = base.mark_area(opacity=0.5, color=QUEMADO).encode(x=x, y="p25:Q", y2="p75:Q")
        linea = base.mark_line(color=TEXTO, strokeWidth=1.5).encode(x=x, y="mediana:Q")
        # puntos invisibles: sólo para el tooltip
        tip = base.mark_point(opacity=0, size=400).encode(
            x=x, y="mediana:Q",
            tooltip=[alt.Tooltip("tramo:O", title="Películas del género"),
                     alt.Tooltip("pares:Q", title="Pares usuario × género", format=","),
                     alt.Tooltip("p05:Q", title="Percentil 5", format="+.2f"),
                     alt.Tooltip("p25:Q", title="Percentil 25", format="+.2f"),
                     alt.Tooltip("mediana:Q", title="Mediana", format="+.2f"),
                     alt.Tooltip("p75:Q", title="Percentil 75", format="+.2f"),
                     alt.Tooltip("p95:Q", title="Percentil 95", format="+.2f")])
        cero = base.mark_rule(color=GRIS, strokeDash=[3, 3]).encode(y=alt.datum(0))
        return alt.layer(banda, caja, linea, tip, cero).properties(
            width=440, height=260,
            title=alt.Title(cabecera[0], subtitle=cabecera[1], anchor="start", fontSize=13,
                            fontWeight="normal", color=TEXTO, subtitleColor=GRIS_CLARO,
                            subtitleFontSize=11))

    izq = panel("Promedio crudo", [-2.5, 2],
                ["Promedio crudo", ["Escala de −2,5 a +2", " "]],
                titulo_y="Promedio del género − promedio propio")
    der = panel(amort, [-tope, tope],
                [amort, [f"Escala ampliada ×{dec(zoom, 0)} (de −{dec(tope)} a +{dec(tope)})",
                         "Solo se aleja de cero quien vio muchas películas del género"]],
                label_expr=expr_factor)
    return alt.hconcat(izq, der, spacing=40).resolve_scale(y="independent").properties(title=titulo(
        f"Con una sola película, el «gusto» por un género varía {dec(ancho_1 / ancho_50, 0)} veces más "
        "que con cincuenta",
        ["Banda clara: 90 % central de los pares usuario × género; banda oscura: 50 % central; línea: mediana.",
         "Debajo de cada tramo del panel derecho, el factor n/(n + m) por el que se multiplica la diferencia "
         "(en los tramos anchos, el de la cantidad típica)."]))


UMBRALES = [100, 150, 200, 250, 300, 350, 400]
POCAS = 5   # una afinidad calculada con menos de 5 películas del género es «frágil»


def tabla_umbral(silver, modelos):
    """Costo (usuarios que quedan) y beneficio (afinidades frágiles, separación) de cada mínimo."""
    generos = modelos.generos["preparar"]["features"].generos_
    filas = []
    for x in UMBRALES:
        D = silver[silver["cantidad_valoraciones"] >= x]
        n = D[[f"n_{g}" for g in generos]].fillna(0).to_numpy()
        Z = preparar_generos().fit_transform(D)
        km = KMeans(n_clusters=2, random_state=SEMILLA, n_init=10).fit(Z)
        filas.append({"minimo": x, "usuarios": len(D), "pct": len(D) / len(silver),
                      "fragiles": (n < POCAS).mean(), "silhouette": silhouette_score(Z, km.labels_)})
    return pd.DataFrame(filas)


def grafico_umbral(tabla):
    fila = tabla.set_index("minimo").loc[MIN_RATINGS]
    tabla = tabla.assign(
        t=[f"{fmt(u)} · {dec(p * 100, 0)} %" for u, p in zip(tabla["usuarios"], tabla["pct"])],
        tf=[f"{dec(v * 100)} %" for v in tabla["fragiles"]])
    x = alt.X("minimo:O", title="Mínimo de ratings por usuario", axis=alt.Axis(labelAngle=0))
    color = alt.condition(alt.datum.minimo == MIN_RATINGS, alt.value(QUEMADO), alt.value(GRIS))
    tip = [alt.Tooltip("minimo:O", title="Mínimo de ratings"),
           alt.Tooltip("usuarios:Q", title="Usuarios que quedan", format=","),
           alt.Tooltip("pct:Q", title="De la muestra", format=".0%"),
           alt.Tooltip("fragiles:Q", title=f"Afinidades con menos de {POCAS} películas", format=".1%")]
    base = alt.Chart(tabla)
    # costo: piruletas (palito fino + punto) desde cero; las etiquetas dan el número, el eje Y sobra
    y_u = alt.Y("usuarios:Q", axis=None, scale=alt.Scale(domain=[0, 5600]))
    # el palito es una barra de 2 px: sale de cero igual que una barra
    palitos = base.mark_bar(size=2).encode(x=x.axis(None), y=y_u, color=color)
    cabezas = base.mark_circle(size=90, opacity=1, stroke=FONDO, strokeWidth=2).encode(
        x=x.axis(None), y=y_u, color=color, tooltip=tip)
    texto = base.mark_text(dy=-14, fontSize=11, color=TEXTO).encode(x=x.axis(None), y=y_u, text="t:N")
    costo = (palitos + cabezas + texto).properties(height=200, title=alt.Title(
        "Usuarios que quedan", anchor="start", fontSize=13, fontWeight="normal", color=TEXTO))
    # beneficio: cuántas afinidades se calculan con muy pocas películas
    y_f = alt.Y("fragiles:Q", axis=None, scale=alt.Scale(domain=[0, tabla["fragiles"].max() * 1.35]))
    linea = base.mark_line(color=GRIS, strokeWidth=1.5).encode(x=x, y=y_f)
    puntos = base.mark_circle(size=70, opacity=1, stroke=FONDO, strokeWidth=2).encode(
        x=x, y=y_f, color=color, tooltip=tip)
    texto_f = base.mark_text(dy=-12, fontSize=11, color=TEXTO).encode(x=x, y=y_f, text="tf:N")
    beneficio = (linea + puntos + texto_f).properties(height=130, title=alt.Title(
        f"Afinidades frágiles (menos de {POCAS} películas del género)", anchor="start", fontSize=13,
        fontWeight="normal", color=TEXTO))
    return alt.vconcat(costo, beneficio, spacing=14).properties(title=titulo(
        f"Con {MIN_RATINGS} ratings como mínimo quedan {fmt(fila['usuarios'])} usuarios: "
        f"el {dec(fila['pct'] * 100, 0)} %",
        ["La muestra ya se armó con usuarios de 100 ratings o más: por eso el corte en 100 conserva el 100 %.",
         "Cada escalón deja afuera a cientos de usuarios y reduce cada vez menos las afinidades frágiles."]))


def alt_umbral(tabla):
    """Texto alternativo del gráfico del umbral: la conclusión y la forma de los datos."""
    t = tabla.set_index("minimo")
    a, b, z = t.index[0], MIN_RATINGS, t.index[-1]
    return (f"Dos gráficos alineados por el mínimo de ratings por usuario, de {a} a {z}. "
            f"Arriba, usuarios que quedan: {fmt(t.loc[a, 'usuarios'])} con {a}, {fmt(t.loc[b, 'usuarios'])} "
            f"con {b} (el corte elegido) y {fmt(t.loc[z, 'usuarios'])} con {z}. Abajo, afinidades frágiles: "
            f"{dec(t.loc[a, 'fragiles'] * 100)} %, {dec(t.loc[b, 'fragiles'] * 100)} % y "
            f"{dec(t.loc[z, 'fragiles'] * 100)} %. La mayor mejora está entre {a} y {b}; después cada "
            "escalón mejora menos y sigue perdiendo usuarios.")


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
        domain=["Géneros", "Épocas"], range=[QUEMADO, ELECTRICO]), legend=None)
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


# ---------------------------------------------------------------------------
# Las variables del modelo y los ejes de las PCA, con nombre
# ---------------------------------------------------------------------------
EPOCA_CORTA = {
    "prop_antes_1980": "antes de 1980",
    "prop_1980s": "los 80",
    "prop_1990s": "los 90",
    "prop_2000s": "los 2000",
    "prop_2010s": "los 2010",
    "prop_2020_mas": "2020 en adelante",
}

NOMBRES_EJES = {
    ("Géneros", 1): "Drama ↔ Aventura",
    ("Géneros", 2): "Romance ↔ Acción",
    ("Géneros", 3): "Terror ↔ Historia",
    ("Géneros", 4): "Romance ↔ Animación",
    ("Géneros", 5): "Misterio ↔ Western",
    ("Géneros", 6): "Western ↔ Crimen",
    ("Géneros", 7): "Crimen ↔ Música",
    ("Géneros", 8): "Crimen ↔ Romance",

    ("Épocas", 1): "Clásicos ↔ 2000–2010",
    ("Épocas", 2): "Pre-80 ↔ 90s",
    ("Épocas", 3): "2010s ↔ 2000s",
}

EJES = {}
EJES_CORTOS = {}

def tabla_cargas_todas(modelos):
    """Cuánto pesa cada variable en cada eje de las dos PCA (una fila por variable y eje)."""
    filas = []
    for bloque, modelo in [("Géneros", modelos.generos), ("Épocas", modelos.decadas)]:
        prep = modelo["preparar"]
        if bloque == "Géneros":
            nombres = [GENERO_ES.get(g, g) for g in prep["features"].generos_]
        else:
            nombres = [EPOCA_CORTA[c] for c in prep["features"].columnas_]
        pca = prep["pca"]
        for i, (componente, var) in enumerate(zip(pca.components_, pca.explained_variance_ratio_), start=1):
            for nombre, carga in zip(nombres, componente):
                filas.append({"bloque": bloque, "eje": i, "variable": nombre, "carga": carga, "varianza": var})
    return pd.DataFrame(filas)


def _polo(serie, cronologico=False, maximo=3):
    """Los nombres que más empujan hacia un lado (los que pesan al menos la mitad del más fuerte)."""
    fuertes = serie[serie.abs() >= serie.abs().max() * 0.5].abs().sort_values(ascending=False).head(maximo).index
    if cronologico:   # las épocas se leen mejor en orden
        orden = list(EPOCA_CORTA.values())
        fuertes = sorted(fuertes, key=orden.index)
    nombres = [n.lower() for n in fuertes]
    return " y ".join([", ".join(nombres[:-1]), nombres[-1]]) if len(nombres) > 1 else nombres[0]


def fijar_ejes(cargas):
    """Le pone nombre a cada eje según qué variables lo empujan para cada lado."""
    EJES.clear()
    EJES_CORTOS.clear()
    for (bloque, eje), grupo in cargas.groupby(["bloque", "eje"]):
        c = grupo.set_index("variable")["carga"]
        prefijo = "más " if bloque == "Géneros" else "cine de "
        lados = (c[c < 0], c[c > 0])
        EJES[(bloque, eje)] = tuple(prefijo + _polo(l, bloque == "Épocas") for l in lados)
        EJES_CORTOS[(bloque, eje)] = tuple(prefijo + _polo(l, bloque == "Épocas", maximo=2) for l in lados)
    return EJES


def titulo_eje(bloque, eje):
    """Título de eje en dos renglones: el número y hacia dónde va cada lado."""
    if (bloque, eje) not in EJES_CORTOS:
        return f"Eje {eje} de {bloque.lower()}"
    negativo, positivo = EJES_CORTOS[(bloque, eje)]
    return [f"Eje {eje} de {bloque.lower()}", f"← {negativo}  ·  {positivo} →"]


def tabla_ejes(cargas):
    """Un renglón por eje: cuánta varianza explica y qué opone."""
    filas = []

    for (bloque, eje), grupo in cargas.groupby(
        ["bloque", "eje"],
        sort=False
    ):
        negativo, positivo = EJES.get(
            (bloque, eje),
            ("", "")
        )

        filas.append({
            "Bloque": bloque,
            "Eje": NOMBRES_EJES[(bloque, eje)],
            "Varianza que explica": grupo["varianza"].iloc[0],
            "Un extremo (−)": negativo,
            "El otro extremo (+)": positivo,
        })

    return pd.DataFrame(filas)

def grafico_cargas_eje(cargas, bloque):
    """Elegí un eje y mirá qué variables lo empujan hacia cada lado."""
    datos = cargas[cargas["bloque"] == bloque].copy()
    datos["lado"] = np.where(datos["carga"] < 0, "Empuja hacia la izquierda (−)", "Empuja hacia la derecha (+)")
    n = int(datos["eje"].max())

    nombre = "eje_" + (
        "generos" if bloque == "Géneros" else "epocas"
    )

    opciones = list(range(1, n + 1))

    etiquetas = [
        NOMBRES_EJES[(bloque, i)]
        for i in opciones
    ]

    eje = alt.param(
        name=nombre,
        value=1,
        bind=alt.binding_radio(
            options=opciones,
            labels=etiquetas,
            name=f"Eje de {bloque.lower()}: "
        )
    )
    # en Streamlit la altura incluye título, leyenda y eje: se suma lo que ocupan
    alto = 28 * datos["variable"].nunique() + 190
    return alt.Chart(datos).transform_filter(f"datum.eje == {nombre}").mark_bar(size=14, cornerRadiusEnd=3).encode(
        x=alt.X("carga:Q", title="Carga: cuánto pesa la variable en el eje", scale=alt.Scale(domain=[-0.9, 0.9])),
        y=alt.Y("variable:N", title=None, sort=alt.EncodingSortField("carga", order="ascending"),
                axis=alt.Axis(labelLimit=0, minExtent=105, labelOverlap=False)),
        color=alt.Color("lado:N", title=None, legend=alt.Legend(orient="top", direction="vertical", labelLimit=0),
                        scale=alt.Scale(domain=["Empuja hacia la izquierda (−)", "Empuja hacia la derecha (+)"],
                                        range=[QUEMADO, EPOCA_RECIENTE])),
        tooltip=[alt.Tooltip("variable:N", title="Variable"), alt.Tooltip("eje:Q", title="Eje"),
                 alt.Tooltip("carga:Q", title="Carga", format="+.2f"),
                 alt.Tooltip("varianza:Q", title="Varianza del eje", format=".0%")],
    ).add_params(eje).properties(height=alto, title=titulo(
        f"Qué variables arman cada eje de {bloque.lower()}",
        "Elegí el eje con los botones de abajo."))


def tabla_fila_usuario(modelos, usuario):
    """Las 22 variables de un usuario, con las cuentas a la vista."""
    D = modelos.entrenamiento
    fila = D.loc[usuario]
    general = fila["rating_promedio_general"]
    generos = []
    for g in modelos.generos["preparar"]["features"].generos_:
        n = 0 if pd.isna(fila[f"n_{g}"]) else int(fila[f"n_{g}"])
        media = general if pd.isna(fila[g]) else float(fila[g])
        factor = n / (n + SUAVIZADO)
        generos.append({"Género": GENERO_ES.get(g, g), "Películas que calificó": n,
                        "Su promedio en el género": media, "Diferencia con su promedio": media - general,
                        f"Peso n/(n+{SUAVIZADO})": factor, "Afinidad": factor * (media - general)})
    generos = pd.DataFrame(generos).sort_values("Afinidad", ascending=False)
    epocas = []
    for c in modelos.decadas["preparar"]["features"].columnas_:
        epocas.append({"Época": EPOCA_CORTA[c].capitalize(), "Parte de su historial": float(fila[c]),
                       "arcsen(√p)": float(np.arcsin(np.sqrt(fila[c])))})
    return generos, pd.DataFrame(epocas), general, int(fila["cantidad_valoraciones"])


def grafico_fanaticos(usuarios):
    """Mapa de calor del espacio de géneros con resaltado cruzado desde las barras de género."""
    datos = usuarios[["G_PC1", "G_PC2", "genero_afin"]]
    genero = alt.selection_point(fields=["genero_afin"], name="genero")
    x = alt.X("G_PC1:Q", bin=alt.Bin(step=0.8), title=titulo_eje("Géneros", 1))
    y = alt.Y("G_PC2:Q", bin=alt.Bin(step=0.8), title=titulo_eje("Géneros", 2))
    calor = alt.Chart(datos).mark_rect().encode(
        x=x, y=y, color=alt.Color("count():Q", title="Usuarios por celda",
                                  scale=alt.Scale(type="sqrt", range=["#f4f1e4", NAVY]),
                                  legend=alt.Legend(orient="top", gradientLength=200)))
    circulos = alt.Chart(datos).transform_filter(genero).mark_point(
        filled=False, color=QUEMADO, strokeWidth=1.6).encode(
        x=x, y=y, size=alt.Size("count():Q", title="Usuarios del género elegido",
                                scale=alt.Scale(range=[8, 260]), legend=alt.Legend(orient="top")))
    barras = alt.Chart(datos).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("genero_afin:N", sort="-y", title="Género con más afinidad", axis=alt.Axis(labelAngle=-40)),
        y=alt.Y("count():Q", title="Usuarios"),
        color=alt.condition(genero, alt.value(ELECTRICO), alt.value(APAGADO)),
        tooltip=[alt.Tooltip("genero_afin:N", title="Género"), alt.Tooltip("count():Q", title="Usuarios")],
    ).add_params(genero).properties(height=190)
    return alt.vconcat((calor + circulos).properties(height=360), barras).resolve_legend(
        color="independent", size="independent").properties(title=titulo(
            "Cada género tiene su zona en el espacio de géneros",
            "Clic en una barra: los círculos muestran dónde están esos usuarios. Doble clic para volver."))


# ---------------------------------------------------------------------------
# 6 · Cuántos grupos
# ---------------------------------------------------------------------------
# colores de la sección «Cuántos grupos», validados con validate_palette.js de la skill dataviz
SENALES = {"Dispersión que queda (codo)": QUEMADO, "Separación (silhouette)": CHARTREUSE,
           "Estabilidad (remuestreo)": EPOCA_RECIENTE}
COLORES_K = [EPOCA_RECIENTE, CHARTREUSE, QUEMADO, EPOCA_CLASICA, "#1f9a8a"]


def estabilidad(Z, k, repeticiones=20):
    """ARI medio entre el K-means con todos los usuarios y K-means entrenados con el 80 % al azar."""
    rng = np.random.default_rng(SEMILLA)
    completo = KMeans(n_clusters=k, random_state=SEMILLA, n_init=10).fit_predict(Z)
    aris = []
    for _ in range(repeticiones):
        idx = rng.choice(len(Z), size=int(0.8 * len(Z)), replace=False)
        parcial = KMeans(n_clusters=k, random_state=int(rng.integers(1_000_000)), n_init=10).fit(Z[idx]).predict(Z)
        aris.append(adjusted_rand_score(completo, parcial))
    return float(np.mean(aris))


def evaluar_k(modelos):
    _, zg, zd = espacios(modelos)
    filas = []
    for bloque, Z in [("Géneros", zg), ("Épocas", zd)]:
        base = None
        for k in range(1, 7):
            km = KMeans(n_clusters=k, random_state=SEMILLA, n_init=10).fit(Z)
            base = km.inertia_ if k == 1 else base
            filas.append({"bloque": bloque, "k": k, "inercia": km.inertia_ / base,
                          "silhouette": silhouette_score(Z, km.labels_) if k > 1 else np.nan,
                          "estabilidad": estabilidad(Z, k) if k > 1 else np.nan})
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


def particiones_generos(modelos):
    """Grupo de cada usuario en el espacio de géneros para k de 2 a 5 (para el gráfico de elegir k)."""
    _, zg, _ = espacios(modelos)
    partes = []
    for k in range(2, 6):
        etiquetas = KMeans(n_clusters=k, random_state=SEMILLA, n_init=10).fit_predict(zg)
        partes.append(pd.DataFrame({"k": k, "x": zg[:, 0], "y": zg[:, 1],
                                    "grupo": [f"Grupo {g + 1}" for g in etiquetas]}))
    return pd.concat(partes, ignore_index=True)


def silhouette_epocas_por_componentes(modelos):
    """Silhouette de épocas guardando 2 o 3 componentes: el 80 % de la PCA queda justo en el borde."""
    D = modelos.entrenamiento
    X = modelos.decadas["preparar"][:-1].transform(D)   # escalado, antes de la PCA
    varianza = np.cumsum(PCA(random_state=SEMILLA).fit(X).explained_variance_ratio_)
    salida = {}
    for k in (2, 3):
        Z = PCA(n_components=k, random_state=SEMILLA).fit_transform(X)
        etiquetas = KMeans(n_clusters=2, random_state=SEMILLA, n_init=10).fit_predict(Z)
        salida[k] = {"varianza": varianza[k - 1], "silhouette": silhouette_score(Z, etiquetas)}
    return salida


def grafico_senales(evaluacion):
    """A · Las tres señales por bloque, en un mismo eje de 0 a 1, con k = 2 marcado."""
    largo = evaluacion.dropna().melt(id_vars=["bloque", "k"], value_vars=["inercia", "silhouette", "estabilidad"],
                                     var_name="medida", value_name="valor")
    largo["medida"] = largo["medida"].map({"inercia": "Dispersión que queda (codo)",
                                           "silhouette": "Separación (silhouette)",
                                           "estabilidad": "Estabilidad (remuestreo)"})
    color = alt.Color("medida:N", title=None, scale=alt.Scale(domain=list(SENALES), range=list(SENALES.values())),
                      legend=alt.Legend(orient="top", labelLimit=0, columnPadding=36, symbolOffset=6))
    escala_x = alt.Scale(domain=[1.6, 7.2], nice=False)
    x = alt.X("k:Q", title="Cantidad de grupos (k)", scale=escala_x,
              axis=alt.Axis(values=[2, 3, 4, 5, 6], format="d", grid=False))
    y = alt.Y("valor:Q", title=None, scale=alt.Scale(domain=[0, 1]),
              axis=alt.Axis(values=[0, 0.25, 0.5, 0.75, 1], gridColor="#e3e4e8", domain=False, ticks=False))

    def panel(bloque):
        datos = largo[largo["bloque"] == bloque]
        k_elegido = 4 if bloque == "Géneros" else 2
        banda = alt.Chart(
            pd.DataFrame({"a": [k_elegido - 0.25],"b": [k_elegido + 0.25]})).mark_rect(color="#eef0fb").encode(
            x=alt.X("a:Q", scale=escala_x), x2="b:Q")
        rotulo_k = alt.Chart(pd.DataFrame({"k": [k_elegido]})).mark_text(
            text="k elegido", dy=-4, fontSize=11, color=TEXTO, baseline="bottom").encode(x="k:Q", y=alt.datum(1))
        lineas = alt.Chart(datos).mark_line(strokeWidth=2).encode(x=x, y=y, color=color)
        puntos = alt.Chart(datos).mark_circle(size=70, opacity=1, stroke=FONDO, strokeWidth=2).encode(
            x=x, y=y, color=color, tooltip=[alt.Tooltip("medida:N", title="Medida"), "k:Q",
                                            alt.Tooltip("valor:Q", title="Valor", format=".3f")])
        # nombres al final de cada línea; si dos quedan muy cerca, se separan unos píxeles
        fin = datos[datos["k"] == 6].sort_values("valor").assign(t=lambda d: d["medida"].str.split(" ").str[0])
        cerca = fin["valor"].diff().fillna(1).abs() < 0.07
        fin["y_rotulo"] = (fin["valor"] + np.where(cerca, 0.035, 0)
                           - np.where(cerca.shift(-1, fill_value=False), 0.035, 0))
        rotulos = alt.Chart(fin).mark_text(align="left", dx=8, fontSize=11, color=TEXTO).encode(
            x=x, y=alt.Y("y_rotulo:Q", scale=alt.Scale(domain=[0, 1])), text="t:N")
        return (banda + rotulo_k + lineas + puntos + rotulos).properties(width=330, height=250, title=bloque)

    est = evaluacion.set_index(["bloque", "k"])["estabilidad"]
    return alt.hconcat(panel("Géneros"), panel("Épocas"), spacing=40).properties(
        title=titulo("Mayor separación con K=2; mayor detalle descriptivo con K=4",
            [f"Géneros: K=2 tiene estabilidad "
                f"{dec(est['Géneros', 2], 2)} y K=4 tiene "
                f"{dec(est['Géneros', 4], 2)}. "
                "Elegimos K=4 para explorar más matices.",
                "Épocas: mantenemos K=2. "
                "La banda celeste señala la elección de cada bloque."
            ])).configure_view(stroke=None)

def grafico_k2_k4(particiones):
    """
    Compara K=2 con K=4 en el espacio PCA de géneros.
    Los dos gráficos tienen el mismo formato.
    """

    datos_k2 = particiones[particiones["k"] == 2].copy()
    datos_k4 = particiones[particiones["k"] == 4].copy()

    grupos_2 = ["Grupo 1", "Grupo 2"]
    grupos_4 = ["Grupo 1", "Grupo 2", "Grupo 3", "Grupo 4"]

    escala_x = alt.Scale(
        domain=[
            float(particiones["x"].min()),
            float(particiones["x"].max())
        ],
        nice=False
    )

    escala_y = alt.Scale(
        domain=[
            float(particiones["y"].min()),
            float(particiones["y"].max())
        ],
        nice=False
    )

    x = alt.X(
        "x:Q",
        title=titulo_eje("Géneros", 1),
        scale=escala_x,
        axis=alt.Axis(grid=False)
    )

    y = alt.Y(
        "y:Q",
        title=titulo_eje("Géneros", 2),
        scale=escala_y,
        axis=alt.Axis(grid=False)
    )

    grafico_2 = alt.Chart(datos_k2).mark_circle(
        size=22,
        opacity=0.6
    ).encode(
        x=x,
        y=y,
        color=alt.Color(
            "grupo:N",
            title="Grupo",
            scale=alt.Scale(
                domain=grupos_2,
                range=[EPOCA_RECIENTE, QUEMADO]
            ),
            legend=alt.Legend(orient="right")
        ),
        tooltip=[
            alt.Tooltip("grupo:N", title="Grupo")
        ]
    ).properties(
        width=350,
        height=370
    ).configure_view(stroke=None)

    grafico_4 = alt.Chart(datos_k4).mark_circle(
        size=22,
        opacity=0.6
    ).encode(
        x=x,
        y=y,
        color=alt.Color(
            "grupo:N",
            title="Grupo",
            scale=alt.Scale(
                domain=grupos_4,
                range=[CHARTREUSE, QUEMADO, NAVY, ELECTRICO]
            ),
            legend=alt.Legend(orient="right")
        ),
        tooltip=[
            alt.Tooltip("grupo:N", title="Grupo")
        ]
    ).properties(
        width=350,
        height=370
    ).configure_view(stroke=None)
    
    return grafico_2, grafico_4

def grafico_elegir_k(particiones, evaluacion):
    """B · Botones para elegir k: la nube de géneros se repinta y las tres medidas resaltan ese k."""
    k_sel = alt.param(name="k_sel", value=2, bind=alt.binding_radio(options=[2, 3, 4, 5], name="Cantidad de grupos: "))
    nube = alt.Chart(particiones).transform_filter("datum.k == k_sel").mark_point(
        filled=False, size=14, strokeWidth=0.9, opacity=0.6).encode(
        x=alt.X("x:Q", title=titulo_eje("Géneros", 1), axis=alt.Axis(grid=False)),
        y=alt.Y("y:Q", title=titulo_eje("Géneros", 2), axis=alt.Axis(grid=False)),
        color=alt.Color("grupo:N", title=None, legend=None,
                        scale=alt.Scale(domain=[f"Grupo {i}" for i in range(1, 6)], range=COLORES_K)),
    ).add_params(k_sel).properties(width=380, height=300)
    m = evaluacion[(evaluacion["bloque"] == "Géneros") & evaluacion["k"].between(2, 5)]
    paneles = []
    for columna, nombre in [("inercia", "Dispersión que queda (codo)"), ("silhouette", "Separación (silhouette)"),
                            ("estabilidad", "Estabilidad (remuestreo)")]:
        paneles.append(alt.Chart(m).mark_bar(size=10, cornerRadiusEnd=3).encode(
            x=alt.X(f"{columna}:Q", title=None, scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(values=[0, 0.5, 1], grid=False)),
            y=alt.Y("k:O", title="k", axis=alt.Axis(labelAngle=0)),
            color=alt.condition("datum.k == k_sel", alt.value(EPOCA_RECIENTE), alt.value(APAGADO)),
            tooltip=["k:O", alt.Tooltip(f"{columna}:Q", title=nombre, format=".3f")],
        ).properties(width=170, height=72, title=alt.Title(nombre, fontSize=11, anchor="start")))
    return alt.hconcat(nube, alt.vconcat(*paneles, spacing=6), spacing=30).properties(title=titulo(
        "Con más de dos grupos, la nube se corta sin ningún hueco que lo justifique",
        "Elegí la cantidad de grupos con los botones de abajo: la nube del espacio de géneros se repinta.")
    ).configure_view(stroke=None)


def grafico_ari(ari):
    t = ari.assign(texto=[dec(v, 3) for v in ari["ari"]])
    y = alt.Y("comparacion:N", title=None, sort=list(t["comparacion"]),
              axis=alt.Axis(labelLimit=0, minExtent=150))
    barras = alt.Chart(t).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("ari:Q", title="Coincidencia entre agrupamientos (ARI, 1 = idénticos)",
                scale=alt.Scale(domain=[0, 1])), y=y,
        color=alt.condition(alt.datum.comparacion == "Conjunto vs. épocas", alt.value(ELECTRICO), alt.value(QUEMADO)))
    texto = alt.Chart(t).mark_text(align="left", dx=6, fontWeight="bold", color=TEXTO).encode(
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
    """Mosaico: el ancho de cada columna es el tamaño del grupo de género y la altura de cada
    bloque, su reparto entre épocas. Si género y época son independientes, los cortes quedan a la
    misma altura en las dos columnas."""
    t = largo.copy()
    total = t["usuarios"].sum()
    t["total_genero"] = t.groupby("nombre_genero")["usuarios"].transform("sum")
    t["parte"] = t["usuarios"] / t["total_genero"]
    t["orden"] = t["nombre_decada"].map({g: i for i, g in enumerate(DECADAS_GRUPOS)})
    t = t.sort_values(["nombre_genero", "orden"])
    t["y1"] = t.groupby("nombre_genero")["parte"].cumsum()
    t["y0"] = t["y1"] - t["parte"]
    anchos = t.groupby("nombre_genero")["usuarios"].sum().reindex(GENEROS_GRUPOS) / total
    x1 = anchos.cumsum()
    t["x0"] = t["nombre_genero"].map(x1 - anchos)
    t["x1"] = t["nombre_genero"].map(x1)
    t["xm"] = (t["x0"] + t["x1"]) / 2
    t["ym"] = (t["y0"] + t["y1"]) / 2
    t["bloque"] = [f"{fmt(u)} usuarios · {dec(q * 100, 0)} %" for u, q in zip(t["usuarios"], t["parte"])]
    # texto dentro del bloque: blanco sobre el azul oscuro, tinta sobre el lavanda
    t["tinta"] = t["nombre_decada"].map({DECADAS_GRUPOS[0]: "#ffffff", DECADAS_GRUPOS[1]: TEXTO})
    cabeceras = t.drop_duplicates("nombre_genero").assign(
        cabecera=lambda d: [f"{g} · {dec(anchos[g] * 100, 0)} % de los usuarios" for g in d["nombre_genero"]])

    sx = alt.Scale(domain=[0, 1], nice=False)
    sy = alt.Scale(domain=[0, 1.12], nice=False)
    bloques = alt.Chart(t).mark_rect(cornerRadius=4, stroke=FONDO, strokeWidth=2).encode(
        x=alt.X("x0:Q", scale=sx, axis=None), x2="x1:Q",
        y=alt.Y("y0:Q", scale=sy, title=None,
                axis=alt.Axis(format="%", values=[0, 0.5, 1], grid=False, domain=False)),
        y2="y1:Q",
        color=alt.Color("nombre_decada:N", title=None, scale=COLOR_DECADA, sort=DECADAS_GRUPOS,
                        legend=alt.Legend(orient="top", symbolType="square", labelLimit=0)),
        tooltip=[alt.Tooltip("nombre_genero:N", title="Grupo por género"),
                 alt.Tooltip("nombre_decada:N", title="Grupo por época"),
                 alt.Tooltip("usuarios:Q", title="Usuarios", format=","),
                 alt.Tooltip("parte:Q", title="% del grupo de género", format=".1%")])
    rotulos = alt.Chart(t).mark_text(fontSize=12, fontWeight=600).encode(
        x="xm:Q", y="ym:Q", text="bloque:N", color=alt.Color("tinta:N", scale=None))
    cabecera = alt.Chart(cabeceras).mark_text(fontSize=12, color=GRIS, baseline="bottom").encode(
        x="xm:Q", y=alt.datum(1.1), text="cabecera:N")
    return (bloques + rotulos + cabecera).properties(
        height=360,
        title=titulo(
            "Los grupos de género se reparten de forma parecida entre las épocas",
            [
                "Cada columna es un grupo de género y su ancho indica cuántos usuarios tiene.",
                f"Test χ²: p = {dec(p, 2)} · V de Cramér = {dec(v, 3)}: género y época describen dimensiones distintas."
            ]
        )
    )


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


def grafico_actividad_perfiles(usuarios, rho):
    """Nube de círculos huecos (promedio x cantidad de ratings) con una zona que reparte por perfil."""
    datos = usuarios[["userId", "rating_promedio_general", "cantidad_valoraciones_total", "perfil"]]
    perfiles = [f"{g} · {d}" for g in GENEROS_GRUPOS for d in DECADAS_GRUPOS]
    colores_perfiles = [
        "#4E79A7",  # azul
        "#F28E2B",  # naranja
        "#E15759",  # rojo
        "#76B7B2",  # turquesa
        "#59A14F",  # verde
        "#EDC948",  # amarillo
        "#B07AA1",  # violeta
        "#FF9DA7",  # rosa
    ]
    escala = alt.Scale(
        domain=perfiles,
        range=colores_perfiles
    )
    zona = alt.selection_interval()
    perfil = alt.selection_point(fields=["perfil"], bind="legend")
    nube = alt.Chart(datos).mark_point(filled=False, size=28, strokeWidth=1.1, clip=True).encode(
        x=alt.X("rating_promedio_general:Q", title="Rating promedio del usuario", scale=alt.Scale(domain=[1, 5])),
        y=alt.Y("cantidad_valoraciones_total:Q", title="Ratings por usuario (escala log)",
                scale=alt.Scale(type="log")),
        color=alt.condition(zona, alt.Color("perfil:N", scale=escala, title=None,
                                             legend=alt.Legend(
                                            orient="top",
                                            direction="vertical",
                                            columns=2,
                                            labelLimit=0)),
                            alt.value(APAGADO)),
        opacity=alt.condition(perfil, alt.value(0.8), alt.value(0.05)),
        tooltip=[alt.Tooltip("userId:N", title="Usuario"), alt.Tooltip("perfil:N", title="Perfil"),
                 alt.Tooltip("rating_promedio_general:Q", format=".2f", title="Promedio"),
                 alt.Tooltip("cantidad_valoraciones_total:Q", title="Ratings", format=",")],
    ).add_params(zona, perfil).properties(height=420)
    conteo = alt.Chart(datos).transform_filter(zona).transform_aggregate(
        n="count()", groupby=["perfil"]).transform_joinaggregate(total="sum(n)").transform_calculate(
        pct="datum.n / datum.total").mark_bar(cornerRadiusEnd=3).encode(
        y=alt.Y("perfil:N", title=None, sort=perfiles, axis=alt.Axis(labelLimit=0)),
        x=alt.X("pct:Q", title="% de los usuarios marcados", scale=alt.Scale(domain=[0, 0.5]),
                axis=alt.Axis(format="%")),
        color=alt.Color("perfil:N", scale=escala, legend=None),
        tooltip=[alt.Tooltip("perfil:N", title="Perfil"), alt.Tooltip("n:Q", title="Usuarios"),
                 alt.Tooltip("pct:Q", title="%", format=".1%")]).properties(height=420, width=230)
    return alt.hconcat(nube, conteo).properties(title=titulo(
        "Los perfiles no son «los que califican mucho» contra «los que califican poco»",
        [f"La actividad casi no se relaciona con los ejes del modelo (|ρ| ≤ {dec(rho.abs().max(), 2)}).",
         "Arrastrá sobre la nube para marcar una zona: el reparto entre perfiles casi no cambia."]))


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
        x=alt.X("x:Q", title=titulo_eje(espacio, 1), scale=alt.Scale(zero=False)),
        y=alt.Y("y:Q", title=titulo_eje(espacio, 2), scale=alt.Scale(zero=False)),
        color=alt.condition(brush, alt.Color("grupo:N", scale=escala, title="Grupo (clic para resaltar)",
                                             legend=alt.Legend(orient="top", labelLimit=320)), alt.value(APAGADO)),
        opacity=alt.condition(elegido, alt.value(0.75), alt.value(0.06)),
        tooltip=[alt.Tooltip("userId:N", title="Usuario"), alt.Tooltip("perfil:N", title="Perfil"),
                 alt.Tooltip("cantidad_valoraciones_total:Q", title="Ratings", format=","),
                 alt.Tooltip("genero_afin:N", title="Género con más afinidad")],
    ).add_params(elegido, brush).properties(width=520, height=440)

    largo = alt.Chart(datos).transform_fold(cols, as_=["variable", "valor"]).transform_calculate(
        nombre="substring(datum.variable, 2)")
    orden = nombres if espacio == "Épocas" else "-x"
    barras = largo.transform_filter(brush).transform_filter(elegido).mark_bar(
        color=QUEMADO, opacity=0.85, cornerRadiusEnd=2).encode(
        y=alt.Y("nombre:N", title=None, sort=orden),
        x=alt.X("mean(valor):Q", title=eje_barras),
        tooltip=[alt.Tooltip("nombre:N", title="Variable"),
                 alt.Tooltip("mean(valor):Q", title="Seleccionados", format=fmt_barras),
                 alt.Tooltip("count():Q", title="Usuarios")])
    todos = largo.mark_tick(color=TEXTO, thickness=2, size=16).encode(
        y=alt.Y("nombre:N", sort=orden), x="mean(valor):Q")
    panel = (barras + todos).properties(width=300, height=440, title=alt.Title(
        "Perfil de la selección", subtitle="Barra: usuarios marcados · raya blanca: todos",
        anchor="start", fontSize=14, subtitleColor=GRIS_CLARO))
    return alt.hconcat(nube, panel, spacing=30), len(datos)


# ---------------------------------------------------------------------------
# 9 · Un usuario: vecinos y recomendaciones
# ---------------------------------------------------------------------------
# cuántas películas recomendar: en evaluacion_recomendador.ipynb (Parte 6), cada una de las primeras 9
# acierta al menos 1 de cada 5 veces y la 10.ª ya no; la precisión baja de a poco, sin un salto
N_RECOMENDACIONES = 9
VECINOS = 100


def distancias(ids, zg, zd, usuario):
    """Distancia del usuario a todos los demás: promedio de la distancia en géneros y en épocas.

    Cada una se divide por la raíz de su cantidad de ejes (8 y 3) para que los dos bloques pesen igual.
    """
    i = int(np.where(ids == usuario)[0][0])
    dg = np.linalg.norm(zg - zg[i], axis=1) / np.sqrt(zg.shape[1])
    dd = np.linalg.norm(zd - zd[i], axis=1) / np.sqrt(zd.shape[1])
    sim = pd.DataFrame({"userId": ids, "distancia": (dg + dd) / 2, "dist_generos": dg, "dist_epocas": dd})
    return sim[sim["userId"] != usuario].sort_values("distancia")


def recomendar(similitud, ratings, peliculas, usuario, n=N_RECOMENDACIONES, cantidad=VECINOS):
    """Votos ponderados: cada uno de los usuarios más parecidos vota por las películas que calificó
    con 4 o más, y su voto pesa más cuanto más cerca está.

    El apoyo de una película es la parte de los votos posibles que recibió: 1 si les hubiera gustado a
    todos los vecinos. Ordenar por apoyo es lo mismo que ordenar por votos.

    Antes se ordenaba por el promedio de los vecinos, aceptando películas que habían visto sólo 2 de
    ellos: así subían rarezas con dos notas altas. En evaluacion_recomendador.ipynb esa regla acertaba
    casi como el azar (precisión@10 = 0,004) y los votos, 0,266, más que recomendar las más populares
    (0,201).
    """
    vistas = set(ratings.loc[ratings["userId"] == usuario, "movieId"])
    con_titulo = set(peliculas["movieId"])
    vecinos = similitud.head(cantidad)
    peso_total = (1 / (vecinos["distancia"] + 0.001)).sum()
    r = ratings[ratings["userId"].isin(vecinos["userId"])
                & ~ratings["movieId"].isin(vistas)
                & ratings["movieId"].isin(con_titulo)].merge(vecinos[["userId", "distancia"]], on="userId")
    r["voto"] = (r["rating"] >= 4) / (r["distancia"] + 0.001)
    rec = r.groupby("movieId").agg(score=("voto", "sum"),
                                   votantes=("rating", lambda x: int((x >= 4).sum())),
                                   cantidad_vecinos=("userId", "nunique"),
                                   rating_promedio=("rating", "mean")).reset_index()
    rec["apoyo"] = rec["score"] / peso_total
    rec = rec[rec["votantes"] > 0].sort_values(["score", "votantes"], ascending=False).head(n)
    return rec.merge(peliculas, on="movieId", how="left"), cantidad


def explicar_vecinos(usuario, similitud, usuarios, ratings, peliculas, k=5):
    """Los k vecinos más cercanos y en qué se parecen al usuario: géneros y películas que comparten."""
    tabla = usuarios.set_index("userId")
    cols = [c for c in tabla.columns if c.startswith("g_")]
    titulo_de = peliculas.set_index("movieId")["titulo"]
    gustan = ratings[ratings["rating"] >= 4].groupby("userId")["movieId"].apply(set)
    mios = gustan.get(usuario, set())
    top_mio = set(tabla.loc[usuario, cols].astype(float).nlargest(4).index)
    filas = []
    for _, v in similitud.head(k).iterrows():
        u = int(v["userId"])
        top_suyo = set(tabla.loc[u, cols].astype(float).nlargest(4).index)
        comunes_g = [c[2:] for c in tabla.loc[usuario, cols].astype(float).sort_values(ascending=False).index
                     if c in top_mio & top_suyo]
        comunes_p = mios & gustan.get(u, set())
        populares = (ratings[ratings["movieId"].isin(comunes_p) & (ratings["rating"] >= 4)]
                     ["movieId"].value_counts().index[:3])
        filas.append({"Usuario": u, "Distancia": v["distancia"], "En géneros": v["dist_generos"],
                      "En épocas": v["dist_epocas"], "Perfil": tabla.loc[u, "perfil"],
                      "Géneros favoritos en común": ", ".join(comunes_g) if comunes_g else "—",
                      "Películas que les gustaron a los dos": len(comunes_p),
                      "Por ejemplo": " · ".join(titulo_de.get(m, str(m)) for m in populares)})
    return pd.DataFrame(filas)

def grafico_top(rec, cantidad, compacto=False):
    """Gráfico de las recomendaciones según el apoyo de los vecinos."""
    t = rec.assign(
        nombre=[
            f"{i}. {pel} ({str(fecha)[:4] if pd.notna(fecha) else 's/f'})"
            for i, (pel, fecha) in enumerate(
                zip(rec["titulo"], rec["fecha_estreno"]), 1
            )
        ],
        generos=[
            " · ".join(GENERO_ES.get(g, g) for g in str(x).split("|"))
            if pd.notna(x) else ""
            for x in rec["genres"]
        ],
        etiqueta=[
            f"{dec(a * 100, 0)} %" if compacto
            else f"{dec(a * 100, 0)} % · a {v} de {cantidad}"
            for a, v in zip(rec["apoyo"], rec["votantes"])
        ]
    )

    y = alt.Y(
        "nombre:N", title=None, sort=list(t["nombre"]),
        axis=alt.Axis(
            labelLimit=160 if compacto else 0,
            labelFontSize=11 if compacto else 12
        )
    )

    x = alt.X(
        "apoyo:Q",
        title="Porcentaje (%)" if compacto else
              f"Apoyo: parte de sus {cantidad} vecinos a los que les gustó (pesando más a los más parecidos)",
        scale=alt.Scale(domain=[0, 1]),
        axis=alt.Axis(format="%", values=[0, 0.25, 0.5, 0.75, 1])
    )

    barras = alt.Chart(t).mark_bar(
        size=16, cornerRadiusEnd=4, color=EPOCA_RECIENTE
    ).encode(
        x=x, y=y,
        tooltip=[
            alt.Tooltip("titulo:N", title="Película"),
            alt.Tooltip("generos:N", title="Géneros"),
            alt.Tooltip("votantes:Q", title=f"Le gustó a (de {cantidad})"),
            alt.Tooltip("apoyo:Q", title="Apoyo", format=".0%"),
            alt.Tooltip("rating_promedio:Q",
                        title="Promedio de los que la vieron", format=".2f")
        ]
    )

    texto = alt.Chart(t).mark_text(
        align="left", dx=6, fontSize=11, color=TEXTO
    ).encode(x=x, y=y, text="etiqueta:N")

    encabezado = None if compacto else titulo(
        f"Su top {len(t)}: las películas que más les gustaron a sus {cantidad} usuarios más parecidos",
        f"Recomendamos {len(t)} porque, en nuestras pruebas, cada una de las primeras {N_RECOMENDACIONES} acierta "
        "al menos 1 de cada 5 veces; de ahí para abajo, menos."
    )

   
    grafico = (barras + texto).properties(
        height=34 * len(t) + 110
    )

    if encabezado is not None:
        grafico = grafico.properties(title=encabezado)

    return grafico.configure_view(stroke=None)



def peliculas_favoritas(ratings, peliculas, usuario, n=N_RECOMENDACIONES):
    """Obtiene las películas mejor calificadas por el usuario."""

    r = ratings.loc[
        ratings["userId"] == usuario,
        ["movieId", "rating"]
    ]

    favoritas = r.merge(
        peliculas[["movieId", "titulo", "fecha_estreno", "genres"]],
        on="movieId",
        how="inner"
    )

    return (
        favoritas.dropna(subset=["titulo"])
        .sort_values(
            ["rating", "titulo", "movieId"],
            ascending=[False, True, True]
        )
        .head(n)
        .copy()
    )


def grafico_favoritas(favoritas):
    """Películas mejor calificadas, expresadas como porcentaje de 5 estrellas."""

    t = favoritas.copy()

    t["nombre"] = [
        f"{i}. {pel} ({str(fecha)[:4] if pd.notna(fecha) else 's/f'})"
        for i, (pel, fecha) in enumerate(
            zip(t["titulo"], t["fecha_estreno"]), 1
        )
    ]

    t["generos"] = [
        " · ".join(GENERO_ES.get(g, g) for g in str(x).split("|"))
        if pd.notna(x) else ""
        for x in t["genres"]
    ]

    # Normalizamos las estrellas para que el eje vaya de 0 a 100 %.
    # 5 estrellas = 100 %, 4 estrellas = 80 %, etc.
    t["porcentaje"] = t["rating"] / 5.0

    t["etiqueta"] = [
        f"{dec(v, 1)} ★" for v in t["rating"]
    ]

    y = alt.Y(
        "nombre:N",
        title=None,
        sort=list(t["nombre"]),
        axis=alt.Axis(labelLimit=160, labelFontSize=11)
    )

    x = alt.X(
        "porcentaje:Q",
        title="Porcentaje (%)",
        scale=alt.Scale(domain=[0, 1]),
        axis=alt.Axis(
            format="%",
            values=[0, 0.25, 0.5, 0.75, 1]
        )
    )

    barras = alt.Chart(t).mark_bar(
        size=16,
        cornerRadiusEnd=4,
        color=QUEMADO
    ).encode(
        x=x,
        y=y,
        tooltip=[
            alt.Tooltip("titulo:N", title="Película"),
            alt.Tooltip("generos:N", title="Géneros"),
            alt.Tooltip(
                "rating:Q",
                title="Calificación (estrellas)",
                format=".1f"
            ),
            alt.Tooltip(
                "porcentaje:Q",
                title="Porcentaje del máximo",
                format=".0%"
            )
        ]
    )

    texto = alt.Chart(t).mark_text(
        align="right",
        dx=-6,
        fontSize=11,
        color=FONDO
    ).encode(
        x=x,
        y=y,
        text="etiqueta:N"
    )

    return (barras + texto).properties(
        height=34 * len(t) + 110
    ).configure_view(stroke=None)



def grafico_usuario_en_espacio(usuarios, usuario, vecinos, espacio):
    px, py = ("G_PC1", "G_PC2") if espacio == "Géneros" else ("D_PC1", "D_PC2")
    d = usuarios[["userId", px, py]].rename(columns={px: "x", py: "y"}).copy()
    d["tipo"] = "Resto"
    d.loc[d["userId"].isin(vecinos), "tipo"] = "Los 5 más parecidos"
    d.loc[d["userId"] == usuario, "tipo"] = "Usuario elegido"
    d["orden"] = d["tipo"].map({"Resto": 0, "Los 5 más parecidos": 1, "Usuario elegido": 2})
    escala = alt.Scale(domain=["Resto", "Los 5 más parecidos", "Usuario elegido"],
                       range=[APAGADO, CHARTREUSE, QUEMADO])
    return alt.Chart(d).mark_circle().encode(
        x=alt.X("x:Q", title=titulo_eje(espacio, 1), scale=alt.Scale(zero=False)),
        y=alt.Y("y:Q", title=titulo_eje(espacio, 2), scale=alt.Scale(zero=False)),
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
        color=alt.condition(alt.datum.afinidad > 0, alt.value(QUEMADO), alt.value(GRIS)),
        tooltip=["genero:N", alt.Tooltip("afinidad:Q", format="+.3f")],
    ).properties(height=380, title=alt.Title("Sus afinidades por género", anchor="start", fontSize=14))


def grafico_epocas_usuario(fila):
    cols = [c for c in fila.index if c.startswith("e_")]
    t = pd.DataFrame({"epoca": [c[2:] for c in cols], "pct": fila[cols].astype(float).to_numpy()})
    return alt.Chart(t).mark_bar(color=ELECTRICO, cornerRadiusEnd=2).encode(
        x=alt.X("epoca:N", sort=ORDEN_EPOCAS, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("pct:Q", title="% de su historial"),
        tooltip=["epoca:N", alt.Tooltip("pct:Q", format=".1f")],
    ).properties(height=380, title=alt.Title("Su historial según época de estreno", anchor="start", fontSize=14))
