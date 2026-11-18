"""Perfiles de cine · Grupo 5K09-05.

Una sola página que se recorre de arriba hacia abajo: primero la historia del
proyecto (el tallo de la copa de martini) y al final la parte para explorar
(la copa). El índice del costado salta a cada sección.
"""
import pandas as pd
import streamlit as st

import historia as h

st.set_page_config(page_title="Perfiles de cine", page_icon=":material/movie:", layout="wide")


# ---------------------------------------------------------------------------
# Datos y cálculos (se hacen una vez y quedan en caché)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Entrenando los modelos…")
def cargar():
    return h.cargar_todo()


silver, modelos, ratings, peliculas = cargar()


@st.cache_data
def usuarios_tabla():
    return h.tabla_usuarios(modelos, silver)


@st.cache_resource
def espacios():
    return h.espacios(modelos)


@st.cache_data(show_spinner="Probando de 1 a 6 grupos…")
def evaluacion_k():
    return h.evaluar_k(modelos)


@st.cache_data
def tablas_historia():
    return {
        "etapas": h.tabla_etapas(silver, ratings, peliculas),
        "trampa": h.tabla_trampa(silver),
        "umbral": h.tabla_umbral(silver),
        "variables": h.tabla_variables(silver, modelos),
        "varianza": h.tabla_varianza(modelos),
        "cargas": h.tabla_cargas(modelos),
    }


usuarios = usuarios_tabla()
T = tablas_historia()
act = h.resumen_actividad(silver)


def decision(texto):
    """Caja con la decisión que tomamos en esa etapa."""
    with st.container(border=True):
        st.markdown(f":red[:material/check_circle: **Decisión**]  \n{texto}")


def seccion(numero, nombre, ancla, afirmacion):
    st.space("large")
    st.markdown(f":red[**{numero} · {nombre.upper()}**]")
    st.header(afirmacion, anchor=ancla)


def grafico(chart, descripcion):
    st.altair_chart(chart, width="stretch", alt=descripcion)


# ---------------------------------------------------------------------------
# Índice lateral
# ---------------------------------------------------------------------------
INDICE = [
    ("La historia", [("inicio", "El proyecto"), ("datos", "1 · Los datos"),
                     ("pipeline", "2 · Lo que se pierde"), ("trampa", "3 · La trampa de los pocos ratings"),
                     ("variables", "4 · Qué entra al modelo"), ("pca", "5 · Menos dimensiones"),
                     ("grupos", "6 · Cuántos grupos"), ("perfiles", "7 · Los perfiles")]),
    ("Tu turno", [("explora", "8 · Explorá los usuarios"), ("recomendador", "9 · Un usuario y sus películas")]),
]

with st.sidebar:
    st.subheader("Perfiles de cine", icon=":material/movie:")
    st.caption("Ciencia de Datos · UTN FRM · 2026 · Grupo 5K09-05")
    for grupo, items in INDICE:
        st.markdown(f"**{grupo}**")
        st.markdown("\n".join(f"- [{texto}](#{ancla})" for ancla, texto in items))
    st.space("small")
    st.caption("MovieLens 32M · 5.000 usuarios al azar (random_state 42) · TMDB")


# ---------------------------------------------------------------------------
# El proyecto
# ---------------------------------------------------------------------------
st.markdown(":red[**PROYECTO INTEGRADOR · CIENCIA DE DATOS**]")
st.title("¿Se pueden armar perfiles de espectadores con lo que califican, y usarlos para recomendar?",
         anchor="inicio")
st.markdown(
    "Tomamos usuarios de MovieLens, cruzamos sus películas con TMDB y buscamos grupos de gente "
    "que califica parecido. Esta página cuenta, paso a paso, qué decisiones tomamos y por qué "
    "los datos nos llevaron a ellas. Al final, el lector explora solo."
)

ev, ari, sil_conjunto = evaluacion_k()
with st.container(horizontal=True):
    st.metric("Usuarios en la muestra", h.fmt(act["usuarios"]), border=True)
    st.metric("Usuarios modelados", h.fmt(act["modelados"]), border=True,
              help="Los que tienen al menos 150 ratings.")
    st.metric("Grupos por género y por época", "2 + 2", border=True)
    st.metric("Perfiles combinados", "4", border=True)


# ---------------------------------------------------------------------------
# 1 · Los datos
# ---------------------------------------------------------------------------
seccion(1, "Los datos", "datos", "La actividad de los usuarios abarca dos órdenes de magnitud")
izq, der = st.columns([3, 2], gap="large")
with izq:
    st.markdown(
        f"El pipeline de Airflow elige **{h.fmt(act['usuarios'])} usuarios al azar** de MovieLens 32M entre "
        f"los que calificaron al menos {act['minimo']} películas, y trae todos sus ratings. "
        f"Hay de todo: la mediana es **{h.fmt(act['mediana'])}**, pero el más activo llega a "
        f"**{h.fmt(act['maximo'])}**. Si usáramos los ratings tal cual, unos pocos usuarios pesarían "
        "decenas de veces más que el resto."
    )
with der:
    decision(f"Recortar a **500 ratings** por usuario, al azar ({h.fmt(act['con_tope'])} usuarios superan "
             "el tope), y modelar sólo a quienes tienen **150 o más**.")
grafico(h.grafico_histograma(silver), "Histograma de ratings por usuario en escala logarítmica")
with st.expander("Ver los números", icon=":material/table_chart:"):
    st.dataframe(h.tabla_histograma(silver), hide_index=True, alt="Usuarios por intervalo de ratings")


# ---------------------------------------------------------------------------
# 2 · Lo que se pierde
# ---------------------------------------------------------------------------
etapas = T["etapas"]
seccion(2, "Lo que se pierde", "pipeline", "Cada etapa del pipeline recorta ratings, no usuarios")
izq, der = st.columns([3, 2], gap="large")
with izq:
    st.markdown(
        "El DAG tiene dos fuentes de pérdida. El **tope de 500** submuestrea a los usuarios más activos. "
        "El **cruce con TMDB** sólo conoce el género de las 10.000 películas más vistas de la muestra: los "
        "ratings de las demás no entran al cálculo por género. Las estadísticas generales de cada usuario "
        "se calculan antes del cruce, así que nadie desaparece por culpa de TMDB."
    )
with der:
    decision("Consultar **10.000 películas** en TMDB (en la Entrega 1 eran 1.000) y calcular las "
             "estadísticas del usuario **antes** del cruce.")
grafico(h.grafico_etapas(etapas), "Ratings que quedan después de cada etapa del pipeline")
with st.expander("Ver los números", icon=":material/table_chart:"):
    st.dataframe(etapas.drop(columns="orden"), hide_index=True, alt="Ratings y usuarios por etapa",
                 column_config={"pct_perdido": st.column_config.NumberColumn("% perdido", format="percent")})


# ---------------------------------------------------------------------------
# 3 · La trampa de los pocos ratings
# ---------------------------------------------------------------------------
seccion(3, "La trampa de los pocos ratings", "trampa",
        "Un género visto una sola vez no dice nada del gusto")
izq, der = st.columns([3, 2], gap="large")
with izq:
    st.markdown(
        "Para saber si a alguien le gusta un género, comparamos su promedio en ese género con su propio "
        "promedio general. Pero si vio **una** película del género, ese número es la nota de esa película "
        "y nada más: por eso en la Entrega 2 el «género mejor valorado» salía casi siempre uno raro, como "
        "Historia o Bélico. El gráfico muestra cuánto se dispersa la diferencia según cuántas películas "
        "del género calificó el usuario."
    )
with der:
    decision(f"Usar una **media amortiguada**: la diferencia se multiplica por n/(n + {h.SUAVIZADO}). "
             "Con pocas películas se acerca a cero; con muchas, casi no cambia.")
grafico(h.grafico_trampa(T["trampa"]), "Dispersión de la afinidad por género según cantidad de películas")
with st.expander("Ver los números", icon=":material/table_chart:"):
    st.dataframe(T["trampa"].round(3), hide_index=True, alt="Percentiles de la afinidad por tramo")

st.markdown(
    "La misma lógica vale para el usuario entero: con pocos ratings, todo su perfil es ruido. "
    "¿Dónde poner el corte?"
)
izq, der = st.columns([3, 2], gap="large")
with izq:
    grafico(h.grafico_umbral(T["umbral"]), "Usuarios que quedan según el mínimo de ratings")
with der:
    st.space("large")
    decision(f"Exigir **{h.MIN_RATINGS} ratings**: conserva {h.fmt(act['modelados'])} usuarios y deja "
             "afuera a los historiales más cortos. No decimos que 150 sea óptimo: es un equilibrio entre "
             "perfiles confiables y usuarios perdidos.")
    st.caption("En la Entrega 3 se probó la sensibilidad: con 100 la separación de géneros baja y con "
               "250 o 300 se pierde más de la mitad de la muestra.")


# ---------------------------------------------------------------------------
# 4 · Qué entra al modelo
# ---------------------------------------------------------------------------
seccion(4, "Qué entra al modelo", "variables", "Entran gustos por género y épocas del historial; nada más")
st.markdown(
    "El silver tiene 54 columnas por usuario. Al modelo entran sólo las que describen **qué le gusta** "
    "(afinidad por género) y **qué mira** (épocas del historial). Todo lo que mide actividad, fechas o "
    "identidad queda afuera, aunque lo usamos para filtrar y para controlar el resultado."
)
st.table(T["variables"].set_index("Variable"))


# ---------------------------------------------------------------------------
# 5 · Menos dimensiones
# ---------------------------------------------------------------------------
seccion(5, "Menos dimensiones", "pca", "Dos PCA separadas: una para géneros, otra para épocas")
izq, der = st.columns([3, 2], gap="large")
with izq:
    st.markdown(
        "Muchos géneros se mueven juntos (quien valora Acción suele valorar Aventura). Antes de agrupar, "
        "resumimos cada bloque con PCA, por separado, para que los 17 géneros no tapen a las 5 épocas. "
        "K-means usa **todas** las componentes conservadas; los gráficos muestran sólo las dos primeras."
    )
with der:
    decision("Conservar al menos el **80 % de la varianza** en cada bloque.")
izq, der = st.columns(2, gap="large")
with izq:
    grafico(h.grafico_varianza(T["varianza"]), "Varianza acumulada de las dos PCA")
with der:
    grafico(h.grafico_cargas(T["cargas"]), "Cargas de la primera componente de géneros")
with st.expander("Ver los números", icon=":material/table_chart:"):
    c1, c2 = st.columns(2)
    c1.dataframe(T["varianza"], hide_index=True, alt="Varianza acumulada por componente",
                 column_config={"acumulada": st.column_config.NumberColumn(format="percent")})
    c2.dataframe(T["cargas"].round(3), hide_index=True, alt="Cargas de la primera componente")


# ---------------------------------------------------------------------------
# 6 · Cuántos grupos
# ---------------------------------------------------------------------------
codo, silueta = h.grafico_codo(ev)
seccion(6, "Cuántos grupos", "grupos", "Dos grupos por bloque, y cada bloque por su lado")
izq, der = st.columns([3, 2], gap="large")
with izq:
    st.markdown(
        "Probamos K-means de 1 a 6 grupos en cada espacio. El codo muestra cuánto se gana con cada grupo "
        "más; la silhouette, qué tan separados quedan. Después probamos un único modelo con los dos "
        "bloques juntos, para ver si convenía no separarlos."
    )
with der:
    decision("**k = 2** en géneros y **k = 2** en épocas. Cada usuario queda con dos etiquetas, que "
             "cruzadas dan cuatro perfiles.")
c1, c2 = st.columns(2, gap="large")
with c1:
    grafico(codo, "Curva del codo para géneros y épocas")
with c2:
    grafico(silueta, "Silhouette según la cantidad de grupos")
grafico(h.grafico_ari(ari), "Coincidencia del modelo conjunto con los modelos separados")
sil_generos = ev[(ev["bloque"] == "Géneros") & (ev["k"] == 2)]["silhouette"].iloc[0]
st.caption(f"La separación en géneros es modesta (silhouette {h.dec(sil_generos, 3)}): los grupos son "
           f"tendencias, no categorías cerradas. El modelo conjunto tiene silhouette {h.dec(sil_conjunto, 3)}.")
with st.expander("Ver los números", icon=":material/table_chart:"):
    c1, c2 = st.columns(2)
    c1.dataframe(ev.round(3), hide_index=True, alt="Inercia y silhouette por k")
    c2.dataframe(ari.round(4), hide_index=True, alt="ARI del modelo conjunto")


# ---------------------------------------------------------------------------
# 7 · Los perfiles
# ---------------------------------------------------------------------------
cruce, p_chi, v_cramer = h.tabla_cruce(usuarios)
perfil_g, orden_g = h.tabla_perfil_generos(usuarios)
perfil_e = h.tabla_perfil_epocas(usuarios)
rho, medianas = h.pregunta_actividad(usuarios, modelos)

seccion(7, "Los perfiles", "perfiles", "Cuatro perfiles, armados con dos preguntas distintas")
st.markdown(
    "Un modelo responde **qué géneros le gustan más que su propio promedio**; el otro, **de qué época es "
    "lo que mira**. Las dos respuestas no dependen una de la otra, así que cada combinación es un perfil."
)
grafico(h.grafico_cruce(cruce, p_chi, v_cramer), "Cruce de los grupos de género y de época")
c1, c2 = st.columns(2, gap="large")
with c1:
    grafico(h.grafico_perfil_generos(perfil_g, orden_g), "Afinidad media por género en cada grupo")
with c2:
    grafico(h.grafico_perfil_epocas(perfil_e), "Porcentaje del historial por época en cada grupo")

st.subheader("¿Gustos distintos, o gente que califica más y gente que califica menos?")
izq, der = st.columns([3, 2], gap="large")
with izq:
    st.markdown(
        "Es la trampa que más nos preocupaba: si los grupos sólo separaran a los usuarios muy activos "
        "de los poco activos, no estaríamos encontrando gustos sino volumen. La cantidad de ratings no "
        "entra al modelo, así que la usamos para controlarlo."
    )
with der:
    with st.container(border=True):
        st.markdown(
            ":red[:material/search: **Lo que encontramos**]  \n"
            f"La actividad casi no se relaciona con los ejes del modelo (|ρ| ≤ {h.dec(rho.abs().max(), 2)}). "
            f"Hay un matiz: el grupo de drama y crimen califica algo más "
            f"(mediana {h.fmt(medianas[h.GENEROS_GRUPOS[1]])} contra "
            f"{h.fmt(medianas[h.GENEROS_GRUPOS[0]])}), pero las cajas se superponen casi por completo."
        )
grafico(h.grafico_actividad(usuarios, rho, medianas), "Ratings por usuario en cada grupo")
with st.expander("Ver los números", icon=":material/table_chart:"):
    c1, c2, c3 = st.columns(3)
    c1.dataframe(cruce, hide_index=True, alt="Usuarios por perfil combinado")
    c2.dataframe(rho.rename("ρ de Spearman").round(3), alt="Correlación de la actividad con cada eje")
    c3.dataframe(medianas.rename("Mediana de ratings"), alt="Mediana de ratings por grupo")


# ---------------------------------------------------------------------------
# 8 · Explorá (la copa)
# ---------------------------------------------------------------------------
seccion(8, "Tu turno", "explora", "Explorá los usuarios")
st.markdown(
    "Cada punto es un usuario. **Arrastrá** sobre la nube para marcar un grupo de usuarios y ver su perfil "
    "a la derecha; **hacé clic en la leyenda** para resaltar un grupo; pasá el mouse por un punto para ver "
    "quién es. La rueda del mouse no hace zoom: el arrastre está reservado para marcar."
)


@st.fragment
def copa():
    c1, c2 = st.columns([1, 2], vertical_alignment="bottom")
    with c1:
        espacio = st.segmented_control("Espacio", ["Géneros", "Épocas"], default="Géneros",
                                       required=True, key="espacio_copa")
    with c2:
        minimo = st.slider("Mínimo de ratings del usuario (historial completo)", 150, 2000, 150, step=50,
                           key="minimo_copa")
    chart, n = h.grafico_copa(usuarios, espacio, minimo)
    st.caption(f"Mostrando {h.fmt(n)} de {h.fmt(len(usuarios))} usuarios.")
    st.altair_chart(chart, alt="Nube de usuarios con selección y perfil de la selección")


copa()


# ---------------------------------------------------------------------------
# 9 · Un usuario y sus recomendaciones
# ---------------------------------------------------------------------------
seccion(9, "Tu turno", "recomendador", "Un usuario, sus vecinos y sus películas")
st.markdown(
    "Elegí un usuario. Buscamos a los más parecidos en los dos espacios (con el mismo peso) y le "
    "recomendamos películas que esos vecinos calificaron bien y que el usuario todavía no calificó."
)


@st.fragment
def un_usuario():
    ids, zg, zd = espacios()
    perfiles = ["Todos"] + sorted(usuarios["perfil"].unique())
    c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
    with c1:
        perfil = st.selectbox("Perfil", perfiles, key="perfil_usuario")
    candidatos = usuarios if perfil == "Todos" else usuarios[usuarios["perfil"] == perfil]
    with c2:
        usuario = st.selectbox("Usuario", candidatos["userId"].tolist(), key="usuario",
                               format_func=lambda u: f"Usuario {u}")
    fila = usuarios.set_index("userId").loc[usuario]
    similitud = h.distancias(ids, zg, zd, usuario)
    vecinos = similitud.head(5)["userId"].tolist()

    with st.container(horizontal=True, vertical_alignment="center"):
        st.metric("Ratings en su historial", h.fmt(fila["cantidad_valoraciones_total"]), border=True)
        st.metric("Rating promedio", f"{h.dec(fila['rating_promedio_general'], 2)} / 5", border=True)
        with st.container(border=True):
            st.caption("Su perfil")
            st.badge(fila["nombre_genero"], icon=":material/theater_comedy:", color="orange")
            st.badge(fila["nombre_decada"], icon=":material/calendar_month:", color="blue")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        grafico(h.grafico_usuario_en_espacio(usuarios, usuario, vecinos, "Géneros"),
                "Posición del usuario en el espacio de géneros")
    with c2:
        grafico(h.grafico_usuario_en_espacio(usuarios, usuario, vecinos, "Épocas"),
                "Posición del usuario en el espacio de épocas")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        grafico(h.grafico_afinidades_usuario(fila), "Afinidades por género del usuario")
    with c2:
        grafico(h.grafico_epocas_usuario(fila), "Historial del usuario por época")

    rec, cantidad = h.recomendar(similitud, ratings, peliculas, usuario)
    st.subheader("Películas para recomendarle", icon=":material/theaters:")
    st.caption(f"Basadas en los {cantidad} usuarios más parecidos. Cada película tiene que estar calificada "
               "por al menos 2 de ellos, con promedio de 3,8 o más; los más parecidos pesan más.")
    if rec.empty:
        st.warning("No encontramos películas con evidencia suficiente para este usuario.")
    else:
        for fila_cartas in [rec.iloc[:3], rec.iloc[3:6]]:
            cols = st.columns(3)
            for col, (_, p) in zip(cols, fila_cartas.iterrows()):
                with col.container(border=True, height="stretch"):
                    anio = str(p["fecha_estreno"])[:4] if pd.notna(p["fecha_estreno"]) else "s/f"
                    st.markdown(f"**{p['titulo']}** · {anio}")
                    generos = str(p["genres"]).split("|") if pd.notna(p["genres"]) else []
                    st.caption(" · ".join(h.GENERO_ES.get(g, g) for g in generos))
                    st.markdown(f":material/star: {h.dec(p['rating_promedio'], 2)} entre "
                                f"{int(p['cantidad_vecinos'])} usuarios parecidos")

    with st.expander("Los 5 usuarios más parecidos", icon=":material/group:"):
        tabla = similitud.head(5).merge(usuarios[["userId", "perfil", "cantidad_valoraciones_total"]], on="userId")
        st.dataframe(tabla, hide_index=True, alt="Usuarios más parecidos",
                     column_config={"userId": "Usuario", "perfil": "Perfil",
                                    "cantidad_valoraciones_total": "Ratings",
                                    "distancia": st.column_config.NumberColumn("Distancia", format="%.3f")})


un_usuario()

st.space("large")
st.caption(
    f"Fuente: MovieLens 32M · {h.fmt(act['usuarios'])} usuarios al azar con ≥ 100 ratings (random_state 42) · "
    f"tope de 500 ratings · TMDB para géneros y fechas · modelo con los {h.fmt(act['modelados'])} usuarios "
    "de 150 ratings o más. Los grupos son descriptivos: no hay una etiqueta verdadera de «tipo de espectador»."
)
