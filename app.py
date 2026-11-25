"""Perfiles de cine · Grupo 5K09-05.

Una sola página que se recorre de arriba hacia abajo: primero la historia del
proyecto (el tallo de la copa de martini) y al final la parte para explorar
(la copa). El índice del costado salta a cada sección.
"""
import hashlib
import importlib
from pathlib import Path

import pandas as pd
import streamlit as st

import historia as h
import modelado

# Streamlit Cloud, al actualizar la app, vuelve a correr este archivo pero puede seguir usando las
# versiones viejas de modelado.py e historia.py que tenía en memoria (pasó al cambiar tabla_umbral y
# al renombrar los grupos). Se recargan en cada corrida, primero modelado.py porque historia.py
# importa de ahí; cuesta milisegundos y asegura que los tres archivos sean de la misma versión.
modelado = importlib.reload(modelado)
h = importlib.reload(h)

# La caché tampoco se entera de esos cambios: guardaría el modelo entrenado con el código anterior
# (por ejemplo, con los nombres viejos de los grupos). Por eso cada función cacheada recibe la
# versión del código, y si modelado.py o historia.py cambian, se vuelve a calcular.
VERSION = hashlib.sha1(b"".join((Path(__file__).parent / f).read_bytes() for f in ("modelado.py", "historia.py"))).hexdigest()

st.set_page_config(page_title="Perfiles de cine", page_icon=":material/movie:", layout="wide")


# ---------------------------------------------------------------------------
# Datos y cálculos (se hacen una vez y quedan en caché)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Entrenando los modelos…")
def cargar(version):
    return h.cargar_todo()


silver, modelos, ratings, peliculas = cargar(VERSION)


@st.cache_data
def cargar_origen():
    return h.cargar_origen()


origen = cargar_origen()


@st.cache_data
def usuarios_tabla(version):
    return h.tabla_usuarios(modelos, silver)


@st.cache_resource
def espacios(version):
    return h.espacios(modelos)


@st.cache_data(show_spinner="Probando de 1 a 6 grupos y midiendo su estabilidad…")
def evaluacion_k(version):
    return h.evaluar_k(modelos)


@st.cache_data
def particiones_k(version):
    return h.particiones_generos(modelos), h.silhouette_epocas_por_componentes(modelos)


@st.cache_data
def tablas_historia(version):
    return {
        "poblacion": h.tabla_poblacion(origen["poblacion"], silver),
        "trampa": h.tabla_trampa(silver),
        "umbral": h.tabla_umbral(silver, modelos),
        "variables": h.tabla_variables(silver, modelos),
        "varianza": h.tabla_varianza(modelos),
        "cargas_todas": h.tabla_cargas_todas(modelos),
    }


usuarios = usuarios_tabla(VERSION)
T = tablas_historia(VERSION)
h.fijar_ejes(T["cargas_todas"])   # cada eje de las PCA con nombre, para todos los gráficos
EJEMPLO = int(usuarios["userId"].iloc[0])   # el mismo usuario que aparece primero en la sección 8
act = h.resumen_actividad(silver)


def decision(texto):
    """Caja con la decisión que tomamos en esa etapa."""
    with st.container(border=True):
        st.markdown(f":violet[:material/check_circle: **Decisión**]  \n{texto}")


# color de la etiqueta de cada sección: el de su tramo en el recorrido
# (red = datos, orange = modelo, blue = resultado; ver .streamlit/config.toml)
COLOR_TRAMO = {1: "red", 2: "orange", 3: "orange", 4: "orange", 5: "orange", 6: "blue", 7: "blue", 8: "blue"}


def seccion(numero, nombre, ancla, afirmacion):
    st.space("large")
    st.markdown(f":{COLOR_TRAMO[numero]}[**{numero} · {nombre.upper()}**]")
    st.header(afirmacion, anchor=ancla)


def grafico(chart, descripcion):
    st.altair_chart(chart, width="stretch", alt=descripcion)


def como_leer(texto):
    """Guía corta debajo de un gráfico: qué es cada marca, qué miden los ejes y qué mirar."""
    st.caption(f":material/menu_book: **Cómo leer este gráfico.** {texto}")


# ---------------------------------------------------------------------------
# Índice lateral
# ---------------------------------------------------------------------------
# (número, título, ancla, subpasos); el color de la insignia es el del tramo (COLOR_TRAMO)
INDICE = [
    (1, "Los datos", "datos", [("Una tabla de votos", "paso-crudo"), ("A quién miramos", "paso-muestra"),
                                ("TMDB", "paso-tmdb"), ("La tabla silver", "paso-silver")]),
    (2, "La trampa de los pocos ratings", "trampa", []),
    (3, "Qué entra al modelo", "variables", []),
    (4, "Menos dimensiones", "pca", []),
    (5, "Cuántos grupos", "grupos", []),
    (6, "Los perfiles", "perfiles", []),
    (7, "Explorá los usuarios", "explora", []),
    (8, "Un usuario y sus películas", "recomendador", []),
]

with st.sidebar:
    st.markdown("**Perfiles de cine**")
    st.caption("Proyecto integrador · Ciencia de Datos · UTN FRM 2026 · Grupo 5K09-05")
    st.markdown(":gray-badge[00] [El problema y el recorrido](#inicio)")
    for numero, titulo, ancla, pasos in INDICE:
        st.markdown(f":{COLOR_TRAMO[numero]}-badge[{numero:02d}] [{titulo}](#{ancla})")
        if pasos:
            st.caption("  \n".join(f":material/subdirectory_arrow_right: [{p}](#{a})" for p, a in pasos))
    st.space("small")
    st.caption("MovieLens 32M · 5.000 usuarios al azar (random_state 42) · TMDB")


# ---------------------------------------------------------------------------
# El proyecto
# ---------------------------------------------------------------------------
st.markdown(":blue[**PROYECTO INTEGRADOR · CIENCIA DE DATOS**]")
st.title("¿Se pueden armar perfiles de espectadores con lo que califican, y usarlos para recomendar?",
         anchor="inicio")
st.markdown("**Sí, es posible.** A partir de las calificaciones de películas, analizamos "
    "patrones de valoración y aplicamos técnicas de aprendizaje automático que nos "
    "permitieron agrupar usuarios con gustos similares. "
    "Estos perfiles son la base de nuestro sistema de recomendación, que busca sugerir "
    "películas a partir de las preferencias de otros usuarios con gustos parecidos."
)
st.markdown(
    "**El problema.** Queremos saber si se pueden armar perfiles de espectadores a partir de cómo "
    "califican películas, y usarlos para recomendar. Pero al principio sólo tenemos una tabla de votos: "
    "quién calificó qué película y con cuántas estrellas, sin géneros, sin fechas de estreno y sin "
    "ninguna descripción del usuario. Esta página cuenta el recorrido desde esa tabla hasta las "
    "recomendaciones, y la decisión que tomamos en cada paso."
)

ev, ari, sil_conjunto = evaluacion_k(VERSION)
R = origen["resumen"]
estaciones = h.tabla_recorrido(R)
st.markdown("**El recorrido.** Partimos de los votos crudos y llegamos a recomendar películas:")
grafico(h.grafico_recorrido(estaciones), "Recorrido del proyecto en nueve pasos, de los votos a la recomendación")
como_leer('Cada punto es un paso del proyecto, de izquierda a derecha. El color marca el tramo: datos (burnt orange), modelo (chartreuse) y resultado (azul). Debajo de cada paso está la decisión que tomamos; pasá el mouse por un punto para verla completa.')


# ---------------------------------------------------------------------------
# 1 · De dónde salen los datos: del ratings.csv crudo a la tabla silver
# ---------------------------------------------------------------------------
seccion(1, "Los datos", "datos", "De una tabla de votos a una tabla de usuarios")
st.markdown(
    "Para armar perfiles de gusto necesitamos saber, de cada usuario, qué películas califica y cómo, y "
    "de cada película, de qué género y de qué época es. Ninguna fuente trae todo eso junto: lo fuimos "
    "juntando con un DAG de Airflow, en cuatro pasos, quedándonos en cada uno con lo que aporta "
    "información para el objetivo."
)

st.subheader("Paso 1 · Los datos originales: calificaciones de MovieLens", anchor="paso-crudo")
izq, der = st.columns([3, 2], gap="large")
with izq:
    st.markdown(
        "Partimos de **MovieLens**, una colección pública de datos para investigación que reúne "
        f"calificaciones de usuarios reales desde {R['anio_primer_voto']} hasta {R['anio_ultimo_voto']}. "
        "Su archivo `ratings.csv` contiene los registros originales de calificaciones: cada fila "
        "representa la valoración de un usuario sobre una película, junto con la fecha de dicha "
        "valoración expresada como marca de tiempo (*timestamp*) en segundos. "
        "Este archivo identifica las películas mediante un código, pero no incluye sus nombres, "
        "géneros ni años de estreno. Para obtener esos atributos, necesitamos incorporar "
        "información adicional. Así comienza el archivo:"
    )
    st.table(origen["crudo"].astype({"rating": str}).set_index("userId"))
with der:
    with st.container(horizontal=True):
        st.metric("Votos", h.fmt(R["ratings_total"]), border=True)
        st.metric("Usuarios", h.fmt(R["usuarios_total"]), border=True)
        st.metric("Películas con votos", h.fmt(R["peliculas_total"]), border=True,
                  help="Películas distintas en ratings.csv. El catálogo (movies.csv) tiene más, "
                       "pero algunas no tienen ningún voto.")

st.subheader("Paso 2 · A quién miramos", anchor="paso-muestra")
izq, der = st.columns([3, 2], gap="large")
with izq:
    st.markdown(
        "Para describir el gusto de alguien hacen falta muchos votos suyos, así que agrupamos los ratings "
        "por usuario y pedimos un mínimo. Entre los que lo cumplen, elegimos usuarios al azar."
    )
with der:
    decision(f"Usuarios con **100 ratings o más**, **{h.fmt(R['usuarios_muestra'])} al azar** "
             f"(random_state 42) y un **tope de 500** ratings por usuario. El modelo, después, pide "
             f"**{h.MIN_RATINGS}**.")
grafico(h.grafico_muestra_vs_poblacion(origen["poblacion"], silver),
        "Histogramas superpuestos de ratings por usuario: MovieLens y la muestra")
como_leer('Cada barra es un rango de cantidad de ratings por usuario. El eje x está en escala logarítmica: de una marca a la siguiente, la cantidad se multiplica. El eje y es qué parte de cada grupo cae en ese rango. En azul oscuro, los usuarios de MovieLens con 100 ratings o más; en burnt orange, nuestra muestra; donde se superponen, el color se mezcla. Si las dos siluetas coinciden, la muestra se parece a MovieLens.')
with st.expander("Ver los números", icon=":material/table_chart:"):
    st.dataframe(T["poblacion"], hide_index=True, alt="Usuarios por intervalo de ratings")

st.subheader("Paso 3 · Qué son esas películas: TMDB", anchor="paso-tmdb")
izq, der = st.columns([3, 2], gap="large")
with izq:
    st.markdown(
        "Un `movieId` no dice nada. Para saber qué tipo de películas califica cada usuario buscamos en "
        "TMDB las características de las películas que aparecen en sus ratings: título, fecha de estreno "
        "y géneros. El puente es `links.csv`, que traduce el `movieId` de MovieLens al id de TMDB."
    )
with der:
    decision(f"Consultar las **{h.fmt(R['peliculas_consultadas'])} películas más vistas** de la muestra, "
             "no todas.")
c1, c2 = st.columns([3, 2], gap="large")
with c1:
    grafico(h.grafico_cobertura(origen["cobertura"], R), "Ratings cubiertos según cuántas películas se consultan")
    como_leer('El eje x ordena las películas de la más vista a la menos vista (escala logarítmica). El eje y es qué parte de todos los ratings de la muestra suman las películas hasta ese punto. La zona en burnt orange son las películas que consultamos a TMDB, y el punto marca cuánto cubren.')
with c2:
    st.space("medium")
    st.caption("Las mismas filas del paso 1, después del cruce con TMDB:")
    st.table(h.tabla_cruda_con_tmdb(origen["crudo"], peliculas)[["movieId", "título", "estreno", "géneros"]]
             .set_index("movieId"))

st.subheader("Paso 4 · De votos a usuarios: la tabla silver", anchor="paso-silver")
izq, der = st.columns([3, 2], gap="large")
with izq:
    st.markdown(
        f"Con los votos y las películas, el DAG resume a cada usuario en una fila: cuánto califica, su "
        f"promedio, su promedio en cada género y qué parte de lo que mira es de cada época. El resultado es "
        f"`usuarios_generos.csv`: **{h.fmt(len(silver))} filas y {silver.shape[1]} columnas**."
    )
with der:
    decision("Calcular las estadísticas generales del usuario **antes** del cruce con TMDB, para que nadie "
             "pierda votos por una película que TMDB no tiene.")
st.markdown(
    "**Qué tiene cada fila.** Una fila es un usuario, y sus columnas se agrupan en cuatro tipos:\n"
    "- **Actividad:** cuántas películas calificó (con y sin el tope de 500), su promedio general, su desvío "
    "y la fecha de su primer y último voto.\n"
    "- **Por género (19 × 2 columnas):** para cada género de TMDB, su promedio en ese género (`Drama`, "
    "`Comedy`…) y cuántas películas de ese género calificó (`n_Drama`, `n_Comedy`…).\n"
    "- **Por época (6 columnas):** qué parte de su historial es de antes de 1980, de los 80, 90, 2000, 2010 "
    "y de 2020 en adelante (`prop_1990s`…).\n"
    "- **Otras:** el género con mejor promedio, cuánto varían los años de estreno y cuánto tarda en ver "
    "una película desde que se estrena."
)
st.dataframe(silver.head(5), hide_index=True, alt="Primeras filas de la tabla silver")
st.caption(f"Las primeras cinco filas. La primera es el usuario {EJEMPLO}: en la sección 3 armamos sus "
           "variables del modelo paso a paso.")


# ---------------------------------------------------------------------------
# 3 · La trampa de los pocos ratings
# ---------------------------------------------------------------------------
seccion(2, "La trampa de los pocos ratings", "trampa",
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
             "Con pocas películas se acerca a cero; con muchas, casi no cambia. "
             f"El {h.SUAVIZADO} se lee así: con {h.SUAVIZADO} películas del género, la diferencia "
             "queda en la mitad.")
grafico(h.grafico_trampa(T["trampa"]), "Dispersión de la afinidad por género según cantidad de películas")
como_leer('Cada columna del eje x junta los pares usuario × género según cuántas películas de ese género calificó el usuario. El eje y es la diferencia entre su promedio en el género y su promedio general (0 = igual que su promedio). La banda clara cubre el 90 % central de esos pares; la oscura, el 50 %; la línea es la mediana. Gris: el promedio crudo; burnt orange: el amortiguado. Con pocas películas la banda gris es muy ancha, porque cualquier diferencia puede salir por azar, y la de burnt orange se cierra hacia 0.')
with st.expander("Ver los números", icon=":material/table_chart:"):
    st.dataframe(T["trampa"].round(3), hide_index=True, alt="Percentiles de la afinidad por tramo")

st.markdown(
    "La misma lógica vale para el usuario entero: con pocos ratings, todo su perfil es ruido. "
    "¿Dónde poner el corte?"
)
izq, der = st.columns([3, 2], gap="large")
with izq:
    grafico(h.grafico_umbral(T["umbral"]), h.alt_umbral(T["umbral"]))
    como_leer('Cada columna es un mínimo de ratings posible, y el elegido está resaltado. Arriba, cuántos usuarios quedan y qué parte de la muestra son. Abajo, qué parte de las afinidades se calcula con muy pocas películas del género (las frágiles). Subir el mínimo baja las frágiles, pero cuesta usuarios.')
    st.caption(f"Fuente: MovieLens 32M, muestra de {h.fmt(len(silver))} usuarios con ≥ 100 ratings "
               f"(random_state 42). Afinidad frágil: par usuario × género con menos de {h.POCAS} películas, "
               "sobre los géneros que usa el modelo.")
with der:
    st.space("large")
    decision(f"Exigir **{h.MIN_RATINGS} ratings**: conserva {h.fmt(act['modelados'])} usuarios y deja "
             "afuera a los historiales más cortos. No decimos que 150 sea óptimo: es un equilibrio entre "
             "perfiles confiables y usuarios perdidos.")
    sil = T["umbral"].set_index("minimo")["silhouette"]
    altos = sil[sil.index > h.MIN_RATINGS]
    st.caption(f"La separación de los grupos de géneros casi no cambia con el corte (silhouette "
               f"{h.dec(sil[100], 3)} con 100, {h.dec(sil[h.MIN_RATINGS], 3)} con {h.MIN_RATINGS} y entre "
               f"{h.dec(altos.min(), 3)} y {h.dec(altos.max(), 3)} con cortes más altos): lo que se gana "
               "al subir el mínimo son perfiles mejor respaldados, no grupos más nítidos.")
with st.expander("Ver los números", icon=":material/table_chart:"):
    st.dataframe(T["umbral"].round(3), hide_index=True, alt="Usuarios y afinidades frágiles por mínimo")


# ---------------------------------------------------------------------------
# 4 · Qué entra al modelo
# ---------------------------------------------------------------------------
seccion(3, "Qué entra al modelo", "variables", "Entran gustos por género y épocas del historial; nada más")
st.markdown(
    "El silver tiene 54 columnas por usuario. Al modelo entran sólo las que describen **qué le gusta** "
    "(afinidad por género) y **qué mira** (épocas del historial). Todo lo que mide actividad, fechas o "
    "identidad queda afuera, aunque lo usamos para filtrar y para controlar el resultado."
)
st.table(T["variables"].set_index("Variable"))

st.subheader(f"Un ejemplo: las 22 variables del usuario {EJEMPLO}", anchor="ejemplo-variables")
gen_ej, epo_ej, prom_ej, n_ej = h.tabla_fila_usuario(modelos, EJEMPLO)
st.markdown(
    f"Calificó **{h.fmt(n_ej)} películas** con un promedio de **{h.dec(prom_ej, 2)} estrellas**. Con su fila "
    "de la tabla silver calculamos dos bloques de variables, todas interpretables."
)
c1, c2 = st.columns([3, 2], gap="large")
with c1:
    st.markdown("**17 afinidades por género:** cuánto más (o menos) le gusta un género que su propio "
                "promedio, achicado cuando vio pocas películas de ese género (sección 2).")
    st.latex(r"\text{afinidad}_g = \frac{n_g}{n_g + 15}\,\left(\bar r_g - \bar r\right)")
    st.caption("n_g: películas del género que calificó · r̄_g: su promedio en el género · r̄: su promedio general.")
    st.dataframe(gen_ej, hide_index=True, alt=f"Afinidades por género del usuario {EJEMPLO}",
                 column_config={c: st.column_config.NumberColumn(format="%+.3f")
                                for c in ["Diferencia con su promedio", "Afinidad"]}
                 | {"Su promedio en el género": st.column_config.NumberColumn(format="%.2f"),
                    f"Peso n/(n+{h.SUAVIZADO})": st.column_config.NumberColumn(format="%.2f")})
with c2:
    st.markdown("**5 épocas:** qué parte de su historial es de cada época. Se transforma con "
                "arcsen(√p) para que las diferencias cerca de 0 % y de 100 % pesen parecido.")
    st.latex(r"\text{época}_e = \arcsin\sqrt{p_e}")
    st.caption("p_e: proporción de sus películas que se estrenaron en esa época. 2020+ queda afuera.")
    st.dataframe(epo_ej, hide_index=True, alt=f"Épocas del usuario {EJEMPLO}",
                 column_config={"Parte de su historial": st.column_config.ProgressColumn(
                     format="percent", min_value=0, max_value=1),
                     "arcsen(√p)": st.column_config.NumberColumn(format="%.3f")})
mejor = gen_ej.iloc[0]
st.caption(f"Cómo leerlo: su afinidad más alta es {mejor['Género']} ({h.dec(mejor['Afinidad'], 3)}): la califica "
           f"{h.dec(mejor['Diferencia con su promedio'], 2)} estrellas por encima de su promedio, con "
           f"{mejor['Películas que calificó']} películas, así que la diferencia casi no se achica. Estas 22 "
           "variables, para cada uno de los 3.503 usuarios, son la entrada de la PCA.")


# ---------------------------------------------------------------------------
# 5 · Menos dimensiones
# ---------------------------------------------------------------------------
seccion(4, "Menos dimensiones", "pca", "Dos PCA separadas: una para géneros, otra para épocas")
izq, der = st.columns([3, 2], gap="large")
with izq:
    st.markdown(
        "Muchos géneros se mueven juntos (quien valora Acción suele valorar Aventura). Antes de agrupar, "
        "resumimos cada bloque con PCA, por separado, para que los 17 géneros no tapen a las 5 épocas. "
        "K-means usa **todas** las componentes conservadas; los gráficos muestran sólo las dos primeras."
    )
with der:
    decision("Conservar al menos el **80 % de la varianza** en cada bloque.")
ejes_tabla = h.tabla_ejes(T["cargas_todas"])
n_g = int((ejes_tabla["Bloque"] == "Géneros").sum())
n_e = int((ejes_tabla["Bloque"] == "Épocas").sum())
var_g = ejes_tabla.loc[ejes_tabla["Bloque"] == "Géneros", "Varianza que explica"].sum()
var_e = ejes_tabla.loc[ejes_tabla["Bloque"] == "Épocas", "Varianza que explica"].sum()
st.markdown("**La receta de cada PCA.** Las dos siguen los mismos pasos, cada una con sus variables:")
r1, r2 = st.columns(2, gap="large")
with r1:
    with st.container(border=True):
        st.markdown(f":orange[**Géneros**]\n"
                    f"1. **17 afinidades** por género (sección 3).\n"
                    f"2. **Estandarizar:** cada afinidad queda con media 0 y desvío 1, para que ningún género "
                    f"pese más sólo por variar más.\n"
                    f"3. **PCA:** se buscan direcciones que resumen géneros que se mueven juntos, y se guardan "
                    f"hasta llegar al 80 % de la varianza: **{n_g} ejes, {h.dec(var_g * 100, 1)} %**.\n"
                    f"4. Los **{n_g} ejes** son la entrada de K-means y de las distancias del recomendador.")
with r2:
    with st.container(border=True):
        st.markdown(f":blue[**Épocas**]\n"
                    f"1. **5 proporciones** del historial por época, con arcsen(√p) (sección 3).\n"
                    f"2. **Estandarizar**, igual que en géneros.\n"
                    f"3. **PCA** hasta el 80 % de la varianza: **{n_e} ejes, {h.dec(var_e * 100, 1)} %**.\n"
                    f"4. Los **{n_e} ejes** van a K-means y a las distancias.")
grafico(h.grafico_varianza(T["varianza"]), "Varianza acumulada de las dos PCA")
como_leer('Cada línea es una PCA. El eje x es cuántos ejes se guardan y el eje y, qué parte de la información (la varianza) explican juntos. La línea punteada es el 80 % que pedimos; el círculo marca dónde lo alcanza cada PCA.')
st.markdown(
    "**Qué significa cada eje.** Un eje de la PCA es una combinación de las variables: cada una entra con un "
    "peso, la **carga**. Las variables con carga grande definen el eje; el signo sólo dice hacia qué lado "
    "empujan. Por eso cada eje se puede leer como una oposición: un usuario a la izquierda del eje 1 de "
    "géneros tiene más afinidad por drama y crimen; uno a la derecha, por aventura y familia."
)
c1, c2 = st.columns(2, gap="large")
with c1:
    grafico(h.grafico_cargas_eje(T["cargas_todas"], "Géneros"), "Cargas de cada eje de la PCA de géneros")
with c2:
    grafico(h.grafico_cargas_eje(T["cargas_todas"], "Épocas"), "Cargas de cada eje de la PCA de épocas")
como_leer("Cada barra es una variable (un género o una "
           "época) y su largo es la carga: cuánto pesa en el eje elegido. Barras largas = variables que "
           "definen el eje. El color dice hacia qué lado empujan: en burnt orange, hacia la izquierda del eje; "
           "en azul, hacia la derecha. El signo en sí es arbitrario: lo que importa es qué variables quedan "
           "de cada lado.")
st.markdown("**Los ejes, con nombre.** Así se leen en todos los gráficos de la página:")
st.dataframe(ejes_tabla, hide_index=True, alt="Cada eje de las dos PCA, con su varianza y lo que opone",
             column_config={"Varianza que explica": st.column_config.ProgressColumn(
                 format="percent", min_value=0, max_value=0.6)})
st.caption("Los primeros ejes son los más importantes y los más fáciles de leer; los últimos de géneros son "
           "matices chicos (5 % cada uno) que K-means usa igual.")
with st.expander("Ver los números", icon=":material/table_chart:"):
    c1, c2 = st.columns(2)
    c1.dataframe(T["varianza"], hide_index=True, alt="Varianza acumulada por componente",
                 column_config={"acumulada": st.column_config.NumberColumn(format="percent")})
    c2.dataframe(T["cargas_todas"].round(3), hide_index=True, alt="Cargas de cada variable en cada eje")


# ---------------------------------------------------------------------------
# 6 · Cuántos grupos
# ---------------------------------------------------------------------------

particiones, epocas_por_comp = particiones_k(VERSION)

seccion(
    5,
    "Cuántos grupos",
    "grupos",
    "Elegimos cuatro grupos de géneros para distinguir más matices"
)

izq, der = st.columns([3, 2], gap="large")

with izq:
    st.markdown(
        "Para decidir cuántos grupos utilizar, probamos K-means "
        "con distintos valores de K y evaluamos tres medidas: "
        "el **codo**, la **silhouette** y la **estabilidad**.\n\n"
        "K=2 obtiene una mejor separación entre usuarios, "
        "pero queríamos investigar si una división más detallada "
        "permitía identificar diferencias adicionales en sus gustos."
    )

with der:
    decision(
        "**K=4 en géneros y K=2 en épocas.** "
        "Elegimos analizar cuatro grupos de preferencias de géneros, "
        "aunque su silhouette sea menor. "
        "Esto permite estudiar patrones más específicos, "
        "que luego contrastaremos con las afinidades medias "
        "de cada grupo."
    )

# RESULTADOS DE SILHOUETTE, CODO Y ESTABILIDAD
grafico(
    h.grafico_senales(ev),
    "Comparación de las medidas de calidad del agrupamiento"
)

como_leer(
    "Cada panel representa uno de los dos bloques del modelo. "
    "El eje horizontal indica la cantidad de grupos (K). "
    "El codo muestra cuánta dispersión permanece dentro de "
    "los grupos; silhouette mide su separación y estabilidad "
    "indica cuánto cambian al entrenar con distintas muestras. "
    "La banda celeste indica el K que decidimos utilizar."
)

# COMPARACIÓN FIJA K=2 CONTRA K=4
st.subheader("¿Qué cambia al pasar de dos a cuatro grupos?")

st.markdown(
    "Con **K=2**, los usuarios se dividen en dos grandes "
    "conjuntos de preferencias. Sin embargo, estos grupos pueden "
    "contener diferencias internas que vale la pena investigar.\n\n"
    "Para observarlo, comparamos esa división con **K=4**, "
    "manteniendo el mismo espacio PCA y los mismos límites en los ejes. "
    "Así podemos comparar cómo cambia la segmentación "
    "cuando pasamos de una división más simple a otra más detallada."
)

st.altair_chart(h.grafico_k2_k4(particiones), width="content",
                alt="Usuarios en el espacio PCA de géneros, agrupados con K=2 y con K=4")

como_leer(
    "\n\n"
    "**→ Izquierda — K=2:** cada punto representa un usuario y su color indica "
    "el grupo asignado por K-means cuando dividimos a los usuarios en dos conjuntos.\n\n"

    "**→ Derecha — K=4:** cada punto representa un usuario y su color indica "
    "el grupo asignado por K-means cuando dividimos a los usuarios en cuatro conjuntos.\n\n"

    "**→ Misma base de comparación:** ambos gráficos usan los mismos dos primeros "
    "componentes principales de géneros, por lo que podemos comparar visualmente "
    "cómo cambia la partición del mismo espacio.\n\n"

    "**→ Qué observar:** K=2 produce una separación más simple y clara; K=4 agrega "
    "detalle, aunque los grupos quedan menos separados y se superponen más."
)

# COMPARACIÓN DE LAS MÉTRICAS
sil_k2 = ev[
    (ev["bloque"] == "Géneros") & (ev["k"] == 2)
]["silhouette"].iloc[0]

sil_k4 = ev[
    (ev["bloque"] == "Géneros") & (ev["k"] == 4)
]["silhouette"].iloc[0]

st.markdown(
    f"**Resultado:** K=2 alcanza una silhouette de "
    f"**{h.dec(sil_k2, 3)}**, mientras que K=4 alcanza "
    f"**{h.dec(sil_k4, 3)}**. Por lo tanto, K=4 no mejora "
    "la separación: lo elegimos para analizar diferencias "
    "más específicas entre usuarios."
)

# COMPARACIÓN CON EL MODELO CONJUNTO
grafico(
    h.grafico_ari(ari),
    "Coincidencia entre el modelo conjunto y los modelos separados"
)

como_leer(
    "Cada barra indica cuánto coincide el modelo conjunto "
    "con el agrupamiento de géneros o el de épocas. "
    "El ARI vale 1 cuando las particiones coinciden "
    "perfectamente y se aproxima a 0 cuando su coincidencia "
    "no supera la esperable por azar."
)

e2, e3 = epocas_por_comp[2], epocas_por_comp[3]

st.caption(
    f"En épocas, con 3 componentes PCA se conserva "
    f"{h.dec(e3['varianza'] * 100, 1)} % de la varianza "
    f"y la silhouette es {h.dec(e3['silhouette'], 3)}. "
    f"Con 2 componentes, se conserva "
    f"{h.dec(e2['varianza'] * 100, 1)} % y la silhouette "
    f"es {h.dec(e2['silhouette'], 3)}."
)

with st.expander(
    "Ver los números",
    icon=":material/table_chart:"
):
    c1, c2 = st.columns(2)

    c1.dataframe(
        ev.round(3),
        hide_index=True,
        alt="Métricas de K-means para cada cantidad de grupos"
    )

    c2.dataframe(
        ari.round(4),
        hide_index=True,
        alt="Comparación de agrupamientos mediante ARI"
    )
# ---------------------------------------------------------------------------
# 7 · Los perfiles
# ---------------------------------------------------------------------------
cruce, p_chi, v_cramer = h.tabla_cruce(usuarios)
perfil_g, orden_g = h.tabla_perfil_generos(usuarios)
perfil_e = h.tabla_perfil_epocas(usuarios)
rho, medianas = h.pregunta_actividad(usuarios, modelos)

seccion(6, "Los perfiles", "perfiles", "Ocho perfiles, armados con dos preguntas distintas")
st.markdown(
    "Un modelo responde **qué géneros le gustan más que su propio promedio**; el otro, **de qué época es "
    "lo que mira**. Las dos respuestas no dependen una de la otra, así que cada combinación es un perfil."
)
grafico(h.grafico_cruce(cruce, p_chi, v_cramer), "Cruce de los grupos de género y de época")

grupos_ordenados = sorted(cruce["nombre_genero"].unique())

st.caption(
    "**Grupos de géneros (de izquierda a derecha):**  \n"
    + " · ".join(
        f"**G{i}:** {nombre}"
        for i, nombre in enumerate(grupos_ordenados, start=1)
    )
)

como_leer(
    "Cada columna corresponde a un grupo de género y su ancho indica cuántos usuarios tiene. "
    "Dentro de cada columna, los bloques muestran cómo se reparte ese grupo entre las dos "
    "categorías de épocas. La cantidad de usuarios y el porcentaje dentro de cada grupo "
    "aparecen escritos en cada bloque."
)
st.altair_chart(h.grafico_perfiles(perfil_g, orden_g, perfil_e), width="content",
                alt="Afinidad media por género y porcentaje del historial por época en cada grupo")
c1, c2 = st.columns(2, gap="large")
with c1:
    como_leer(
        "Cada fila es un género. Los cuatro puntos muestran la afinidad media de cada grupo: "
        "a la derecha del 0, ese género les gusta más que su propio promedio; a la izquierda, menos. "
        "La línea une el grupo con menor y mayor afinidad: cuanto más larga, más diferencia hay entre los grupos."
    )
with c2:
    como_leer('Para cada época (eje x) hay dos barras, una por grupo de época: qué parte del historial de ese grupo, en promedio, es de películas de esa época. Mide qué ven, no cuánto les gustan.')

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
            ":blue[:material/search: **Lo que encontramos**]  \n"
            f"La actividad casi no se relaciona con los ejes del modelo (|ρ| ≤ {h.dec(rho.abs().max(), 2)}). "
            f"Hay un matiz: el grupo que más califica tiene una mediana de "
            f"{h.fmt(medianas[h.GENEROS_GRUPOS].max())} ratings y el que menos, de "
            f"{h.fmt(medianas[h.GENEROS_GRUPOS].min())}, pero las nubes de los ocho perfiles se superponen casi por completo."
        )
st.altair_chart(h.grafico_actividad_perfiles(usuarios, rho), width="content",
                alt="Nube de usuarios por rating promedio y cantidad de ratings, con reparto por perfil")
como_leer('Cada círculo hueco es un usuario: su rating promedio (eje x) y cuántas películas calificó (eje y, en escala logarítmica). El color es su perfil. Arrastrá sobre la nube para marcar una zona: las barras muestran cómo se reparten los marcados entre los ocho perfiles. Si los perfiles fueran «mucha actividad» contra «poca», el reparto cambiaría mucho al marcar arriba o abajo de la nube.')
with st.expander("Ver los números", icon=":material/table_chart:"):
    c1, c2, c3 = st.columns(3)
    c1.dataframe(cruce, hide_index=True, alt="Usuarios por perfil combinado")
    c2.dataframe(rho.rename("ρ de Spearman").round(3), alt="Correlación de la actividad con cada eje")
    c3.dataframe(medianas.rename("Mediana de ratings"), alt="Mediana de ratings por grupo")



# ---------------------------------------------------------------------------
# 8 · Explorá (la copa)
# ---------------------------------------------------------------------------
seccion(7, "Tu turno", "explora", "Explorá los datos")
st.subheader("Explorá dónde aparecen los gustos en el espacio de géneros")

st.markdown(
    "Con dos ejes ya se ve la estructura: cada género tiene su zona. "
    "Elegí un género abajo para ver dónde están los usuarios a los que más les gusta."
)

st.altair_chart(h.grafico_fanaticos(usuarios), width="content",
                alt="Mapa de calor del espacio de géneros con los usuarios de cada género resaltados")

como_leer(
    'Arriba, el plano de los dos primeros ejes de géneros dividido en celdas: cuanto más oscura la celda, '
    'más usuarios hay en esa zona. Abajo, cuántos usuarios tienen cada género como su afinidad más alta. '
    'Al hacer clic en una barra, los círculos muestran en qué celdas están esos usuarios '
    '(más grande = más usuarios). Doble clic para volver a ver todos.'
)
st.subheader("Explorá los usuarios")

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
    como_leer("Cada punto es un usuario en los dos primeros ejes del espacio elegido, con el color de su grupo. "
              "Arrastrá para marcar un rectángulo: las barras de la derecha muestran el promedio de los "
              "marcados en cada variable, y la raya negra, el de todos los usuarios. Clic en la leyenda para "
              "dejar un solo grupo.")


copa()


# ---------------------------------------------------------------------------
# 9 · Un usuario y sus recomendaciones
# ---------------------------------------------------------------------------
seccion(8, "Tu turno", "recomendador", "Un usuario, sus vecinos y sus películas")
st.markdown(
    "Elegí un usuario y te mostramos, paso a paso, cómo le recomendamos películas. La idea es la de un "
    "amigo con gustos parecidos: buscamos a los usuarios que más se le parecen y le recomendamos lo que a "
    "ellos les gustó y él todavía no vio."
)
with st.container(border=True):
    st.markdown(
        ":blue[**Cómo recomendamos, en tres pasos**]\n"
        "1. **Ubicamos al usuario** con las coordenadas que le da el modelo: los **8 ejes de géneros** y los "
        "**3 ejes de épocas** de la sección 4.\n"
        "2. **Medimos qué tan lejos está de cada uno de los otros 3.502 usuarios.** En cada bloque usamos la "
        "distancia de siempre (la euclídea: la raíz de la suma de las diferencias al cuadrado, eje por eje), "
        "dividida por la raíz de la cantidad de ejes para que los dos bloques pesen igual, y promediamos:"
    )
    st.latex(r"d(u, v) = \frac{1}{2}\left(\frac{\lVert g_u - g_v \rVert}{\sqrt{8}}"
             r" + \frac{\lVert e_u - e_v \rVert}{\sqrt{3}}\right)")
    st.markdown(
        "3. **Los 100 más cercanos votan.** Cada uno vota por las películas que calificó con 4 o más y que el "
        "usuario no vio; su voto pesa más cuanto más cerca está. Recomendamos las más votadas."
    )


@st.fragment
def un_usuario():
    ids, zg, zd = espacios(VERSION)
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

    st.altair_chart(h.grafico_usuario_espacios(usuarios, usuario, vecinos), width="content",
                    alt="Posición del usuario y sus vecinos en el espacio de géneros y en el de épocas")
    como_leer("Cada punto gris es un usuario, en los dos primeros ejes de cada espacio. En burnt orange está el "
              "usuario elegido y en chartreuse, sus 5 más parecidos. Ojo: los parecidos se buscan con todos los "
              "ejes (8 de géneros y 3 de épocas), así que en este plano de dos ejes pueden verse algo separados.")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        grafico(h.grafico_afinidades_usuario(fila), "Afinidades por género del usuario")
    with c2:
        grafico(h.grafico_epocas_usuario(fila), "Historial del usuario por época")
    como_leer("A la izquierda, su afinidad por cada género, con las mismas cuentas de la sección 3: a la derecha "
              "del 0 están los géneros que califica por encima de su propio promedio. A la derecha, qué parte "
              "de su historial es de cada época.")

    st.subheader("Sus vecinos: por qué se parecen", icon=":material/group:")
    lejanos = similitud["distancia"]
    st.markdown(
        f"Estos son sus 5 usuarios más cercanos. Para tener una escala: la distancia a los otros usuarios va de "
        f"{h.dec(lejanos.min(), 2)} a {h.dec(lejanos.max(), 2)}, y la mitad está a más de "
        f"{h.dec(lejanos.median(), 2)}. Al lado, en qué se parecen: los géneros que están entre los 4 favoritos "
        "de los dos y las películas que a los dos les gustaron (4 o más)."
    )
    st.dataframe(h.explicar_vecinos(usuario, similitud, usuarios, ratings, peliculas), hide_index=True,
                 alt="Los 5 usuarios más parecidos y en qué se parecen",
                 column_config={c: st.column_config.NumberColumn(format="%.3f")
                                for c in ["Distancia", "En géneros", "En épocas"]})
    como_leer("Cada fila es un vecino. La distancia total es el promedio de la distancia en géneros y en épocas "
              "(menos es más parecido). Los vecinos se parecen en qué tipo de cine les gusta, no necesariamente "
              "en las películas concretas: puede haber un vecino muy cercano que no comparta ninguna película "
              "que les haya gustado a los dos, porque vieron películas distintas del mismo estilo. Por eso usamos "
              "100 vecinos y no 5: entre todos cubren muchas más películas.")

    
    
    rec, cantidad = h.recomendar(similitud, ratings, peliculas, usuario)
    favoritas = h.peliculas_favoritas(ratings, peliculas, usuario)

    st.subheader(
        "Recomendaciones y películas mejor calificadas",
        icon=":material/theaters:"
    )

    st.markdown(
        "Queremos comparar las **películas que nuestro sistema recomienda** "
        "con las **películas que el usuario realmente calificó mejor**. "
        "A la izquierda mostramos las recomendaciones obtenidas a partir de "
        "sus 100 usuarios más parecidos; a la derecha, las nueve películas "
        "a las que el usuario seleccionado les dio sus puntuaciones más altas.\n\n"
        "Esta comparación permite observar si las recomendaciones son "
        "coherentes con sus preferencias. Además, comprobamos que ninguna "
        "de las películas recomendadas figure entre las que ya calificó "
        "en nuestro conjunto de datos."
    )

    col_rec, col_fav = st.columns(2, gap="large")

    # GRÁFICO IZQUIERDO: RECOMENDACIONES
    with col_rec:
        st.markdown("**Las 9 películas que le recomendamos**")

        if rec.empty:
            st.warning(
                "No encontramos películas con evidencia suficiente "
                "para este usuario."
            )
        else:
            grafico(
                h.grafico_top(rec, cantidad, compacto=True),
                "Nueve películas recomendadas según el apoyo de sus vecinos"
            )

            st.caption(
                "Apoyo ponderado de los 100 vecinos más cercanos. "
                "Cuanto mayor es el porcentaje, mayor es el respaldo "
                "de los vecinos a esa recomendación."
            )

    # GRÁFICO DERECHO: PELÍCULAS MEJOR CALIFICADAS
    with col_fav:
        st.markdown("**Las 9 películas que mejor calificó**")

        if favoritas.empty:
            st.warning(
                "No encontramos películas calificadas con título "
                "disponible para este usuario."
            )
        else:
            grafico(
                h.grafico_favoritas(favoritas),
                "Las nueve películas mejor calificadas por el usuario"
            )

            st.caption(
                "Calificación del usuario expresada sobre el máximo "
                "de 5 estrellas. Cuando hay empates, las películas "
                "se ordenan alfabéticamente."
            )

    # EXPLICACIÓN DE AMBOS GRÁFICOS
    como_leer(
        "\n\n"
        " **→ Izquierda — Recomendaciones:** cada barra representa una película "
        "que el usuario no calificó en nuestros datos. Su longitud indica el "
        "apoyo ponderado de sus 100 vecinos más cercanos: cuanto mayor es "
        "el porcentaje, mayor es el respaldo de usuarios con gustos similares.\n\n"

        " **→ Derecha — Películas mejor calificadas:** cada barra representa "
        "una película que el usuario sí calificó. Su longitud muestra la "
        "puntuación que le dio: 5 estrellas equivalen al 100 %, "
        "4 estrellas al 80 %, y así sucesivamente.\n\n"

        " **→ Misma escala, distinto significado:** ambos gráficos van del "
        "0 al 100 %, pero el izquierdo mide el apoyo de los vecinos y el "
        "derecho la calificación individual. Por eso, los porcentajes "
        "no se comparan directamente.\n\n"
    )

    # CONTROL DE PELÍCULAS YA CALIFICADAS
    if not rec.empty:
        ya_calificadas = set(
            ratings.loc[
                ratings["userId"] == usuario, "movieId"
            ]
        )

        coincidencias = len(
            set(rec["movieId"]) & ya_calificadas
        )

        st.markdown(
            f"**Control de coincidencias:** {coincidencias} de "
            f"{len(rec)} películas recomendadas aparecen entre las "
            "que este usuario calificó en el conjunto de datos utilizado."
        )

        # EXPLICACIÓN DE LA CANTIDAD DE RECOMENDACIONES
        with st.container(border=True):
            st.markdown(
                f":blue[**¿Por qué {len(rec)} y no otra cantidad?**] "
                "Lo medimos escondiendo ratings "
                "(`evaluacion_recomendador.ipynb`, Parte 6): "
                "la primera recomendación acierta 4 de cada 10 veces, "
                "y la precisión baja de a poco. No hay un salto entre "
                "una película y la siguiente donde cortar, así que "
                "usamos un criterio de calidad: cada una de las primeras "
                f"{h.N_RECOMENDACIONES} acierta al menos 1 de cada 5 veces, "
                "y de ahí para abajo, menos. En todas las posiciones "
                "le gana a recomendar las películas más populares."
            )

un_usuario()


st.markdown("#### Limitaciones y alcance del proyecto")

limitaciones = [
    (
        "Trabajamos con una muestra limitada.",
        "De los 200.948 usuarios de MovieLens, seleccionamos 5.000, "
        "con un máximo de 500 calificaciones por persona. Finalmente, "
        "3.503 usuarios cumplieron los requisitos del modelo. "
        "También limitamos la consulta a TMDB a las 10.000 películas "
        "más vistas de la muestra, por lo que los resultados no "
        "representan necesariamente a todos los espectadores y películas."
    ),
    (
        "Los usuarios nuevos quedan afuera.",
        "El modelo exige al menos 150 calificaciones para construir "
        "perfiles suficientemente respaldados. Esto reduce el ruido "
        "de historiales pequeños, pero impide recomendar a usuarios "
        "con pocas valoraciones, como alguien que calificó solamente "
        "20 películas."
    ),
    (
        "Las épocas no necesariamente reflejan preferencias.",
        "Analizamos qué proporción del historial corresponde a cada "
        "época de estreno, pero tener más películas calificadas de "
        "una década no significa necesariamente preferirla."
    ),
    (
        "No podemos garantizar que una película no haya sido vista.",
        "MovieLens registra calificaciones, no reproducciones. "
        "El recomendador excluye las películas ya calificadas en los "
        "datos disponibles, pero no puede saber si el usuario vio "
        "otras películas sin calificarlas."
    )
]

# Misma paleta y orden que COLOR_GENERO en historia.py
colores = [
    h.CHARTREUSE,
    h.QUEMADO,
    h.NAVY,
    h.ELECTRICO
]

with st.container(border=True):
    for i, (titulo, descripcion) in enumerate(limitaciones):
        col_num, col_texto = st.columns(
            [1, 13],
            vertical_alignment="top"
        )

        with col_num:
            st.markdown(
                f"<span style='color:{colores[i]};"
                f"font-size:28px;font-weight:700;'>"
                f"{i+1:02d}</span>",
                unsafe_allow_html=True
            )

        with col_texto:
            st.markdown(
                f"**{titulo}**\n\n{descripcion}"
            )

        if i < len(limitaciones) - 1:
            st.divider()


st.space("medium")
st.caption(
    f"Fuente: MovieLens 32M · {h.fmt(act['usuarios'])} usuarios al azar con ≥ 100 ratings (random_state 42) · "
    f"tope de 500 ratings · TMDB para géneros y fechas · modelo con los {h.fmt(act['modelados'])} usuarios "
    "de 150 ratings o más. Los grupos son descriptivos: no hay una etiqueta verdadera de «tipo de espectador»."
)
