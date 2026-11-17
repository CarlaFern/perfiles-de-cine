import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
alt.themes.enable("dark")
from modelado import entrenar_desde_csv

@st.cache_resource
def cargar_modelos():
    return entrenar_desde_csv("usuarios_generos.csv")

modelos = cargar_modelos()

@st.cache_data
def cargar_ratings():
    return pd.read_csv("ratings_selected_users.csv")


@st.cache_data
def cargar_peliculas():
    return pd.read_csv("tmdb_movies_prepared.csv")


ratings = cargar_ratings()
peliculas = cargar_peliculas()

st.set_page_config(
    page_title="Perfiles de cine",
    page_icon="🎬",
    layout="wide"
)

# =========================
# ESTILO
# =========================

st.markdown("""
<style>

.stApp {
    background-color: #0b0b0d;
    color: white;
}

.block-container {
    max-width: 1200px;
    padding-top: 4rem;
    padding-bottom: 4rem;
}

[data-testid="stSidebar"] {
    background-color: #111114;
}

.titulo-rojo {
    color: #e50914;
    font-size: 14px;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 2px;
}

.texto-secundario {
    color: #bbbbc2;
    font-size: 18px;
}

.tarjeta {
    background-color: #161619;
    border: 1px solid #29292e;
    border-radius: 15px;
    padding: 20px;
    min-height: 105px;
}

.numero {
    font-size: 36px;
    font-weight: bold;
}

.descripcion {
    color: #aaaab2;
}

.decision {
    background-color: #151517;
    border-left: 4px solid #e50914;
    padding: 18px;
    border-radius: 8px;
}
div.stButton > button {
    width: 100% !important;
    display: flex !important;
    justify-content: flex-start !important;
    align-items: center !important;
    text-align: left !important;
    padding-left: 14px !important;
}

div.stButton > button > div {
    width: 100% !important;
    justify-content: flex-start !important;
    text-align: left !important;
}

div.stButton > button p {
    width: 100% !important;
    text-align: left !important;
    margin: 0 !important;
}

</style>
""", unsafe_allow_html=True)


# =========================
# MENÚ
# =========================

with st.sidebar:

    st.title("🎬 Perfiles de cine")
    st.caption("Ciencia de Datos · Grupo 5K09-05")

    if "seccion" not in st.session_state:
        st.session_state.seccion = "El proyecto"

    def ir_a(nombre):
        st.session_state.seccion = nombre

    st.markdown("### La historia")

    for nombre in [
        "El proyecto",
        "Los datos",
        "Lo que encontramos",
        "El modelo"
    ]:
        activo = st.session_state.seccion == nombre

        st.button(
            nombre,
            key=f"nav_{nombre}",
            type="primary" if activo else "tertiary",
            width="stretch",
            on_click=ir_a,
            args=(nombre,)
        )

    st.markdown("### Tu turno")

    for nombre in [
        "Explorá",
        "Recomendador"
    ]:
        activo = st.session_state.seccion == nombre

        st.button(
            nombre,
            key=f"nav_{nombre}",
            type="primary" if activo else "tertiary",
            width="stretch",
            on_click=ir_a,
            args=(nombre,)
        )

    st.divider()

    st.caption("MovieLens + TMDB")

seccion = st.session_state.seccion


# =========================
# EL PROYECTO
# =========================

if seccion == "El proyecto":

    st.markdown(
        '<div class="titulo-rojo">El proyecto</div>',
        unsafe_allow_html=True
    )

    st.title(
        "¿Podemos encontrar usuarios con gustos similares "
        "y usar esa similitud para recomendar películas "
        "que todavía no calificaron?"
    )

    st.markdown(
        """
        <div class="texto-secundario">
        Analizamos patrones de valoración por género y época
        para representar usuarios, encontrar comportamientos
        similares y orientar recomendaciones.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="tarjeta">
            <div class="numero">5.000</div>
            <div class="descripcion">
                usuarios seleccionados inicialmente
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="tarjeta">
            <div class="numero">3.503</div>
            <div class="descripcion">
                usuarios utilizados en el modelado
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="tarjeta">
            <div class="numero">150</div>
            <div class="descripcion">
                valoraciones mínimas por usuario
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.caption(
        "Fuente: MovieLens para las valoraciones de usuarios "
        "y TMDB para los metadatos de las películas."
    )


# =========================
# LOS DATOS
# =========================

elif seccion == "Los datos":

    st.markdown(
        '<div class="titulo-rojo">Los datos</div>',
        unsafe_allow_html=True
    )

    st.title(
        "Necesitamos suficiente historial para representar "
        "el gusto de cada usuario"
    )

    st.write(
        """
        Partimos de 5.000 usuarios y comparamos distintos mínimos
        de valoraciones. Buscamos evitar perfiles construidos con
        muy poca información sin perder demasiados usuarios.
        """
    )

    # Datos reales del notebook
    df_umbrales = pd.DataFrame({

        "Mínimo": [
            100,
            150,
            200,
            250,
            300,
            400
        ],

        "Usuarios": [
            5000,
            3503,
            2595,
            1991,
            1616,
            1105
        ]
    })

    # Gráfico base
    barras = (
        alt.Chart(df_umbrales)
        .mark_bar(
            cornerRadiusTopLeft=5,
            cornerRadiusTopRight=5
        )
        .encode(

            x=alt.X(
                "Mínimo:O",
                title="Mínimo de valoraciones por usuario",
                axis=alt.Axis(labelAngle=0)
            ),

            y=alt.Y(
                "Usuarios:Q",
                title="Usuarios conservados",
                scale=alt.Scale(
                    domain=[0, 5200]
                )
            ),

            color=alt.condition(

                alt.datum.Mínimo == 150,

                alt.value("#e50914"),

                alt.value("#45454d")
            ),

            tooltip=[
                alt.Tooltip(
                    "Mínimo:O",
                    title="Valoraciones mínimas"
                ),

                alt.Tooltip(
                    "Usuarios:Q",
                    title="Usuarios conservados"
                )
            ]
        )
        .properties(
            height=380
        )
    )

    # Números arriba de las barras
    etiquetas = (
        alt.Chart(df_umbrales)
        .mark_text(
            dy=-10,
            fontSize=14,
            fontWeight="bold"
        )
        .encode(

            x="Mínimo:O",

            y="Usuarios:Q",

            text=alt.Text(
                "Usuarios:Q",
                format=","
            ),

            color=alt.value("white")
        )
    )

    chart = (barras + etiquetas).configure_axis(
        gridColor="#2a2a2f",
        domainColor="#666666",
        tickColor="#666666"
    )

    st.altair_chart(
        chart,
        width="stretch"
    )

    st.markdown("""
    <div class="decision">

    <b>Decisión:</b> usar un mínimo de <b>150 valoraciones</b>.

    Así conservamos <b>3.503 de 5.000 usuarios</b>
    y evitamos historiales con muy poca información.

    </div>
    """, unsafe_allow_html=True)

    st.write("")

    st.subheader("¿De dónde salen los datos?")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown("""
        <div class="tarjeta">

        <b>⭐ MovieLens</b>

        <br>

        <span class="descripcion">
        Usuario, película, puntuación
        y fecha de valoración.
        </span>

        </div>
        """, unsafe_allow_html=True)


    with c2:

        st.markdown("""
        <div class="tarjeta">

        <b>🎞️ TMDB</b>

        <br>

        <span class="descripcion">
        Géneros, año de estreno
        y metadatos de películas.
        </span>

        </div>
        """, unsafe_allow_html=True)


    with c3:

        st.markdown("""
        <div class="tarjeta">

        <b>👤 Dataset final</b>

        <br>

        <span class="descripcion">
        Una fila representa
        un usuario.
        </span>

        </div>
        """, unsafe_allow_html=True)


    # Información técnica
    with st.expander(
        "Cómo lo sabemos · obtención y preparación"
    ):

        st.markdown("""
        **Fuentes**

        - MovieLens aporta ratings por usuario y película.
        - TMDB aporta metadatos.
        - La integración se realiza mediante Airflow.

        **Unidad de análisis**

        Una fila representa un usuario.

        **Selección**

        - Mínimo 100 → 5.000 usuarios.
        - Mínimo 150 → 3.503 usuarios.
        - Mínimo 200 → 2.595.
        - Mínimo 250 → 1.991.
        - Mínimo 300 → 1.616.
        - Mínimo 400 → 1.105.

        Se eligió **150 valoraciones** como compromiso entre
        cantidad de información por usuario y cobertura.

        **Historiales grandes**

        Para usuarios con muchos ratings se procesan como máximo
        500 valoraciones.

        `cantidad_valoraciones_total` conserva el total original.

        **Control de fuga**

        `userId` no participa como característica del modelo.

        Tampoco se utilizan columnas que ya contengan
        la asignación de cluster.

        El problema es **no supervisado**.
        """)


# =========================
# LO QUE ENCONTRAMOS
# =========================
elif seccion == "Lo que encontramos":

    st.markdown(
        '<div class="titulo-rojo">Lo que encontramos</div>',
        unsafe_allow_html=True
    )

    st.title(
        "Los géneros y las épocas describen "
        "aspectos diferentes del mismo usuario"
    )

    st.write(
        "Cada punto representa un usuario: cuanto más cerca aparecen dos puntos, "
        "más parecidos son sus patrones dentro de esa dimensión."
    )

    # Copiamos los resultados del modelo
    datos = modelos.etiquetas.reset_index()

    # =========================
    # GRÁFICO DE GÉNEROS
    # =========================

    grafico_generos = (
        alt.Chart(datos)
        .mark_circle(
            size=45,
            opacity=0.55
        )
        .encode(

            x=alt.X(
                "G_PC1:Q",
                title="Dimensión de género 1",
                scale=alt.Scale(zero=False)
            ),

            y=alt.Y(
                "G_PC2:Q",
                title="Dimensión de género 2",
                scale=alt.Scale(zero=False)
            ),

            color=alt.Color(
                "nombre_genero:N",
                title="Grupo de género",
                scale=alt.Scale(
                    range=[
                        "#e50914",
                        "#4e79a7"
                    ]
                )
            ),

            tooltip=[
                alt.Tooltip(
                    "userId:O",
                    title="Usuario"
                ),

                alt.Tooltip(
                    "nombre_genero:N",
                    title="Perfil de género"
                )
            ]
        )
        .properties(
            height=430,
            title="Afinidades por género"
        )
        .interactive()
    )


    # =========================
    # GRÁFICO DE ÉPOCAS
    # =========================

    grafico_decadas = (
        alt.Chart(datos)
        .mark_circle(
            size=45,
            opacity=0.55
        )
        .encode(

            x=alt.X(
                "D_PC1:Q",
                title="Dimensión temporal 1",
                scale=alt.Scale(zero=False)
            ),

            y=alt.Y(
                "D_PC2:Q",
                title="Dimensión temporal 2",
                scale=alt.Scale(zero=False)
            ),

            color=alt.Color(
                "nombre_decada:N",
                title="Grupo temporal",
                scale=alt.Scale(
                    range=[
                        "#f28e2b",
                        "#59a14f"
                    ]
                )
            ),

            tooltip=[
                alt.Tooltip(
                    "userId:O",
                    title="Usuario"
                ),

                alt.Tooltip(
                    "nombre_decada:N",
                    title="Perfil temporal"
                )
            ]
        )
        .properties(
            height=430,
            title="Películas calificadas según época de estreno"
        )
        .interactive()
    )


    # =========================
    # MOSTRAR LADO A LADO
    # =========================

    col1, col2 = st.columns(2)

    with col1:

        st.altair_chart(
            grafico_generos,
            width="stretch"
        )

        st.caption(
            "La cercanía representa similitud en los patrones "
            "de valoración por género."
        )


    with col2:

        st.altair_chart(
            grafico_decadas,
            width="stretch"
        )

        st.caption(
            "La cercanía representa similitud en la distribución "
            "de películas calificadas según su época de estreno."
        )


    # Aclaración conceptual
    st.caption(
        "Importante: la época no indica qué década le gusta más al usuario; "
        "representa cuándo fueron estrenadas las películas presentes en su historial."
    )   


    st.markdown("""
    <div class="decision">

    <b>Hallazgo:</b> los agrupamientos por género y por época
    no cuentan exactamente lo mismo.

    Un usuario puede parecerse a otro en sus preferencias por género
    y, al mismo tiempo, tener un historial temporal diferente.

    </div>
    """, unsafe_allow_html=True)


    # =========================
    # CÓMO LO SABEMOS
    # =========================

    with st.expander(
        "Cómo lo sabemos · hipótesis y transformaciones"
    ):

        st.markdown("""
        **Preferencias por género**

        Se compara la valoración media de cada género con
        el promedio general del propio usuario.

        Para evitar que una única película determine una preferencia,
        se aplica suavizado:

        `n / (n + 15)`

        Solo se incorporan géneros con cobertura suficiente
        entre los usuarios.

        ---

        **Épocas**

        No se utiliza el rating promedio de cada década.

        Se calcula qué proporción de las películas calificadas
        por el usuario pertenece a cada época de estreno.

        Estas proporciones se transforman mediante:

        `arcsin(sqrt(p))`

        ---

        **Reducción dimensional**

        Se aplica PCA antes de K-means.

        - Géneros: 8 componentes retenidas.
        - Décadas: 3 componentes retenidas.

        Estos gráficos muestran únicamente las dos primeras
        componentes para poder visualizar los usuarios.

        K-means utiliza todas las componentes retenidas.

        ---

        **Resultado**

        No encontramos evidencia de asociación entre los grupos
        de género y los grupos temporales.

        - χ²: `p ≈ 0,9915`
        - V de Cramér: aproximadamente `0`

        Esto sugiere que ambas dimensiones aportan
        información complementaria.
        """)



# =========================
# MODELO
# =========================

elif seccion == "El modelo":

    st.markdown(
        '<div class="titulo-rojo">El modelo</div>',
        unsafe_allow_html=True
    )

    st.title(
        "Separar géneros y épocas conserva información "
        "que se pierde al juntarlos"
    )

    st.write(
    "Probamos un modelo conjunto, pero su agrupamiento terminó "
    "pareciéndose casi por completo al modelo temporal. "
    "Por eso conservamos géneros y épocas como dos dimensiones separadas."
    )

    # =========================
    # COMPARACIÓN DE MODELOS
    # =========================



    st.markdown("""
    <div class="decision">

    <b>Decisión:</b> mantener dos modelos separados.

    El modelo conjunto reproduce principalmente la estructura temporal
    y prácticamente no conserva el agrupamiento obtenido a partir de géneros.

    </div>
    """, unsafe_allow_html=True)


    # =========================
    # COMPARACIÓN DEL MODELO CONJUNTO
    # =========================

    st.subheader(
        "¿A cuál de los dos modelos se parece el conjunto?"
    )

    ari = pd.DataFrame({
        "Comparación": [
            "Con décadas",
            "Con géneros"
        ],
        "ARI": [
            0.922,
            0.0
        ]
    })

    barras_ari = (
        alt.Chart(ari)
        .mark_bar(
            cornerRadiusTopRight=6,
            cornerRadiusBottomRight=6
        )
        .encode(
            y=alt.Y(
                "Comparación:N",
                sort=["Con décadas", "Con géneros"],
                title=None
            ),

            x=alt.X(
                "ARI:Q",
                title="Similitud entre agrupamientos (ARI)",
                scale=alt.Scale(domain=[0, 1])
            ),

            color=alt.condition(
                alt.datum["Comparación"] == "Con décadas",
                alt.value("#e50914"),
                alt.value("#55555d")
            ),

            tooltip=[
                "Comparación:N",
                alt.Tooltip(
                    "ARI:Q",
                    title="ARI",
                    format=".3f"
                )
            ]
        )
    )

    texto_ari = (
        alt.Chart(ari)
        .mark_text(
            align="left",
            dx=8,
            fontWeight="bold",
            color="white"
        )
        .encode(
            y=alt.Y(
                "Comparación:N",
                sort=["Con décadas", "Con géneros"]
            ),

            x="ARI:Q",

            text=alt.Text(
                "ARI:Q",
                format=".3f"
            )
        )
    )

    grafico_ari = (
        barras_ari + texto_ari
    ).properties(
        height=180
    ).configure_axis(
        gridColor="#2a2a2f",
        domainColor="#666666",
        tickColor="#666666"
    )

    st.altair_chart(
        grafico_ari,
        width="stretch"
    )

    st.caption(
        "Un ARI cercano a 1 indica que dos agrupamientos son muy parecidos. "
        "El modelo conjunto coincide fuertemente con el temporal y casi nada con el de géneros."
    )


    # =========================
    # CÓMO LO SABEMOS
    # =========================

    with st.expander(
        "Cómo lo sabemos · selección y estabilidad del modelo"
    ):

        st.markdown("""
        **Algoritmo**

        Para géneros y décadas se utilizó:

        `StandardScaler → PCA → K-means`

        ---

        **PCA**

        Se conservó al menos el 80 % de la varianza.

        - Géneros: 8 componentes.
        - Décadas: 3 componentes.

        Los gráficos anteriores muestran PC1 y PC2,
        pero K-means utiliza todas las componentes retenidas.

        ---

        **Elección de k**

        Se compararon valores de `k` entre 2 y 6.

        El modelo principal quedó con:

        - Géneros: `k = 2`
        - Décadas: `k = 2`

        Silhouette:

        - Géneros: `0.180`
        - Décadas: `0.424`
        - Modelo conjunto: `0.224`

        Como experimento adicional se probó `k = 5`
        para géneros.

        Permitía describir perfiles temáticos más específicos,
        pero la separación empeoraba:

        - Géneros con `k = 5`: silhouette `0.102`

        Por eso no se eligió como modelo principal.

        ---

        **Estabilidad**

        Al repetir el entrenamiento sobre submuestras del 80 %:

        - ARI medio en géneros: aproximadamente `0.968`
        - ARI medio en décadas: aproximadamente `0.990`

        Esto indica que los grupos son bastante estables
        frente a cambios moderados en la muestra.

        ---

        **Modelo conjunto**

        El modelo que mezclaba géneros y décadas obtuvo:

        - silhouette: `0.224`
        - ARI con décadas: aproximadamente `0.922`
        - ARI con géneros: aproximadamente `0`

        Por eso se decidió conservar ambas dimensiones
        por separado.

        ---

        **Límite**

        Los clusters son grupos descriptivos.

        No existe una etiqueta verdadera de
        "tipo de espectador" y K-means no entrega
        probabilidades de pertenencia.
        """)


# =========================
# EXPLORÁ
# =========================

elif seccion == "Explorá":

    st.markdown(
        '<div class="titulo-rojo">Tu turno · Explorá</div>',
        unsafe_allow_html=True
    )

    st.title(
        "Explorá los usuarios y descubrí quiénes se parecen"
    )

    st.write(
        """
        Filtrá los usuarios, elegí uno y observá cómo se ubica
        según sus patrones de género y de época.
        """
    )

    # =========================
    # PREPARAR DATOS
    # =========================

    datos = modelos.etiquetas.reset_index()

    # Agregamos datos generales del usuario
    generales = (
        modelos.entrenamiento
        .reset_index()[
            [
                "userId",
                "cantidad_valoraciones",
                "cantidad_valoraciones_total",
                "rating_promedio_general"
            ]
        ]
    )

    datos = datos.merge(
        generales,
        on="userId",
        how="left"
    )

    datos["Perfil"] = (
        datos["nombre_genero"]
        + " · "
        + datos["nombre_decada"]
    )

    # =========================
    # CONTROLES
    # =========================

    st.subheader("Filtrá los usuarios")

    c1, c2 = st.columns([1.4, 1])

    perfiles_disponibles = sorted(
        datos["Perfil"].unique()
    )

    with c1:

        perfiles_elegidos = st.multiselect(
            "Perfil",
            options=perfiles_disponibles,
            default=perfiles_disponibles
        )

    with c2:

        minimo_ratings = st.slider(
            "Mínimo de valoraciones procesadas",
            min_value=150,
            max_value=500,
            value=150,
            step=25
        )

    filtrados = datos[
        datos["Perfil"].isin(perfiles_elegidos)
        &
        (datos["cantidad_valoraciones"] >= minimo_ratings)
    ].copy()

    st.caption(
        f"Mostrando {len(filtrados):,} usuarios de {len(datos):,}."
        .replace(",", ".")
    )

    if filtrados.empty:

        st.warning(
            "Con estos filtros no quedan usuarios. "
            "Probá bajar el mínimo o seleccionar más perfiles."
        )

        st.stop()


    # =========================
    # ELEGIR USUARIO
    # =========================

    usuario = st.selectbox(
        "Elegí un usuario para analizar",
        options=filtrados["userId"].tolist(),
        format_func=lambda x: f"Usuario {x}"
    )


    # =========================
    # BUSCAR USUARIOS SIMILARES
    # =========================

    # Usamos TODAS las componentes PCA retenidas,
    # no solamente PC1 y PC2.

    entrenamiento = modelos.entrenamiento

    Z_generos = modelos.generos["preparar"].transform(
        entrenamiento
    )

    Z_decadas = modelos.decadas["preparar"].transform(
        entrenamiento
    )

    ids_modelo = entrenamiento.index.to_numpy()

    posicion = np.where(ids_modelo == usuario)[0][0]

    usuario_g = Z_generos[posicion]
    usuario_d = Z_decadas[posicion]

    # Distancia dentro de cada espacio.
    # Dividimos por sqrt(n_componentes) para evitar
    # que el bloque con más componentes pese más.

    dist_g = (
        np.linalg.norm(
            Z_generos - usuario_g,
            axis=1
        )
        / np.sqrt(Z_generos.shape[1])
    )

    dist_d = (
        np.linalg.norm(
            Z_decadas - usuario_d,
            axis=1
        )
        / np.sqrt(Z_decadas.shape[1])
    )

    # Distancia combinada:
    # mismo peso para género y época.

    distancia_total = (
        dist_g + dist_d
    ) / 2

    similitud = pd.DataFrame({
        "userId": ids_modelo,
        "distancia": distancia_total
    })

    similitud = similitud[
        similitud["userId"] != usuario
    ]

    vecinos = (
        similitud
        .sort_values("distancia")
        .head(5)
    )
    ids_vecinos = vecinos["userId"].tolist()



    # =========================
    # MARCAR PUNTOS
    # =========================

    filtrados["tipo_punto"] = "Otros usuarios"

    filtrados.loc[
        filtrados["userId"].isin(ids_vecinos),
        "tipo_punto"
    ] = "Usuarios similares"

    filtrados.loc[
        filtrados["userId"] == usuario,
        "tipo_punto"
    ] = "Usuario seleccionado"


    # =========================
    # SCATTER GÉNEROS
    # =========================

    grafico_generos = (
        alt.Chart(filtrados)
        .mark_circle()
        .encode(

            x=alt.X(
                "G_PC1:Q",
                title="Dimensión de género 1",
                scale=alt.Scale(zero=False)
            ),

            y=alt.Y(
                "G_PC2:Q",
                title="Dimensión de género 2",
                scale=alt.Scale(zero=False)
            ),

            color=alt.Color(
                "tipo_punto:N",
                title=None,

                scale=alt.Scale(
                    domain=[
                        "Otros usuarios",
                        "Usuarios similares",
                        "Usuario seleccionado"
                    ],

                    range=[
                        "#55555d",
                        "#f0a202",
                        "#e50914"
                    ]
                )
            ),

            size=alt.Size(
                "tipo_punto:N",

                scale=alt.Scale(
                    domain=[
                        "Otros usuarios",
                        "Usuarios similares",
                        "Usuario seleccionado"
                    ],

                    range=[
                        20,
                        100,
                        180
                    ]
                ),

                legend=None
            ),

            opacity=alt.condition(
                alt.datum.tipo_punto == "Otros usuarios",
                alt.value(0.35),
                alt.value(1)
            ),

            tooltip=[
                alt.Tooltip(
                    "userId:O",
                    title="Usuario"
                ),

                alt.Tooltip(
                    "nombre_genero:N",
                    title="Perfil de género"
                ),

                alt.Tooltip(
                    "Perfil:N",
                    title="Perfil combinado"
                )
            ]
        )
        .properties(
            height=430,
            title="Espacio de géneros"
        )
        .interactive()
    )


    # =========================
    # SCATTER ÉPOCAS
    # =========================

    grafico_decadas = (
        alt.Chart(filtrados)
        .mark_circle()
        .encode(

            x=alt.X(
                "D_PC1:Q",
                title="Dimensión temporal 1",
                scale=alt.Scale(zero=False)
            ),

            y=alt.Y(
                "D_PC2:Q",
                title="Dimensión temporal 2",
                scale=alt.Scale(zero=False)
            ),

            color=alt.Color(
                "tipo_punto:N",
                title=None,

                scale=alt.Scale(
                    domain=[
                        "Otros usuarios",
                        "Usuarios similares",
                        "Usuario seleccionado"
                    ],

                    range=[
                        "#55555d",
                        "#f0a202",
                        "#e50914"
                    ]
                )
            ),

            size=alt.Size(
                "tipo_punto:N",

                scale=alt.Scale(
                    domain=[
                        "Otros usuarios",
                        "Usuarios similares",
                        "Usuario seleccionado"
                    ],

                    range=[
                        20,
                        100,
                        180
                    ]
                ),

                legend=None
            ),

            opacity=alt.condition(
                alt.datum.tipo_punto == "Otros usuarios",
                alt.value(0.35),
                alt.value(1)
            ),

            tooltip=[
                alt.Tooltip(
                    "userId:O",
                    title="Usuario"
                ),

                alt.Tooltip(
                    "nombre_decada:N",
                    title="Perfil temporal"
                ),

                alt.Tooltip(
                    "Perfil:N",
                    title="Perfil combinado"
                )
            ]
        )
        .properties(
            height=430,
            title="Espacio temporal"
        )
        .interactive()
    )


    # =========================
    # MOSTRAR SCATTERS
    # =========================

    g1, g2 = st.columns(2)

    with g1:

        st.altair_chart(
            grafico_generos,
            width="stretch"
        )

    with g2:

        st.altair_chart(
            grafico_decadas,
            width="stretch"
        )


    st.caption(
        "🔴 Usuario seleccionado · "
        "🟠 cinco usuarios más similares · "
        "⚫ resto de los usuarios"
    )


    # =========================
    # DETALLE DEL USUARIO
    # =========================

    st.divider()

    st.subheader(
        f"Usuario {usuario}"
    )

    detalle = modelos.entrenamiento.loc[usuario]

    fila_etiqueta = datos[
        datos["userId"] == usuario
    ].iloc[0]

    a, b, c = st.columns(3)

    with a:

        st.metric(
            "Valoraciones procesadas",
            int(
                detalle[
                    "cantidad_valoraciones"
                ]
            )
        )

    with b:

        st.metric(
            "Valoraciones originales",
            int(
                detalle[
                    "cantidad_valoraciones_total"
                ]
            )
        )

    with c:

        st.metric(
            "Rating promedio",
            f'{detalle["rating_promedio_general"]:.2f} / 5'
        )


    st.write(
        "**Perfil de género:** "
        + fila_etiqueta["nombre_genero"]
    )

    st.write(
        "**Perfil temporal:** "
        + fila_etiqueta["nombre_decada"]
    )


    # =========================
    # PREFERENCIAS POR GÉNERO
    # =========================

    transformador_generos = (
        modelos
        .generos["preparar"]["features"]
    )

    fila_usuario = (
        modelos
        .entrenamiento
        .loc[[usuario]]
    )

    preferencias = (
        transformador_generos
        .transform(fila_usuario)
        .iloc[0]
    )

    df_generos = pd.DataFrame({
        "Género": [
            x.replace(
                "pref_",
                ""
            )
            for x in preferencias.index
        ],

        "Preferencia": preferencias.values
    })

    df_generos = (
        df_generos
        .sort_values(
            "Preferencia",
            ascending=False
        )
        .head(5)
    )


    grafico_preferencias = (
        alt.Chart(df_generos)
        .mark_bar(
            color="#e50914"
        )
        .encode(

            y=alt.Y(
                "Género:N",
                sort="-x",
                title=None
            ),

            x=alt.X(
                "Preferencia:Q",
                title="Preferencia relativa"
            ),

            tooltip=[
                "Género:N",

                alt.Tooltip(
                    "Preferencia:Q",
                    format=".3f"
                )
            ]
        )
        .properties(
            height=300
        )
    )


    # =========================
    # DISTRIBUCIÓN POR DÉCADA
    # =========================

    columnas_decadas = {
        "prop_antes_1980": "Antes de 1980",
        "prop_1980s": "1980s",
        "prop_1990s": "1990s",
        "prop_2000s": "2000s",
        "prop_2010s": "2010s",
        "prop_2020_mas": "2020+"
    }

    df_decadas = pd.DataFrame({

        "Época": list(
            columnas_decadas.values()
        ),

        "Porcentaje": [
            detalle[columna] * 100
            for columna
            in columnas_decadas.keys()
        ]
    })


    grafico_decadas_usuario = (
        alt.Chart(df_decadas)
        .mark_bar(
            color="#f0a202"
        )
        .encode(

            x=alt.X(
                "Época:N",
                sort=None,
                title=None,
                axis=alt.Axis(
                    labelAngle=-30
                )
            ),

            y=alt.Y(
                "Porcentaje:Q",
                title="% de películas calificadas"
            ),

            tooltip=[
                "Época:N",

                alt.Tooltip(
                    "Porcentaje:Q",
                    format=".1f"
                )
            ]
        )
        .properties(
            height=300
        )
    )


    d1, d2 = st.columns(2)

    with d1:

        st.markdown(
            "### Afinidades por género"
        )

        st.altair_chart(
            grafico_preferencias,
            width="stretch"
        )

    with d2:

        st.markdown(
            "### Películas según época de estreno"
        )

        st.altair_chart(
            grafico_decadas_usuario,
           width="stretch"
        )


    # =========================
    # USUARIOS MÁS SIMILARES
    # =========================

    st.divider()

    st.subheader(
        "Los 5 usuarios más similares"
    )

    tabla_vecinos = vecinos.merge(
        datos[
            [
                "userId",
                "Perfil",
                "rating_promedio_general",
                "cantidad_valoraciones"
            ]
        ],

        on="userId"
    )

    tabla_vecinos["Distancia"] = (
        tabla_vecinos["distancia"]
        .round(3)
    )

    tabla_vecinos = tabla_vecinos.rename(
        columns={
            "userId": "Usuario",
            "rating_promedio_general":
                "Rating promedio",
            "cantidad_valoraciones":
                "Valoraciones"
        }
    )
    tabla_vecinos["Rating promedio"] = (
        tabla_vecinos["Rating promedio"].round(2)
    )

    st.dataframe(
        tabla_vecinos[
            [
                "Usuario",
                "Perfil",
                "Rating promedio",
                "Valoraciones",
                "Distancia"
            ]
        ],

        hide_index=True,
        width="stretch"
    )

    st.caption(
        "Cuanto menor es la distancia, más parecido es el usuario seleccionado. "
        "Para calcularla combinamos con el mismo peso sus patrones de género y de época."
    )


# =========================
# RECOMENDADOR
# =========================

elif seccion == "Recomendador":

    st.markdown(
        '<div class="titulo-rojo">Tu turno · Recomendador</div>',
        unsafe_allow_html=True
    )

    st.title(
        "Descubrí películas a partir de usuarios con gustos parecidos"
    )

    st.write(
        """
        Elegí un usuario. Buscamos personas con patrones de valoración
        similares y recomendamos películas que ellas calificaron bien
        pero que el usuario seleccionado todavía no calificó.
        """
    )

    # =========================
    # USUARIOS DISPONIBLES
    # =========================

    entrenamiento = modelos.entrenamiento

    usuarios_disponibles = sorted(
        entrenamiento.index.tolist()
    )

    usuario = st.selectbox(
        "Elegí un usuario",
        options=usuarios_disponibles,
        format_func=lambda x: f"Usuario {x}"
    )


    # =========================
    # REPRESENTACIONES PCA
    # =========================

    Z_generos = modelos.generos[
        "preparar"
    ].transform(entrenamiento)

    Z_decadas = modelos.decadas[
        "preparar"
    ].transform(entrenamiento)

    ids_modelo = entrenamiento.index.to_numpy()

    posicion = np.where(
        ids_modelo == usuario
    )[0][0]

    usuario_g = Z_generos[posicion]
    usuario_d = Z_decadas[posicion]


    # =========================
    # DISTANCIAS
    # =========================

    dist_g = (
        np.linalg.norm(
            Z_generos - usuario_g,
            axis=1
        )
        / np.sqrt(Z_generos.shape[1])
    )

    dist_d = (
        np.linalg.norm(
            Z_decadas - usuario_d,
            axis=1
        )
        / np.sqrt(Z_decadas.shape[1])
    )

    distancia_total = (
        dist_g + dist_d
    ) / 2


    similitud = pd.DataFrame({
        "userId": ids_modelo,
        "distancia": distancia_total
    })

    similitud = similitud[
        similitud["userId"] != usuario
    ]

    # Probamos progresivamente con más usuarios similares
    # hasta conseguir suficientes películas candidatas.

    cantidades_vecinos = [100, 200, 400, 800]

    vecinos = None
    recomendaciones = pd.DataFrame()

    for cantidad in cantidades_vecinos:

        vecinos_prueba = (
            similitud
            .sort_values("distancia")
            .head(cantidad)
            .copy()
        )

        ids_vecinos_prueba = vecinos_prueba["userId"].tolist()

        # Películas que el usuario ya calificó
        vistas_usuario = set(
            ratings.loc[
                ratings["userId"] == usuario,
                "movieId"
            ]
        )

        # Ratings de los vecinos
        ratings_vecinos_prueba = ratings[
            ratings["userId"].isin(ids_vecinos_prueba)
        ].copy()

        # Quitamos películas ya calificadas por el usuario
        ratings_vecinos_prueba = ratings_vecinos_prueba[
            ~ratings_vecinos_prueba["movieId"].isin(vistas_usuario)
        ]

        # Agregamos distancia
        ratings_vecinos_prueba = ratings_vecinos_prueba.merge(
            vecinos_prueba,
            on="userId",
            how="left"
        )

        # Cuanto más cerca está el usuario, más peso tiene
        ratings_vecinos_prueba["peso"] = (
            1 / (ratings_vecinos_prueba["distancia"] + 0.001)
        )

        ratings_vecinos_prueba["rating_ponderado"] = (
            ratings_vecinos_prueba["rating"]
            * ratings_vecinos_prueba["peso"]
        )

        # Resumen por película
        recomendaciones_prueba = (
            ratings_vecinos_prueba
            .groupby("movieId")
            .agg(
                rating_promedio=("rating", "mean"),
                cantidad_vecinos=("userId", "nunique"),
                suma_ponderada=("rating_ponderado", "sum"),
                suma_pesos=("peso", "sum")
            )
            .reset_index()
        )

        recomendaciones_prueba["score"] = (
            recomendaciones_prueba["suma_ponderada"]
            / recomendaciones_prueba["suma_pesos"]
        )

        # Evidencia mínima
        recomendaciones_prueba = recomendaciones_prueba[
            recomendaciones_prueba["cantidad_vecinos"] >= 2
        ]

        recomendaciones_prueba = recomendaciones_prueba[
            recomendaciones_prueba["rating_promedio"] >= 3.8
        ]

        # Si encontramos al menos 5, nos quedamos con esta cantidad
        if len(recomendaciones_prueba) >= 5:

            vecinos = vecinos_prueba
            recomendaciones = recomendaciones_prueba
            break


    # Si ni con 800 conseguimos 5, usamos igualmente
    # lo mejor que haya encontrado la última búsqueda.
    if vecinos is None:

        vecinos = vecinos_prueba
        recomendaciones = recomendaciones_prueba





    # =========================
    # PERFIL DEL USUARIO
    # =========================

    etiquetas = (
        modelos
        .etiquetas
        .reset_index()
    )

    perfil_usuario = etiquetas[
        etiquetas["userId"] == usuario
    ].iloc[0]

    detalle = entrenamiento.loc[usuario]


    st.subheader(
        f"Usuario {usuario}"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Valoraciones",
            int(
                detalle[
                    "cantidad_valoraciones"
                ]
            )
        )

    with c2:

        st.metric(
            "Rating promedio",
            f'{detalle["rating_promedio_general"]:.2f} / 5'
        )

    with c3:

        st.metric(
            "Usuarios similares usados",
            len(vecinos)
        )


    st.write(
        "**Perfil de género:** "
        + perfil_usuario["nombre_genero"]
    )

    st.write(
        "**Perfil temporal:** "
        + perfil_usuario["nombre_decada"]
    )



    # =========================
    # AGREGAR INFORMACIÓN TMDB
    # =========================

    recomendaciones = (
        recomendaciones
        .merge(
            peliculas,
            on="movieId",
            how="left"
        )
    )


    # =========================
    # ORDEN FINAL
    # =========================

    recomendaciones = (
        recomendaciones
        .sort_values(
            [
                "score",
                "cantidad_vecinos"
            ],
            ascending=[
                False,
                False
            ]
        )
        .head(5)
    )


    # =========================
    # MOSTRAR VECINOS
    # =========================

    st.divider()

    st.subheader(
        "¿En quiénes se basa la recomendación?"
    )

    tabla_vecinos = (
        vecinos
        .head(5)
        .copy()
    )

    tabla_vecinos[
        "distancia"
    ] = (
        tabla_vecinos[
            "distancia"
        ].round(3)
    )

    tabla_vecinos = tabla_vecinos.rename(
        columns={
            "userId": "Usuario",
            "distancia": "Distancia"
        }
    )
    

    st.dataframe(
        tabla_vecinos,
        hide_index=True,
        width="stretch"
    )

    st.caption(
        "Una distancia menor indica un patrón "
        "de valoración más parecido."
    )


    # =========================
    # MOSTRAR RECOMENDACIONES
    # =========================

    st.divider()

    st.subheader(
        "Películas recomendadas"
    )

    if recomendaciones.empty:

        st.warning(
            "No encontramos suficientes películas "
            "con evidencia para este usuario."
        )

    else:

        for i, (_, fila) in enumerate(
            recomendaciones.iterrows(),
            start=1
        ):

            titulo = (
                fila["titulo"]
                if pd.notna(fila["titulo"])
                else f"Película {fila['movieId']}"
            )

            st.markdown(
                f"### {i}. {titulo}"
            )

            if pd.notna(fila["fecha_estreno"]):
                st.caption(
                    f"📅 Estreno: {fila['fecha_estreno']}"
                )

            if pd.notna(fila["genres"]):
                st.caption(
                    f"🎬 {fila['genres']}"
                )

            c1, c2 = st.columns(2)

            with c1:
                st.metric(
                    "Rating entre similares",
                    f"{fila['rating_promedio']:.2f} / 5"
                )

            with c2:
                st.metric(
                    "Usuarios similares",
                    int(fila["cantidad_vecinos"])
                )

            st.write(
                "Te la recomendamos porque usuarios "
                "con un patrón parecido al tuyo "
                "la calificaron bien y vos todavía "
                "no la calificaste."
            )

            st.divider()


    # =========================
    # EXPLICACIÓN
    # =========================

    with st.expander(
        "Cómo funciona este recomendador"
    ):

        st.markdown("""
        **1. Representación del usuario**

        Se utilizan las mismas representaciones PCA de
        géneros y épocas usadas en el modelo.

        **2. Usuarios similares**

        Se calcula la distancia del usuario elegido
        respecto de los demás.

        Géneros y épocas tienen el mismo peso.

        Para obtener recomendaciones, la aplicación comienza
        con los **100 usuarios más cercanos**.

        Si no encuentra al menos 5 películas con evidencia suficiente,
        amplía automáticamente la búsqueda a **200, 400 y hasta 800 usuarios similares**.

        **3. Películas candidatas**

        Se buscan las películas calificadas por esos usuarios.

        Se eliminan todas las películas que el usuario
        seleccionado ya calificó.

        **4. Evidencia mínima**

        Una película debe haber sido calificada por
        al menos **2 usuarios similares**.

        Además, el promedio entre esos usuarios debe ser
        de al menos **3,8 estrellas**.

        **5. Ranking**

        Los usuarios más cercanos tienen mayor peso.

        El score final es un promedio ponderado de sus ratings.

        Por lo tanto, las recomendaciones salen de
        valoraciones reales de MovieLens y no de una lista editorial.
        """)