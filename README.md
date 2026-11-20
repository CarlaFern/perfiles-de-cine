# 🎬 Perfiles de cine

Aplicación web desarrollada como trabajo integrador de **Ciencia de Datos**.

El proyecto analiza patrones de valoración de usuarios de películas para identificar perfiles similares según sus preferencias por género y por época de estreno, y utiliza esas similitudes para generar recomendaciones de películas.

## Pregunta del proyecto

**¿Podemos encontrar usuarios con gustos similares y usar esa similitud para recomendar películas que todavía no calificaron?**

## Fuentes de datos

Se utilizaron dos fuentes principales:

- **MovieLens:** valoraciones realizadas por usuarios sobre películas.
- **TMDB:** metadatos de las películas, como título, géneros y fecha de estreno.

La preparación de los datos se realizó mediante un pipeline construido con **Airflow**.

## Unidad de análisis

La unidad de análisis es el **usuario**.

Para el modelado se conservaron usuarios con al menos **150 valoraciones**, resultando en **3.503 usuarios**.

## Metodología

El análisis se realizó en dos dimensiones independientes:

### Preferencias por género

Se construyeron variables que representan la afinidad relativa de cada usuario hacia diferentes géneros cinematográficos.

Se aplicó:

- estandarización;
- PCA para reducción dimensional;
- K-means con `k = 2`.

### Preferencias temporales

Se utilizaron las proporciones de películas calificadas por cada usuario según su época de estreno.

Se aplicó:

- transformación de proporciones;
- estandarización;
- PCA;
- K-means con `k = 2`.

Los modelos de géneros y épocas se conservaron por separado porque el modelo conjunto reproducía principalmente la estructura temporal.

## La aplicación

Es **una sola página** que se recorre de arriba hacia abajo, con un índice en la barra lateral que salta a cada sección. Sigue la estructura de **copa de martini** de la Unidad 4: primero la historia guiada (el tallo) y al final la parte para explorar (la copa).

Arranca con **el problema** y **el recorrido**: un esquema de los nueve pasos del proyecto, de la tabla de votos a la recomendación, con la decisión de cada uno.

**La historia**

1. **Los datos**, en cuatro pasos: la tabla cruda de votos de MovieLens; a quién miramos (histogramas superpuestos de MovieLens y la muestra, con intervalos ajustables); qué son las películas (cobertura de las 10.000 consultadas a TMDB); y la tabla silver.
2. **La trampa de los pocos ratings**: por qué usamos una media amortiguada y un mínimo de 150 ratings.
3. **Qué entra al modelo**: qué columnas se usan y por qué.
4. **Menos dimensiones**: varianza acumulada, cargas de las dos PCA y un mapa de calor del espacio de géneros con resaltado cruzado por género.
5. **Cuántos grupos**: codo, silhouette y comparación con un modelo conjunto.
6. **Los perfiles**: cruce de los grupos, perfil de cada uno y una nube de usuarios para comprobar que no dependen de la actividad.

**Tu turno**

7. **Explorá los usuarios**: nube de usuarios con selección por leyenda y un recuadro que filtra el perfil de los marcados.
8. **Un usuario y sus películas**: vecinos más parecidos y recomendaciones.

Todos los números de los títulos y textos se calculan a partir de los archivos del pipeline (`historia.py`); cada gráfico tiene debajo un desplegable con la tabla de números que muestra.

## Recomendador

El recomendador utiliza las representaciones obtenidas en los modelos de género y época.

Para un usuario seleccionado:

1. Se calcula su distancia respecto de los demás usuarios.
2. Se combinan las distancias de género y época con el mismo peso.
3. Se buscan los usuarios más cercanos.
4. Se recuperan las películas calificadas por esos usuarios.
5. Se eliminan las películas que el usuario seleccionado ya calificó.
6. Se conservan películas con evidencia suficiente.
7. Las recomendaciones se ordenan mediante un promedio ponderado donde los usuarios más cercanos tienen mayor peso.

La búsqueda comienza con los **100 usuarios más similares** y puede ampliarse automáticamente a **200, 400 u 800** si no existen suficientes películas candidatas.

## Tecnologías utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn
- Altair
- Streamlit
- Airflow

## Estructura principal

```text
├── app.py                       # la página: textos, orden de las secciones e interacción
├── historia.py                  # cálculos y gráficos (Altair) de cada sección
├── modelado.py                  # pipeline de la Entrega 3 (StandardScaler → PCA → K-means)
├── usuarios_generos.csv         # silver del DAG: una fila por usuario (5.000)
├── ratings_selected_users.csv   # ratings del DAG (con el tope de 500) de los 3.503 usuarios modelados
├── tmdb_movies_prepared.csv     # películas consultadas en TMDB por el DAG
├── preparar_origen.py           # resume una vez la salida cruda del DAG para la sección 1
├── origen_*.csv                 # esos resúmenes: votos crudos, población de MovieLens, cobertura
├── .streamlit/config.toml       # tema visual
├── requirements.txt
└── README.md
```

## Ejecución local

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar la aplicación:

```bash
streamlit run app.py
```

Luego acceder desde el navegador a la dirección local indicada por Streamlit, normalmente:

```text
http://localhost:8501
```

## Grupo

**Ciencia de Datos · Grupo 5K09-05**