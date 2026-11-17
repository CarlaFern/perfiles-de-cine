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

## Exploración interactiva

La aplicación permite:

- filtrar usuarios por perfil;
- seleccionar un usuario;
- visualizar su posición en los espacios de género y época;
- observar sus afinidades;
- consultar su distribución de películas según época de estreno;
- encontrar los usuarios más similares.

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
entrega4_final/
│
├── app.py
├── modelado.py
├── usuarios_generos.csv
├── ratings_selected_users.csv
├── tmdb_movies_prepared.csv
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

## Navegación de la aplicación

La aplicación sigue una estructura narrativa dividida en dos partes.

### La historia

- El proyecto
- Los datos
- Lo que encontramos
- El modelo

### Tu turno

- Explorá
- Recomendador

## Grupo

**Ciencia de Datos · Grupo 5K09-05**