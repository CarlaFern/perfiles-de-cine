"""Resume los datos crudos del DAG para la primera parte de la app.

El ratings.csv de MovieLens 32M pesa casi 1 GB y no entra al repo. Este script
se corre una sola vez, sobre la salida del DAG movielens_ingest, y guarda tres
tablas chicas que la app lee:

- origen_crudo.csv       las primeras filas de ratings.csv, como vienen
- origen_resumen.csv     los números de cada paso del pipeline (clave, valor)
- origen_poblacion.csv   ratings por usuario de los 200.948 usuarios de MovieLens
- origen_cobertura.csv   qué parte de los ratings de la muestra cubren las N
                         películas más vistas (la curva que justifica pedir
                         10.000 películas a TMDB)

Uso:
    python preparar_origen.py "<carpeta include/output del DAG>"
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEMILLA = 42


def main(salida_dag: Path):
    crudo = salida_dag / "bronze" / "movielens" / "ml-32m" / "ratings.csv"
    inter = salida_dag / "intermediate"

    # 1 · la tabla cruda: una fila por voto (se guardan las primeras filas, tal cual)
    pd.read_csv(crudo, nrows=8).to_csv("origen_crudo.csv", index=False)
    r = pd.read_csv(crudo, usecols=["userId", "movieId", "timestamp"])
    fechas = pd.to_datetime(r["timestamp"], unit="s")
    por_usuario = r["userId"].value_counts().rename_axis("userId").rename("cantidad").sort_index()
    resumen = {
        "ratings_total": len(r),
        "usuarios_total": len(por_usuario),
        "peliculas_total": r["movieId"].nunique(),
        "usuarios_elegibles": int((por_usuario >= 100).sum()),
        "anio_primer_voto": int(fechas.min().year),
        "anio_ultimo_voto": int(fechas.max().year),
    }
    del r
    por_usuario.reset_index().to_csv("origen_poblacion.csv", index=False)

    # 2 · la muestra: 5.000 usuarios al azar, con tope de 500 ratings
    muestra = pd.read_csv(inter / "ratings_selected_users.csv", usecols=["userId", "movieId"])
    historia = pd.read_csv(inter / "user_history_full.csv")
    resumen.update({
        "usuarios_muestra": muestra["userId"].nunique(),
        "ratings_muestra_completo": int(historia["cantidad_valoraciones_total"].sum()),
        "ratings_muestra": len(muestra),
        "peliculas_muestra": muestra["movieId"].nunique(),
    })

    # 3 · TMDB: las películas más vistas de la muestra, y cuánto cubren
    mapa = pd.read_csv(inter / "movie_tmdb_map.csv")
    tmdb = pd.read_csv(inter / "tmdb_movies_prepared.csv", usecols=["movieId", "genres"])
    con_genero = set(tmdb.loc[tmdb["genres"].fillna("") != "", "movieId"])
    resumen.update({
        "peliculas_consultadas": len(mapa),
        "peliculas_tmdb": len(tmdb),
        "peliculas_con_genero": len(con_genero),
        "ratings_con_genero": int(muestra["movieId"].isin(con_genero).sum()),
    })
    conteo = muestra["movieId"].value_counts().to_numpy()
    acumulado = np.cumsum(conteo) / conteo.sum()
    # unos 300 puntos espaciados en escala log alcanzan para dibujar la curva
    rangos = np.unique(np.round(np.logspace(0, np.log10(len(conteo)), 300)).astype(int))
    rangos = np.union1d(rangos, [len(mapa)])
    pd.DataFrame({"peliculas": rangos, "cubierto": acumulado[rangos - 1]}).to_csv(
        "origen_cobertura.csv", index=False)

    pd.Series(resumen, name="valor").rename_axis("clave").reset_index().to_csv(
        "origen_resumen.csv", index=False)
    for k, v in resumen.items():
        print(f"{k:28s} {v:>12,}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
