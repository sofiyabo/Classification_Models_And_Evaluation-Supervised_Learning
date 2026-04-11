import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath("../src"))
sys.path.append(os.path.abspath("../data"))
import preprocessing as prep
import utils as utls


df_raw = pd.read_csv("../data/raw/rendimiento_estudiantes_dev.csv") #esto funciona solo si me paro en cd src,
# igual al cambiar a notebook no deberia haber problema.
#print(df_raw.sample(15))

#print(df_raw.describe())
#print(df_raw.isnull().sum())

#Total de datos: 5058 
"""
Nans:
horas_estudio            275
asistencia                 0
nota_previa              528
horas_sueno              273
participacion            180
horas_extracurricular      0
acceso_internet            0
distancia_escuela_km       0
nivel_socioeconomico      92
tamano_clase               0
escuela                    0
semestre                   0
rendimiento                0

"""

ord_rend = {
    "Insuficiente": 0,
    "Regular": 1,
    "Bueno": 2, 
    "Excelente": 3
}

ord_sem = {
    "2022-1": 0,
    "2022-2": 1,
    "2023-1": 2,
    "2023-2": 3,
    "2024-1": 4,
    "2024-2": 5
}

df = df_raw.copy()
df = prep.ordinal_encoding(df, ord_rend, "rendimiento")
df = prep.ordinal_encoding(df, ord_sem, "semestre") # Me importa el orden cronologico, se podrian hacer solo dos columnas de anio y semestre

"""
horas estudio: contemplando que tiene 275 Nans, se computa con la mediana que se ve es aprox 5 hs. Se ve una ligera distribucion asimetrica a la derecha
asistencia: ver outliers inferiores
nota_previa: parece que hay gran parte de las notas con notas 0-100 en vez de 0-10. Mezcla de escalas. 528 Nans, imputar?
horas sueno: distribucion razonable. Imputar Nans con mediana aprox 7.5
participacion: 180 Nans, imputar con mediana?
horas extracurricular: cola derecha muy larga y muchos outliers. Aplicar log para los outliers.
dist: muchos outliers, aplicar log.
socioeco: bien

scatters:
Los pares más interesantes serían:

horas_estudio vs nota_previa
asistencia vs nota_previa
horas_estudio vs asistencia
nivel_socioeconomico vs nota_previa

"""
df = prep.one_hot_encoding(df, "escuela")
df = prep.notes_scale(df)
