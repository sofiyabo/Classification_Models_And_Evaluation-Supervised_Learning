import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("../src"))
sys.path.append(os.path.abspath("../data"))
import preprocessing as prep

df_raw = pd.read_csv("../data/raw/rendimiento_estudiantes_dev.csv") #esto funciona solo si me paro en cd src,
# igual al cambiar a notebook no deberia haber problema.
#print(df_raw.sample(15))

#print(df_raw.describe())
#print(df_raw.isnull().sum())

#Total de datos: 5058 
"""
Nulls:
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

mapeo = {
    "Insuficiente": 0,
    "Regular": 1,
    "Bueno": 2, 
    "Excelente": 3
}
df = df_raw.copy()
df = prep.ordinal_encoding(df, mapeo, "rendimiento")
df = prep.one_hot_encoding(df, "escuela")
print(df_raw.sample(15))
df.hist(bins=50)
plt.show()

