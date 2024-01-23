import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import math
from moviepy.editor import ImageSequenceClip
import re

def main(**datos):
    """
    Funcion principal que conectara a las demas funciones
    """
    # Ruta de la carpeta que deseas eliminar
    carpeta = 'imagenes'
    nombre_video = 'video_generaciones.mp4'

    # Comprobar si la carpeta existe
    if os.path.exists(carpeta):
        # Eliminar la carpeta y todo su contenido
        shutil.rmtree(carpeta)
        
    # Verificar si el video ya existe y eliminarlo en caso afirmativo
    if os.path.exists(nombre_video):
        os.remove(nombre_video)

    os.makedirs(carpeta, exist_ok=True)
    data_molde = {
        'ID': [],
        'Individuo': [],
        'i': [],
        'x': [],
        'f(x)': []
    }
    estadisticas_generaciones = pd.DataFrame(columns=['Generacion', 'Mejor', 'Peor', 'Promedio'])
    
    p0 = int(datos.get("P0", None))
    pmax = int(datos.get("Pmax", None))
    pmut = float(datos.get("Pmut ind", None))
    p_mut_gen = float(datos.get("Pmut gen", None))
    opt = datos.get("opt", None)
    p = float(datos.get("Res Des", None))
    a = float(datos.get("a", None))
    b = float(datos.get("b", None))
    r = b - a
    num_gen = int(datos.get("#iteracion", None))
    
    num_pasos = math.ceil((r / p))
    num_saltos = num_pasos + 1
    
    bits = calcular_bits(num_saltos)
    delta_x = (r / (2**bits - 1))
    
    cadenas_iniciales = generar_cadenas_aleatorias(p0, bits)
    data = data_molde.copy()
    df = pd.DataFrame(data)
    df = generar_poblacion_inicial(cadenas_iniciales, a, delta_x, df)
    generations = []
    porcentaje = 10
    for index in range(num_gen):
        #print(df)
        mejores_individuos = seleccionar_mejores_individuos(df, porcentaje, opt)
        resto_individuos = obtener_resto_individuos(df, mejores_individuos)
        #print(mejores_individuos)
        #print(resto_individuos)
        parejas = generar_parejas(mejores_individuos, resto_individuos)
        #print(parejas)
        descendencia = cruzas(parejas, bits)
        #print(descendencia)
        descendencia_mutada = mutaciones(descendencia, pmut, p_mut_gen)
        #print(descendencia_mutada)
        df = combinar_poblacion(descendencia_mutada, df, a, delta_x)
        df = ordenar_dataframe(df, 'f(x)', opt)
        #print(df)
        generations.append(df)
        #print(df)
        estadisticas_generaciones = añadir_estadisticas_generacion(df, estadisticas_generaciones, index+1, opt)
        #print(estadisticas_generaciones)
        df = eliminar_duplicados(df)
        df = poda(df, pmax, opt)
    graficar_generaciones(generations, carpeta, a, b)
    crear_video(carpeta, nombre_video, 2)
    graficar_estadisticas(estadisticas_generaciones)
    

        
def calcular_bits(num_saltos):
    return math.ceil(math.log2(num_saltos))

def generar_cadenas_aleatorias(num_iteraciones, num_bits):
    """
    Genera las cadenas aleatorias de bits dependiendo de la poblacion inicial
    """
    cadenas = []
    for _ in range(num_iteraciones):
        # Generar una cadena de bits aleatoria
        cadena_bits = ''.join(random.choice('01') for _ in range(num_bits))
        cadenas.append(cadena_bits)
    return cadenas

def binario_a_decimal(binario):
    """
    Funcion para transformar un binario a decimal
    """
    return int(binario, 2)

def generar_poblacion_inicial(cadenas_iniciales, a, delta_x, df):
    id = 1
    for cadena_binario in cadenas_iniciales:
        decimal = binario_a_decimal(cadena_binario)
        x = round(calcular_x(a, decimal, delta_x), 4)
        nueva_fila = pd.DataFrame({
        'ID': [id],
        'Individuo': [cadena_binario],
        'i': [decimal],
        'x': [x], 
        'f(x)': [calcular_fx(x)]
        })

        # Usar pd.concat para agregar la nueva fila
        df = pd.concat([df, nueva_fila], ignore_index=True)
        id += 1
    return df

def calcular_x(a, i, delta_x):
    """
    Regresa el valor de x dependiendo de la posicion de i y usando delta_x
    """
    return a + i * delta_x  # Corregir aquí, i es un índice, no una función

def calcular_fx(x):
    """
    Regresa el valor de f(x) usando la formula
    NOTA: Aca se tiene que modificar la formula
    """
    return np.log(1 + np.abs(x)**3) * np.cos(x) - np.sin(x) * np.log(2 + np.abs(x)**5)

def seleccionar_mejores_individuos(df, porcentaje_mejores, modo):
    """
    Selecciona el top porcentaje de individuos del DataFrame basándose en la columna 'f(x)'.

    :param df: DataFrame que contiene los datos de la población.
    :param porcentaje_mejores: Porcentaje de los mejores individuos a seleccionar.
    :param modo: Modo de optimización, 'MIN' para minimización o 'MAX' para maximización.
    :return: DataFrame con los mejores individuos seleccionados.
    """
    num_seleccionados = max(1, int(len(df) * porcentaje_mejores / 100))
    ascending = True if modo == 'MIN' else False
    df_ordenado = df.sort_values(by='f(x)', ascending=ascending)
    mejores = df_ordenado.head(num_seleccionados)
    return mejores

def obtener_resto_individuos(df, mejores_individuos):
    """
    Obtiene los individuos del DataFrame que no están en el conjunto de los mejores individuos.

    :param df: DataFrame que contiene los datos de la población.
    :param mejores_individuos: DataFrame de los mejores individuos seleccionados previamente.
    :return: DataFrame con los individuos restantes.
    """
    # Realizar la fusión usando solo la columna 'ID'
    resto_individuos = pd.merge(df[['ID']], mejores_individuos[['ID']], 
                                on='ID', how='outer', indicator=True)
    
    # Filtrar para obtener solo los individuos que no están en los mejores
    resto_ids = resto_individuos[resto_individuos['_merge'] == 'left_only']['ID']
    
    # Seleccionar las filas del DataFrame original que corresponden a los IDs restantes
    resto_individuos = df[df['ID'].isin(resto_ids)]
    return resto_individuos

def generar_parejas(mejores_individuos, resto_individuos):
    parejas = []
    for _, mejor in mejores_individuos.iterrows():
        for _, resto in resto_individuos.iterrows():
            pareja = {
                'Padre 1': mejor['Individuo'],
                'IDPadre 1': mejor['ID'],
                'Padre 2': resto['Individuo'],
                'IDPadre 2': resto['ID']
            }
            parejas.append(pareja)
    parejas_df = pd.DataFrame(parejas)
    return parejas_df

def cruzas(df_parejas, num_bits):
    # Añadir nuevas columnas para los hijos
    df_parejas['Hijo 1'] = ""
    df_parejas['Hijo 2'] = ""

    for index, pareja in df_parejas.iterrows():
        padre1 = pareja['Padre 1']
        padre2 = pareja['Padre 2']

        # Asegurarse de que el punto de cruza sea válido
        punto_cruza = random.randint(1, num_bits - 1)
        
        # Realizar la cruza
        hijo1 = padre1[:punto_cruza] + padre2[punto_cruza:]
        hijo2 = padre2[:punto_cruza] + padre1[punto_cruza:]

        # Actualizar el DataFrame con los hijos resultantes
        df_parejas.at[index, 'Hijo 1'] = hijo1
        df_parejas.at[index, 'Hijo 2'] = hijo2
        df_parejas.at[index, 'Punto Cruza'] = punto_cruza

    return df_parejas

def mutaciones(df, probabilidad_mutacion_individuo, probabilidad_mutacion_gen):
    for index, fila in df.iterrows():
        # Decidir si el individuo muta
        numero_aleatorio = (random.uniform(0,100)/100)
        if numero_aleatorio <= probabilidad_mutacion_individuo:
            individuo_mutado = ""
            for gen in fila['Hijo 1']:
                # Decidir si el gen muta
                if random.random() <= probabilidad_mutacion_gen:
                    # Cambiar de '0' a '1' o de '1' a '0'
                    gen_mutado = '1' if gen == '0' else '0'
                else:
                    gen_mutado = gen
                individuo_mutado += gen_mutado
            
            # Actualizar el DataFrame con el individuo mutado
            df.at[index, 'Hijo 1'] = individuo_mutado
            
        numero_aleatorio = (random.uniform(0,100)/100)
        if numero_aleatorio <= probabilidad_mutacion_individuo:
            individuo_mutado = ""
            for gen in fila['Hijo 2']:
                # Decidir si el gen muta
                if random.random() <= probabilidad_mutacion_gen:
                    # Cambiar de '0' a '1' o de '1' a '0'
                    gen_mutado = '1' if gen == '0' else '0'
                else:
                    gen_mutado = gen
                individuo_mutado += gen_mutado
            
            # Actualizar el DataFrame con el individuo mutado
            df.at[index, 'Hijo 2'] = individuo_mutado
    return df

def combinar_poblacion(df_descendencia, df_original, a, delta_x):
    for _, fila in df_descendencia.iterrows():
        i = binario_a_decimal(fila['Hijo 1'])
        x = calcular_x(a, i, delta_x)
        nuevo_registro_1 = pd.DataFrame([{
            'ID': len(df_original) + 1,
            'Individuo': fila['Hijo 1'],
            'i': i,
            'x': x,
            'f(x)': calcular_fx(x)
        }])

        i = binario_a_decimal(fila['Hijo 2'])
        x = calcular_x(a, i, delta_x)
        nuevo_registro_2 = pd.DataFrame([{
            'ID': len(df_original) + 2,
            'Individuo': fila['Hijo 2'],
            'i': i,
            'x': x,
            'f(x)': calcular_fx(x)
        }])

        df_original = pd.concat([df_original, nuevo_registro_1, nuevo_registro_2], ignore_index=True)
    return df_original

def obtener_estadisticas(df, modo):
    if modo == 'MAX':
        mejor = df['f(x)'].max()  # En maximización, el mejor es el máximo
        peor = df['f(x)'].min()   # En maximización, el peor es el mínimo
    else:  # Modo por defecto es 'MIN'
        mejor = df['f(x)'].min()  # En minimización, el mejor es el mínimo
        peor = df['f(x)'].max()   # En minimización, el peor es el máximo

    promedio = df['f(x)'].mean()

    return mejor, peor, promedio

def añadir_estadisticas_generacion(df, estadisticas_generaciones, generacion, modo='MIN'):
    if modo == 'MAX':
        mejor = df['f(x)'].max()
        peor = df['f(x)'].min()
    else:  # 'MIN'
        mejor = df['f(x)'].min()
        peor = df['f(x)'].max()

    promedio = df['f(x)'].mean()
    # Crear un nuevo DataFrame para el registro
    nuevo_registro_df = pd.DataFrame({
        'Generacion': [generacion],
        'Mejor': [mejor],
        'Peor': [peor],
        'Promedio': [promedio]
    })
    # Usar pd.concat para añadir el nuevo registro
    estadisticas_generaciones = pd.concat([estadisticas_generaciones, nuevo_registro_df], ignore_index=True)
    return estadisticas_generaciones

def eliminar_duplicados(df, columna='Individuo'):
    df_sin_duplicados = df.drop_duplicates(subset=[columna], keep='first')
    return df_sin_duplicados

def poda(df, tamaño_maximo, modo):
    # Ordenar el DataFrame: ascendente para MIN, descendente para MAX
    if modo == 'MIN':
        orden_ascendente = True
    else:
        orden_ascendente = False

    df_ordenado = df.sort_values(by='f(x)', ascending=orden_ascendente)

    # Mantener solo los primeros N registros
    df_reducido = df_ordenado.head(tamaño_maximo)

    return df_reducido

def graficar_estadisticas(df_estadisticas):
    plt.figure(figsize=(10, 6))

    # Graficar cada estadística
    plt.plot(df_estadisticas['Generacion'], df_estadisticas['Mejor'], color='green', label='Mejor')
    plt.plot(df_estadisticas['Generacion'], df_estadisticas['Peor'], color='red', label='Peor')
    plt.plot(df_estadisticas['Generacion'], df_estadisticas['Promedio'], color='blue', label='Promedio')

    # Añadir título y etiquetas
    plt.title('Evolución de la Población')
    plt.xlabel('Generación')
    plt.ylabel('Valor')

    # Añadir leyenda
    plt.legend()

    # Mostrar la gráfica
    plt.show()

def graficar_generaciones(arreglo_dataframes, carpeta, a, b):
    # Generar puntos x dentro del intervalo [a, b] para la línea de la función
    x_line = np.linspace(a, b, 400)  # 400 puntos para una línea suave
    y_line = calcular_fx(x_line)

    for i, df in enumerate(arreglo_dataframes):
        # Asumiendo que cada DataFrame tiene columnas 'x' y 'f(x)' y están ordenados
        valores_x = df['x']
        valores_fx = df['f(x)']
        mejor_x = df.iloc[0]['x']
        mejor_fx = df.iloc[0]['f(x)']
        peor_x = df.iloc[-1]['x']
        peor_fx = df.iloc[-1]['f(x)']

        # Crear la gráfica
        plt.figure(figsize=(10, 6))
        plt.scatter(valores_x[1:-1], valores_fx[1:-1], color='blue', label='Individuos')
        plt.scatter([mejor_x], [mejor_fx], color='green', label='Mejor', zorder=5)
        plt.scatter([peor_x], [peor_fx], color='red', label='Peor', zorder=5)
        
        plt.plot(x_line, y_line, 'k--')
        
        plt.xlabel('Valor de x')
        plt.ylabel('f(x)')
        plt.title(f'Generación {i + 1}')
        plt.legend()
        plt.xlim(a, b)
        # Guardar la gráfica en la carpeta 'images'
        nombre_archivo = f'generacion_{i + 1}.png'
        ruta_archivo = os.path.join(carpeta, nombre_archivo)
        plt.savefig(ruta_archivo)
        plt.close()

def ordenar_dataframe(df, columna, modo):
    """
    Ordena el DataFrame de mejor a peor basado en la columna especificada.

    :param df: DataFrame a ordenar.
    :param columna: Nombre de la columna basada en la cual se realizará el ordenamiento.
    :param modo: 'MIN' para minimización (valores más bajos son mejores), 
                 'MAX' para maximización (valores más altos son mejores).
    :return: DataFrame ordenado.
    """
    if modo == 'MIN':
        df_ordenado = df.sort_values(by=columna, ascending=True)
    elif modo == 'MAX':
        df_ordenado = df.sort_values(by=columna, ascending=False)
    else:
        raise ValueError("Modo no reconocido. Usa 'MIN' o 'MAX'.")

    return df_ordenado

def ordenar_archivos_alphanumericamente(archivo):
    numeros = re.findall(r'\d+', archivo)
    return int(numeros[0]) if numeros else 0

def crear_video(carpeta_imagenes, nombre_video, fps=2):
    rutas_imagenes = sorted(
        [os.path.join(carpeta_imagenes, img) for img in os.listdir(carpeta_imagenes) if img.endswith(".png")],
        key=ordenar_archivos_alphanumericamente
    )

    # Verificar si la lista de imágenes está vacía
    if not rutas_imagenes:
        print("No se encontraron imágenes en la carpeta especificada.")
        return

    # Crear un clip de video a partir de las imágenes
    clip = ImageSequenceClip(rutas_imagenes, fps=fps)

    # Guardar el video en el archivo especificado
    clip.write_videofile(nombre_video)