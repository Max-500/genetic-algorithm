import tkinter as tk
from tkinter import ttk
import main

# Crear la ventana
ventana = tk.Tk()

# Configurar el tamaño de la ventana
ventana.geometry("400x500")
ventana.title("Ventana con Entradas")

# Crear etiquetas y entradas
campos = ["P0", "Pmax", "Res Des", "a", "b", "Pmut ind", "Pmut gen", "#iteracion", "opt"]
etiquetas = [tk.Label(ventana, text=f"{campo}:") for campo in campos]
entradas = [tk.Entry(ventana) for _ in range(len(campos) - 1)]  # Menos una entrada

# Ubicar etiquetas y entradas en la ventana
for i, campo in enumerate(campos[:-1]):  # Excluir la última entrada
    etiquetas[i].grid(row=i, column=0, padx=10, pady=10)
    entradas[i].grid(row=i, column=1, padx=10, pady=10)

# Agregar un Combobox (menú desplegable) para la última entrada
opciones_opt = ["MAX", "MIN"]
var_opt = tk.StringVar()
var_opt.set(opciones_opt[0])  # Establecer el valor predeterminado
combo_opt = ttk.Combobox(ventana, textvariable=var_opt, values=opciones_opt)
combo_opt.grid(row=len(campos) - 1, column=1, padx=10, pady=10)

# Agregar un Label al lado del menú desplegable
label_opt = tk.Label(ventana, text="opt:")
label_opt.grid(row=len(campos) - 1, column=0, padx=10, pady=10)

def obtener_datos():
    datos = {}
    for i, campo in enumerate(campos[:-1]):
        datos[campo] = entradas[i].get()

    # Agregar el dato del menú desplegable
    datos["opt"] = var_opt.get()

    main.main(**datos)  # Usar la sintaxis **datos para pasar los datos como argumentos nombrados

# Botón para obtener datos
boton_obtener = tk.Button(ventana, text="Obtener Datos", command=obtener_datos)
boton_obtener.grid(row=len(campos), column=0, columnspan=2, pady=10)

# Mostrar la ventana
ventana.mainloop()