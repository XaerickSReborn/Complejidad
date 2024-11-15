import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from typing import Tuple, Dict, List

G = None
aeropuertos_df = None
distancias_minimas = None
frecuencia_rutas = {}


# Cargar los datos de rutas y aeropuertos
def cargar_datos() -> Tuple[pd.DataFrame, pd.DataFrame]:
    url_rutas = 'https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat'
    url_aeropuertos = 'https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat'
    
    rutas_cols = ['airline', 'airline_id', 'source_airport', 'source_airport_id', 
                  'destination_airport', 'destination_airport_id', 'codeshare', 'stops', 'equipment']
    aeropuertos_cols = ['airport_id', 'name', 'city', 'country', 'IATA', 'ICAO', 
                        'lat', 'long', 'alt', 'timezone', 'DST', 'tz', 'type', 'source']
    
    rutas_df = pd.read_csv(url_rutas, header=None, names=rutas_cols)
    aeropuertos_df = pd.read_csv(url_aeropuertos, header=None, names=aeropuertos_cols)

    paises_norteamerica = ['United States', 'Canada', 'Mexico']
    aeropuertos_na = aeropuertos_df[aeropuertos_df['country'].isin(paises_norteamerica)]
    aeropuertos_na_ids = set(aeropuertos_na['airport_id'].astype(str))
    rutas_na = rutas_df[
        (rutas_df['source_airport_id'].astype(str).isin(aeropuertos_na_ids)) &
        (rutas_df['destination_airport_id'].astype(str).isin(aeropuertos_na_ids))
    ]
    return rutas_na, aeropuertos_na

# Crear el grafo de rutas
def crear_grafo(rutas_df: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    for _, row in rutas_df.iterrows():
        source, destination = str(row['source_airport_id']), str(row['destination_airport_id'])
        if source != '\\N' and destination != '\\N':
            G.add_edge(source, destination, weight=1)
    return G

# Calcular rutas desde el aeropuerto de origen y visualizar
def calcular_y_visualizar_rutas():
    global G, aeropuertos_df

    origen = aeropuerto_origen.get()
    origen_id = aeropuertos_df[aeropuertos_df['name'] == origen]['airport_id']
    
    if origen_id.empty:
        resultado.set("Aeropuerto de origen no encontrado.")
        return
    
    origen_id = str(origen_id.values[0])
    rutas_desde_origen = list(G.successors(origen_id))
    
    if rutas_desde_origen:
        resultado.set(f"Rutas desde {origen}: {len(rutas_desde_origen)}")
        visualizar_rutas_conectadas(origen_id, rutas_desde_origen)
        
        # Actualizar la frecuencia de cada ruta recorrida
        for destino_id in rutas_desde_origen:
            actualizar_frecuencia_ruta(origen_id, destino_id)
    else:
        resultado.set(f"No se encontraron rutas desde {origen}.")

# Visualizar rutas desde el aeropuerto de origen
def visualizar_rutas_conectadas(origen_id, rutas_desde_origen):

    # Muestra las rutas conectadas desde un aeropuerto de origen
    plt.figure(figsize=(15, 10))
    m = Basemap(llcrnrlon=-130, llcrnrlat=20, urcrnrlon=-60, urcrnrlat=55, projection='lcc', lat_1=33, lat_2=45, lon_0=-95)
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    origen = aeropuertos_df[aeropuertos_df['airport_id'] == int(origen_id)]
    origen_lon, origen_lat = float(origen['long'].iloc[0]), float(origen['lat'].iloc[0])
    x_origen, y_origen = m(origen_lon, origen_lat)
    m.plot(x_origen, y_origen, 'go', markersize=12, label="Origen")
    
    for destino_id in rutas_desde_origen:
        destino = aeropuertos_df[aeropuertos_df['airport_id'] == int(destino_id)]
        if not destino.empty:
            dest_lon, dest_lat = float(destino['long'].iloc[0]), float(destino['lat'].iloc[0])
            x_dest, y_dest = m(dest_lon, dest_lat)
            m.plot(x_dest, y_dest, 'bo', markersize=8)
            m.drawgreatcircle(origen_lon, origen_lat, dest_lon, dest_lat, linewidth=1, color='b')

    plt.title(f"Rutas desde {origen['name'].values[0]}")
    plt.legend()
    plt.show()

# Calcular la ruta rapida entre el aeropuerto de origen y destino
def calcular_ruta_rapida() -> None:
    global G, aeropuertos_df, distancias_minimas
    
    origen = aeropuerto_origen.get()
    destino = aeropuerto_destino.get()
    
    origen_id = aeropuertos_df[aeropuertos_df['name'] == origen]['airport_id']
    destino_id = aeropuertos_df[aeropuertos_df['name'] == destino]['airport_id']
    
    if origen_id.empty or destino_id.empty:
        resultado.set("Aeropuerto de origen o destino no encontrado.")
        return
    
    origen_id = str(origen_id.values[0])
    destino_id = str(destino_id.values[0])
    
    try:
        # Calcula la ruta más rapida entre los aeropuertos de origen y destino
        ruta_rapida = nx.shortest_path(G, source=origen_id, target=destino_id, weight='weight')
        resultado.set(f"Ruta rapida encontrada: {ruta_rapida}")
        
        visualizar_ruta(ruta_rapida)
    except nx.NetworkXNoPath:
        resultado.set("No existe una ruta entre los aeropuertos seleccionados.")

# Visualizar una ruta usando Basemap
def visualizar_ruta(path: list) -> None:
    plt.figure(figsize=(15, 10))
    m = Basemap(llcrnrlon=-130, llcrnrlat=20, urcrnrlon=-60, urcrnrlat=55,
                projection='lcc', lat_1=33, lat_2=45, lon_0=-95)
    
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    
    for i in range(len(path) - 1):
        start = aeropuertos_df[aeropuertos_df['airport_id'] == int(path[i])]
        end = aeropuertos_df[aeropuertos_df['airport_id'] == int(path[i+1])]
        
        if not start.empty and not end.empty:
            start_lon = float(start['long'].iloc[0])
            start_lat = float(start['lat'].iloc[0])
            end_lon = float(end['long'].iloc[0])
            end_lat = float(end['lat'].iloc[0])
            
            x1, y1 = m(start_lon, start_lat)
            x2, y2 = m(end_lon, end_lat)
            
            m.drawgreatcircle(start_lon, start_lat, end_lon, end_lat, linewidth=2, color='r')
            
            m.plot(x1, y1, 'bo', markersize=10)
            m.plot(x2, y2, 'bo', markersize=10)
    
    plt.title("Ruta Aerea")
    plt.show()

def calcular_flujo_maximo():

    # Calcula el flujo maximo entre el aeropuerto de origen y destino.
    global G, aeropuertos_df
    origen, destino = aeropuerto_origen.get(), aeropuerto_destino.get()
    origen_id, destino_id = aeropuertos_df[aeropuertos_df['name'] == origen]['airport_id'], aeropuertos_df[aeropuertos_df['name'] == destino]['airport_id']
    
    if origen_id.empty or destino_id.empty:
        resultado.set("Aeropuerto de origen o destino no encontrado.")
        return
    origen_id, destino_id = str(origen_id.values[0]), str(destino_id.values[0])
    
    try:
        G_flow = G.copy()
        for u, v in G_flow.edges():
            G_flow[u][v]['capacity'] = 1
        flow_value, flow_dict = nx.maximum_flow(G_flow, origen_id, destino_id)
        flow_edges = [(u, v) for u in flow_dict for v, flow in flow_dict[u].items() if flow > 0]
        resultado.set(f"Flujo maximo encontrado: {flow_value} rutas independientes")
        visualizar_flujo_maximo(flow_edges, origen_id, destino_id)
        
    except nx.NetworkXError as e:
        resultado.set(f"Error al calcular el flujo maximo: {str(e)}")

# Visualiza el grafo con las rutas del flujo maximo
def visualizar_flujo_maximo(flow_edges: List[Tuple[str, str]], origen_id: str, destino_id: str):
    
    plt.figure(figsize=(15, 10))
    m = Basemap(llcrnrlon=-130, llcrnrlat=20, urcrnrlon=-60, urcrnrlat=55, projection='lcc', lat_1=33, lat_2=45, lon_0=-95)
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    
    for start_id, end_id in flow_edges:
        start, end = aeropuertos_df[aeropuertos_df['airport_id'] == int(start_id)], aeropuertos_df[aeropuertos_df['airport_id'] == int(end_id)]
        if not start.empty and not end.empty:
            start_lon, start_lat = float(start['long'].iloc[0]), float(start['lat'].iloc[0])
            end_lon, end_lat = float(end['long'].iloc[0]), float(end['lat'].iloc[0])
            
            
            x1, y1 = m(start_lon, start_lat)
            x2, y2 = m(end_lon, end_lat)
            
            m.plot([x1, x2], [y1, y2], marker='o', color='purple', linewidth=2, alpha=0.6)
            
            # Marcar los puntos de origen y destino
            m.plot(x1, y1, 'bo', markersize=10)
            m.plot(x2, y2, 'bo', markersize=10)
    
    plt.title("Rutas de Flujo Maximo")
    plt.legend()
    plt.show()

# Actualiza la frecuencia de la ruta entre dos aeropuertos
def actualizar_frecuencia_ruta(origen_id: str, destino_id: str):
    
    if (origen_id, destino_id) not in frecuencia_rutas:
        frecuencia_rutas[(origen_id, destino_id)] = 0
    frecuencia_rutas[(origen_id, destino_id)] += 1

def crear_interfaz():
    global root, aeropuerto_origen, aeropuerto_destino, resultado, text_area
    
    # Frame principal con scroll
    main_frame = ttk.Frame(root, padding="10")
    main_frame.pack(fill=tk.BOTH, expand=True)
    control_frame = ttk.Frame(main_frame)
    control_frame.pack(fill=tk.X, pady=(0, 10))
    
    ttk.Label(control_frame, text="Aeropuerto de origen:").grid(row=0, column=0, sticky=tk.W, pady=5)
    aeropuerto_origen = ttk.Combobox(control_frame, values=aeropuertos_df['name'].tolist(), width=50)
    aeropuerto_origen.grid(row=0, column=1, pady=5, padx=5)
    
    ttk.Label(control_frame, text="Aeropuerto de destino:").grid(row=1, column=0, sticky=tk.W, pady=5)
    aeropuerto_destino = ttk.Combobox(control_frame, values=aeropuertos_df['name'].tolist(), width=50)
    aeropuerto_destino.grid(row=1, column=1, pady=5, padx=5)
    
    # Frame para botones
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(fill=tk.X, pady=(0, 10))
    
    # Botones
    boton_calcular_rutas = ttk.Button(button_frame, text="Calcular y Visualizar Rutas", 
                                     command=calcular_y_visualizar_rutas)
    boton_calcular_rutas.pack(side=tk.LEFT, padx=5)
    
    boton_calcular_ruta_rapida = ttk.Button(button_frame, text="Calcular Ruta Rápida", 
                                           command=calcular_ruta_rapida)
    boton_calcular_ruta_rapida.pack(side=tk.LEFT, padx=5)
    
    boton_flujo_maximo = ttk.Button(button_frame, text="Calcular Flujo Máximo", 
                                   command=calcular_flujo_maximo)
    boton_flujo_maximo.pack(side=tk.LEFT, padx=5)
    
    boton_frecuencia_rutas = ttk.Button(button_frame, text="Mostrar Frecuencia de Rutas",
                                       command=mostrar_frecuencia_rutas)
    boton_frecuencia_rutas.pack(side=tk.LEFT, padx=5)
    

    # Scroll
    result_frame = ttk.Frame(main_frame)
    result_frame.pack(fill=tk.BOTH, expand=True)
    
    scrollbar = ttk.Scrollbar(result_frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    text_area = tk.Text(result_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set, height=10)
    text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    scrollbar.config(command=text_area.yview)
    
    resultado = tk.StringVar()
    resultado.trace('w', lambda *args: actualizar_resultado())

def actualizar_resultado():
    text_area.delete(1.0, tk.END)
    text_area.insert(tk.END, resultado.get())

# Modificar las funciones que muestran resultados para usar el area de texto
def mostrar_frecuencia_rutas():
    mensaje = "Frecuencia de Rutas:\n\n"
    for (origen_id, destino_id), frecuencia in frecuencia_rutas.items():
        origen = aeropuertos_df[aeropuertos_df['airport_id'] == int(origen_id)]['name'].values[0]
        destino = aeropuertos_df[aeropuertos_df['airport_id'] == int(destino_id)]['name'].values[0]
        mensaje += f"{origen} -> {destino}: {frecuencia} veces\n"
    text_area.delete(1.0, tk.END)
    text_area.insert(tk.END, mensaje)

if __name__ == "__main__":
    # Carga datasets y crear el grafo
    rutas_df, aeropuertos_df = cargar_datos()
    G = crear_grafo(rutas_df)

    root = tk.Tk()
    root.title("Optimizacion de Rutas Aareas en Norteamerica")
    root.geometry("1200x800")

    crear_interfaz()
    root.mainloop()