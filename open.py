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

# Cargar los datos de rutas y aeropuertos
def cargar_datos() -> Tuple[pd.DataFrame, pd.DataFrame]:
    url_rutas = 'https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat'
    rutas_cols = ['airline', 'airline_id', 'source_airport', 'source_airport_id', 'destination_airport', 'destination_airport_id', 'codeshare', 'stops', 'equipment']
    rutas_df = pd.read_csv(url_rutas, header=None, names=rutas_cols)
    
    url_aeropuertos = 'https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat'
    aeropuertos_cols = ['airport_id', 'name', 'city', 'country', 'IATA', 'ICAO', 'lat', 'long', 'alt', 'timezone', 'DST', 'tz', 'type', 'source']
    aeropuertos_df = pd.read_csv(url_aeropuertos, header=None, names=aeropuertos_cols)
    
    # Filtrar aeropuertos de Norteamerica
    paises_norteamerica = ['United States', 'Canada', 'Mexico']
    aeropuertos_na = aeropuertos_df[aeropuertos_df['country'].isin(paises_norteamerica)]
    
    # Filtrar rutas que conectan aeropuertos de Norteamerica
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
        source = str(row['source_airport_id'])
        destination = str(row['destination_airport_id'])
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
    
    # Encontrar aeropuertos conectados directamente desde el aeropuerto de origen
    rutas_desde_origen = list(G.successors(origen_id))
    if rutas_desde_origen:
        resultado.set(f"Rutas desde {origen}: {len(rutas_desde_origen)}")
        visualizar_rutas_conectadas(origen_id, rutas_desde_origen)
    else:
        resultado.set(f"No se encontraron rutas desde {origen}.")

# Visualizar rutas desde el aeropuerto de origen usando Basemap
def visualizar_rutas_conectadas(origen_id, rutas_desde_origen):
    plt.figure(figsize=(15, 10))
    m = Basemap(llcrnrlon=-130, llcrnrlat=20, urcrnrlon=-60, urcrnrlat=55,
                projection='lcc', lat_1=33, lat_2=45, lon_0=-95)
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    
    # Coordenadas del aeropuerto de origen
    origen = aeropuertos_df[aeropuertos_df['airport_id'] == int(origen_id)]
    origen_lon = float(origen['long'].iloc[0])
    origen_lat = float(origen['lat'].iloc[0])
    x_origen, y_origen = m(origen_lon, origen_lat)
    m.plot(x_origen, y_origen, 'go', markersize=12, label="Origen")  # Origen en verde
    
    # Coordenadas de aeropuertos conectados
    for destino_id in rutas_desde_origen:
        destino = aeropuertos_df[aeropuertos_df['airport_id'] == int(destino_id)]
        if not destino.empty:
            dest_lon = float(destino['long'].iloc[0])
            dest_lat = float(destino['lat'].iloc[0])
            x_dest, y_dest = m(dest_lon, dest_lat)
            m.plot(x_dest, y_dest, 'bo', markersize=8)  # Destinos en azul
            m.drawgreatcircle(origen_lon, origen_lat, dest_lon, dest_lat, linewidth=1, color='b')

    plt.title(f"Rutas desde {origen['name'].values[0]}")
    plt.legend()
    plt.show()

# Calcular la ruta rápida entre el aeropuerto de origen y destino
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
        # Calcular la ruta más rápida entre los aeropuertos de origen y destino
        ruta_rapida = nx.shortest_path(G, source=origen_id, target=destino_id, weight='weight')
        resultado.set(f"Ruta rápida encontrada: {ruta_rapida}")
        
        # Visualizar la ruta rápida en el mapa
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
    
    plt.title("Ruta Aérea")
    plt.show()

def calcular_flujo_maximo():
    global G, aeropuertos_df
    
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
        # Crear una copia del grafo para el cálculo del flujo máximo
        G_flow = G.copy()
        
        # Asignar capacidades a las aristas (asumimos capacidad 1 por cada ruta)
        for u, v in G_flow.edges():
            G_flow[u][v]['capacity'] = 1
        
        # Calcular el flujo máximo
        flow_value, flow_dict = nx.maximum_flow(G_flow, origen_id, destino_id)
        
        # Obtener las aristas con flujo positivo
        flow_edges = []
        for u in flow_dict:
            for v, flow in flow_dict[u].items():
                if flow > 0:
                    flow_edges.append((u, v))
        
        resultado.set(f"Flujo máximo encontrado: {flow_value} rutas independientes")
        
        # Visualizar el grafo de flujo máximo
        visualizar_flujo_maximo(flow_edges, origen_id, destino_id)
        
    except nx.NetworkXError as e:
        resultado.set(f"Error al calcular el flujo máximo: {str(e)}")

def visualizar_flujo_maximo(flow_edges: List[Tuple[str, str]], origen_id: str, destino_id: str):
    """Visualiza el grafo con las rutas del flujo máximo."""
    plt.figure(figsize=(15, 10))
    
    # Crear el mapa base
    m = Basemap(llcrnrlon=-130, llcrnrlat=20, urcrnrlon=-60, urcrnrlat=55,
                projection='lcc', lat_1=33, lat_2=45, lon_0=-95)
    
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    
    # Dibujar las aristas del flujo máximo
    for start_id, end_id in flow_edges:
        start = aeropuertos_df[aeropuertos_df['airport_id'] == int(start_id)]
        end = aeropuertos_df[aeropuertos_df['airport_id'] == int(end_id)]
        
        if not start.empty and not end.empty:
            start_lon = float(start['long'].iloc[0])
            start_lat = float(start['lat'].iloc[0])
            end_lon = float(end['long'].iloc[0])
            end_lat = float(end['lat'].iloc[0])
            
            # Dibujar la ruta
            m.drawgreatcircle(start_lon, start_lat, end_lon, end_lat, 
                            linewidth=2, color='purple', alpha=0.6)
            
            # Dibujar los puntos de los aeropuertos
            x1, y1 = m(start_lon, start_lat)
            x2, y2 = m(end_lon, end_lat)
            
            # Si es origen o destino, usar un color diferente
            if start_id == origen_id:
                m.plot(x1, y1, 'go', markersize=12, label='Origen')
            elif end_id == destino_id:
                m.plot(x2, y2, 'ro', markersize=12, label='Destino')
            else:
                m.plot(x1, y1, 'bo', markersize=8)
                m.plot(x2, y2, 'bo', markersize=8)
    
    plt.title("Rutas de Flujo Máximo")
    plt.legend()
    plt.show()

# Modificar la creación de la interfaz para agregar el nuevo botón
def crear_interfaz():
    global root, aeropuerto_origen, aeropuerto_destino, resultado
    
    # [Mantener el código existente de la interfaz]
    
    # Agregar el nuevo botón para flujo máximo
    boton_flujo_maximo = ttk.Button(main_frame, text="Calcular Flujo Máximo", 
                                   command=calcular_flujo_maximo)
    boton_flujo_maximo.grid(row=6, column=0, pady=5, columnspan=2)

# Modificar la parte principal del código
if __name__ == "__main__":
    # Cargar datasets y crear el grafo globalmente
    rutas_df, aeropuertos_df = cargar_datos()
    G = crear_grafo(rutas_df)

    # Crear la ventana principal
    root = tk.Tk()
    root.title("Optimización de Rutas Aéreas en Norteamérica")
    root.geometry("600x400")

    # Frame principal
    main_frame = ttk.Frame(root, padding="10")
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Etiquetas y selección de aeropuertos
    ttk.Label(main_frame, text="Aeropuerto de origen:").grid(row=0, column=0, sticky=tk.W, pady=5)
    aeropuerto_origen = ttk.Combobox(main_frame, values=aeropuertos_df['name'].tolist(), width=50)
    aeropuerto_origen.grid(row=0, column=1, pady=5)

    ttk.Label(main_frame, text="Aeropuerto de destino:").grid(row=1, column=0, sticky=tk.W, pady=5)
    aeropuerto_destino = ttk.Combobox(main_frame, values=aeropuertos_df['name'].tolist(), width=50)
    aeropuerto_destino.grid(row=1, column=1, pady=5)

    # Botones
    boton_calcular_rutas = ttk.Button(main_frame, text="Calcular y Visualizar Rutas", 
                                     command=calcular_y_visualizar_rutas)
    boton_calcular_rutas.grid(row=5, column=0, pady=5, columnspan=2)

    boton_calcular_ruta_rapida = ttk.Button(main_frame, text="Calcular Ruta Rápida", 
                                           command=calcular_ruta_rapida)
    boton_calcular_ruta_rapida.grid(row=3, column=0, pady=5, columnspan=2)

    boton_flujo_maximo = ttk.Button(main_frame, text="Calcular Flujo Máximo", 
                                   command=calcular_flujo_maximo)
    boton_flujo_maximo.grid(row=6, column=0, pady=5, columnspan=2)

    # Resultado
    resultado = tk.StringVar()
    ttk.Label(main_frame, textvariable=resultado, wraplength=500).grid(row=4, column=0, columnspan=2, pady=5)

    root.mainloop()