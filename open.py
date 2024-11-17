import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from typing import Tuple, Dict, List
from math import radians, sin, cos, sqrt, atan2

G = None
aeropuertos_df = None
distancias_minimas = None
frecuencia_rutas = {}

# Formula de Haversine 
def calcular_distancia_haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # Radio de la Tierra en km
    R = 6371  

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c


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

    # Filtro de aeropuertos de Estados Unidos
    aeropuertos_usa = aeropuertos_df[aeropuertos_df['country'] == 'United States']
    aeropuertos_usa_ids = set(aeropuertos_usa['airport_id'].astype(str))
    
    rutas_usa = rutas_df[
        (rutas_df['source_airport_id'].astype(str).isin(aeropuertos_usa_ids)) &
        (rutas_df['destination_airport_id'].astype(str).isin(aeropuertos_usa_ids))
    ]
    return rutas_usa, aeropuertos_usa


# Crear el grafo de rutas
def crear_grafo(rutas_df: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    
    for _, ruta in rutas_df.iterrows():
        origen_id = str(ruta['source_airport_id'])
        destino_id = str(ruta['destination_airport_id'])
        
        if origen_id != '\\N' and destino_id != '\\N':
            # Obtener coordenadas de los aeropuertos
            origen = aeropuertos_df[aeropuertos_df['airport_id'] == int(origen_id)]
            destino = aeropuertos_df[aeropuertos_df['airport_id'] == int(destino_id)]
            
            if not origen.empty and not destino.empty:
                distancia = calcular_distancia_haversine(
                    float(origen['lat'].iloc[0]), float(origen['long'].iloc[0]),
                    float(destino['lat'].iloc[0]), float(destino['long'].iloc[0])
                )
                G.add_edge(origen_id, destino_id, weight=distancia)
    
    return G


# Calcular rutas desde el aeropuerto de origen y visualizar
def dijkstra_shortest_path():
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
        distancia, camino = nx.single_source_dijkstra(G, origen_id, destino_id, weight='weight')
        
        # Actualizar frecuencias de rutas
        for i in range(len(camino) - 1):
            actualizar_frecuencia_ruta(camino[i], camino[i + 1])
        
        nombres_camino = [aeropuertos_df[aeropuertos_df['airport_id'] == int(n)]['name'].values[0] 
                         for n in camino]
        
        mensaje = f"Ruta más corta (Dijkstra) desde {origen} hacia {destino}:\n\n"
        mensaje += " -> ".join(nombres_camino)
        mensaje += f"\n\nDistancia total: {distancia:.2f} km"
        
        text_area.delete(1.0, tk.END)
        text_area.insert(tk.END, mensaje)
        
        visualizar_ruta(camino)
        
    except nx.NetworkXNoPath:
        resultado.set("No existe una ruta entre los aeropuertos seleccionados.")


# Algoritmo Floyd Warshall
def floyd_warshall():
    """Implementa el algoritmo de Floyd-Warshall para todas las rutas posibles."""
    global G
    
    try:
        # Calcular las distancias más cortas entre todos los pares de nodos
        distancias = dict(nx.floyd_warshall(G, weight='weight'))
        
        mensaje = "Análisis de rutas (Floyd-Warshall):\n\n"
        mensaje += "Top 10 rutas más cortas:\n"
        
        # Convertir el diccionario de distancias a una lista para ordenar
        # Filtrar las rutas que van de un aeropuerto a sí mismo
        rutas_ordenadas = [(origen, destino, dist) 
                          for origen in distancias 
                          for destino, dist in distancias[origen].items()
                          if origen != destino]  # Excluir rutas hacia el mismo aeropuerto
        
        # Ordenar por distancia
        rutas_ordenadas.sort(key=lambda x: x[2])
        
        # Mostrar las 10 rutas más cortas (excluyendo rutas a sí mismo)
        for origen_id, destino_id, distancia in rutas_ordenadas[:10]:
            origen = aeropuertos_df[aeropuertos_df['airport_id'] == int(origen_id)]['name'].values[0]
            destino = aeropuertos_df[aeropuertos_df['airport_id'] == int(destino_id)]['name'].values[0]
            mensaje += f"{origen} -> {destino}: {distancia:.2f} km\n"
        
        text_area.delete(1.0, tk.END)
        text_area.insert(tk.END, mensaje)
        
    except nx.NetworkXError as e:
        resultado.set(f"Error en Floyd-Warshall: {str(e)}")


# Calcula rutas desde el aeropuerto de origen y visualizar
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


# Visualiza rutas desde el aeropuerto de origen
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



# Algoritmo Ford Fulkerson
def ford_fulkerson():
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
        # Crear una copia del grafo para el flujo
        G_flow = G.copy()
        for u, v in G_flow.edges():
            G_flow[u][v]['capacity'] = 1

        # Calcular flujo máximo
        flow_value, flow_dict = nx.maximum_flow(G_flow, origen_id, destino_id)
        
        # Obtener todas las conexiones con flujo positivo
        flow_edges = [(u, v) for u in flow_dict for v, flow in flow_dict[u].items() if flow > 0]

        # Primera visualización: Todas las conexiones posibles
        plt.figure(figsize=(15, 10))
        m = Basemap(llcrnrlon=-130, llcrnrlat=20, urcrnrlon=-60, urcrnrlat=55, 
                   projection='lcc', lat_1=33, lat_2=45, lon_0=-95)
        m.drawcoastlines()
        m.drawcountries()
        m.drawstates()

        # Dibujar todas las conexiones con flujo positivo
        for start_id, end_id in flow_edges:
            start = aeropuertos_df[aeropuertos_df['airport_id'] == int(start_id)]
            end = aeropuertos_df[aeropuertos_df['airport_id'] == int(end_id)]
            
            if not start.empty and not end.empty:
                start_lon, start_lat = float(start['long'].iloc[0]), float(start['lat'].iloc[0])
                end_lon, end_lat = float(end['long'].iloc[0]), float(end['lat'].iloc[0])
                
                x1, y1 = m(start_lon, start_lat)
                x2, y2 = m(end_lon, end_lat)
                
                # Dibujar líneas para todas las conexiones
                m.plot([x1, x2], [y1, y2], 'b-', linewidth=1, alpha=0.6)
                m.plot(x1, y1, 'bo', markersize=6)
                m.plot(x2, y2, 'bo', markersize=6)

        plt.title("Todas las conexiones posibles en el flujo máximo")
        plt.show()

        # Preparar mensaje con todas las conexiones
        mensaje = f"Flujo máximo entre {origen} y {destino}: {flow_value} rutas\n\n"
        mensaje += "Todas las conexiones utilizadas en el flujo:\n"
        for u, v in flow_edges:
            origen_nombre = aeropuertos_df[aeropuertos_df['airport_id'] == int(u)]['name'].values[0]
            destino_nombre = aeropuertos_df[aeropuertos_df['airport_id'] == int(v)]['name'].values[0]
            mensaje += f"{origen_nombre} -> {destino_nombre}\n"

        # Encontrar el camino efectivo usando DFS
        rutas_utilizadas = []
        visitados = set()

        def dfs(nodo):
            if nodo == destino_id:
                return True
            visitados.add(nodo)
            for vecino in flow_dict[nodo]:
                if flow_dict[nodo][vecino] > 0 and vecino not in visitados:
                    if dfs(vecino):
                        rutas_utilizadas.append((nodo, vecino))
                        return True
            return False

        dfs(origen_id)

        # Agregar rutas efectivas al mensaje
        mensaje += "\nRutas utilizadas en el camino efectivo:\n"
        for u, v in reversed(rutas_utilizadas):
            origen_nombre = aeropuertos_df[aeropuertos_df['airport_id'] == int(u)]['name'].values[0]
            destino_nombre = aeropuertos_df[aeropuertos_df['airport_id'] == int(v)]['name'].values[0]
            mensaje += f"{origen_nombre} -> {destino_nombre}\n"

        # Mostrar el mensaje en la interfaz
        text_area.delete(1.0, tk.END)
        text_area.insert(tk.END, mensaje)

        # Segunda visualización: Solo el camino efectivo
        plt.figure(figsize=(15, 10))
        m = Basemap(llcrnrlon=-130, llcrnrlat=20, urcrnrlon=-60, urcrnrlat=55, 
                   projection='lcc', lat_1=33, lat_2=45, lon_0=-95)
        m.drawcoastlines()
        m.drawcountries()
        m.drawstates()

        # Dibujar solo las rutas efectivas
        for start_id, end_id in rutas_utilizadas:
            start = aeropuertos_df[aeropuertos_df['airport_id'] == int(start_id)]
            end = aeropuertos_df[aeropuertos_df['airport_id'] == int(end_id)]
            
            if not start.empty and not end.empty:
                start_lon, start_lat = float(start['long'].iloc[0]), float(start['lat'].iloc[0])
                end_lon, end_lat = float(end['long'].iloc[0]), float(end['lat'].iloc[0])
                
                x1, y1 = m(start_lon, start_lat)
                x2, y2 = m(end_lon, end_lat)
                
                # Dibujar líneas más gruesas para el camino efectivo
                m.plot([x1, x2], [y1, y2], 'r-', linewidth=2)
                m.plot(x1, y1, 'ro', markersize=8)
                m.plot(x2, y2, 'ro', markersize=8)

        plt.title("Camino efectivo en el flujo máximo")
        plt.show()

    except nx.NetworkXError as e:
        resultado.set(f"Error en Ford-Fulkerson: {str(e)}")


# Actualiza la frecuencia de uso de una ruta.
def actualizar_frecuencia_ruta(origen_id: str, destino_id: str):
    
    global frecuencia_rutas
    
    if len(frecuencia_rutas) > 100:
        frecuencia_rutas.clear()
    
    key = (origen_id, destino_id)
    frecuencia_rutas[key] = frecuencia_rutas.get(key, 0) + 1


# Algoritmo Bellman-Ford
def calcular_camino_minimo_bellman_ford():
    global G, aeropuertos_df

    origen = aeropuerto_origen.get()
    destino = aeropuerto_destino.get()

    # IDs de origen y destino
    origen_id = aeropuertos_df[aeropuertos_df['name'] == origen]['airport_id']
    destino_id = aeropuertos_df[aeropuertos_df['name'] == destino]['airport_id']

    if origen_id.empty or destino_id.empty:
        resultado.set("Aeropuerto de origen o destino no encontrado.")
        return

    origen_id = str(origen_id.values[0])
    destino_id = str(destino_id.values[0])
    
    try:
        # Usa Bellman-Ford para calcular el camino minimo desde el origen
        distancias, rutas = nx.single_source_bellman_ford(G, source=origen_id, weight='weight')

        if destino_id not in distancias:
            resultado.set("No hay ruta entre el aeropuerto de origen y el destino.")
            return

        # Reconstruir la ruta hacia el destino
        distancia = distancias[destino_id]
        camino = []
        nodo_actual = destino_id

        # Manejo de rutas como diccionario
        while nodo_actual != origen_id:
            if isinstance(rutas[nodo_actual], list):
                rutas[nodo_actual] = rutas[nodo_actual][0]
            camino.append(nodo_actual)
            nodo_actual = rutas[nodo_actual]
        camino.append(origen_id)
        camino.reverse()

        # Obtener nombres de los aeropuertos en el camino
        nombres_camino = [aeropuertos_df[aeropuertos_df['airport_id'] == int(n)]['name'].values[0] for n in camino]

        mensaje = f"Camino minimo desde {origen} hacia {destino}:\n\n"
        mensaje += " -> ".join(nombres_camino) + f"\n\nDistancia total: {distancia} unidades"
        text_area.delete(1.0, tk.END)
        text_area.insert(tk.END, mensaje)
    except nx.NetworkXUnbounded:
        resultado.set("El grafo contiene ciclos con pesos negativos.")
    except nx.NetworkXNoPath:
        resultado.set("No hay ruta entre el aeropuerto de origen y el destino.")

def crear_interfaz():
    global root, aeropuerto_origen, aeropuerto_destino, resultado, text_area
    
    main_frame = ttk.Frame(root, padding="10")
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Frame para controles
    control_frame = ttk.Frame(main_frame)
    control_frame.pack(fill=tk.X, pady=(0, 10))
    
    ttk.Label(control_frame, text="Aeropuerto de origen:").grid(row=0, column=0, sticky=tk.W, pady=5)
    aeropuerto_origen = ttk.Combobox(control_frame, values=sorted(aeropuertos_df['name'].tolist()), width=50)
    aeropuerto_origen.grid(row=0, column=1, pady=5, padx=5)
    
    ttk.Label(control_frame, text="Aeropuerto de destino:").grid(row=1, column=0, sticky=tk.W, pady=5)
    aeropuerto_destino = ttk.Combobox(control_frame, values=sorted(aeropuertos_df['name'].tolist()), width=50)
    aeropuerto_destino.grid(row=1, column=1, pady=5, padx=5)
    
    # Frame para botones
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(fill=tk.X, pady=(0, 10))
    
    # Botones para los diferentes algoritmos
    ttk.Button(button_frame, text="Dijkstra - Ruta más corta", 
               command=dijkstra_shortest_path).pack(side=tk.LEFT, padx=5)
    
    ttk.Button(button_frame, text="Floyd-Warshall - Todas las rutas", 
               command=floyd_warshall).pack(side=tk.LEFT, padx=5)
    
    ttk.Button(button_frame, text="Ford-Fulkerson - Flujo máximo", 
               command=ford_fulkerson).pack(side=tk.LEFT, padx=5)
    
    ttk.Button(button_frame, text="Mostrar Frecuencia de Rutas",
               command=mostrar_frecuencia_rutas).pack(side=tk.LEFT, padx=5)
    
    ttk.Button(button_frame, text="Camino minimo Bellman Ford",
               command=calcular_camino_minimo_bellman_ford).pack(side=tk.LEFT, padx=5)
    
    ttk.Button(button_frame, text="Ver todas las rutas",
               command=calcular_y_visualizar_rutas).pack(side=tk.LEFT, padx=5)

    # Área de resultados con scroll
    result_frame = ttk.Frame(main_frame)
    result_frame.pack(fill=tk.BOTH, expand=True)
    
    scrollbar = ttk.Scrollbar(result_frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    text_area = tk.Text(result_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set, height=20)
    text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    scrollbar.config(command=text_area.yview)
    
    resultado = tk.StringVar()
    resultado.trace('w', lambda *args: actualizar_resultado())


def limpiar_resultados():
    text_area.delete(1.0, tk.END)
    frecuencia_rutas.clear()
    resultado.set("")


def actualizar_resultado():
    
    texto_actual = resultado.get()
    if texto_actual:
        text_area.delete(1.0, tk.END)
        text_area.insert(tk.END, texto_actual)

# Frecuencia de Rutas
def mostrar_frecuencia_rutas():
    mensaje = "Frecuencia de Rutas:\n\n"
    for (origen_id, destino_id), frecuencia in frecuencia_rutas.items():
        origen = aeropuertos_df[aeropuertos_df['airport_id'] == int(origen_id)]['name'].values[0]
        destino = aeropuertos_df[aeropuertos_df['airport_id'] == int(destino_id)]['name'].values[0]
        mensaje += f"{origen} -> {destino}: {frecuencia} veces\n"
    text_area.delete(1.0, tk.END)
    text_area.insert(tk.END, mensaje)

if __name__ == "__main__":
    # Cargar datos y crear el grafo
    rutas_df, aeropuertos_df = cargar_datos()
    G = crear_grafo(rutas_df)
    
    root = tk.Tk()
    root.title("Optimización de Rutas Aéreas en Estados Unidos")
    root.geometry("1200x800")
    
    crear_interfaz()
    
    button_frame = root.winfo_children()[0].winfo_children()[1]
    ttk.Button(button_frame, text="Limpiar Resultados", 
               command=limpiar_resultados).pack(side=tk.LEFT, padx=5)
    
    root.mainloop()