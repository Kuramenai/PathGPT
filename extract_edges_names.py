# %%
import pickle
import geopandas as gpd

# %%
CITY = "beijing"
PREFIX_PATH = f"preprocessed_data/{CITY}_data/"
EDGE_DATA = PREFIX_PATH + "map/edges.shp"
NODE_DATA = PREFIX_PATH + "map/nodes.shp"
PICKLED_GRAPH = PREFIX_PATH + "map/graph_with_haversine.pkl"

edges = gpd.read_file(EDGE_DATA)
node_df = gpd.read_file(NODE_DATA)
edges_with_names = edges[edges["name"].notna()]
edges_with_names = edges_with_names[["u", "v", "name"]].to_numpy()

# %%
print(len(edges_with_names))
print(len(edges))
# %%
f = open(PICKLED_GRAPH, "rb")
graph = pickle.load(f)
f.close()


roads_names = {}
for u, v, key, data in graph.edges(keys=True, data=True):
    # print(data)
    if "name" in data:
        if (u, v, key) in roads_names:
            print(f"0_{roads_names[(u, v)]}")
            print(f"1_{data['name']}")
            print(f"1_{data['name']}")
        roads_names[(u, v, key)] = data["name"]

# %%
with open(f"edges_names/{CITY}_edges_names", "wb") as f:
    pickle.dump(roads_names, f)

# %%
