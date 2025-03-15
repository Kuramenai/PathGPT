import pickle
import re
import random
import osmnx as ox 
import geopandas as gpd
import networkx as nx
from tqdm import tqdm
from collections import defaultdict
from termcolor import cprint 

from utils import get_args, make_dir
from variables import *

edges_df = gpd.read_file(EDGE_DATA)
map_edge_id_to_u_v = edges_df[['u', 'v']].to_numpy()
map_u_v_to_edge_id = {(u,v):i for i,(u,v) in enumerate(map_edge_id_to_u_v)}

def welcome_text():
	cprint('\n\nGENERATING DATASET FOR :', 'light_yellow', attrs=['bold'])
	cprint(f'-PLACE NAME : {place_name}', 'green')
	cprint(f'-USE FOR : {dataset_usage}', 'green')

def condense_edges(edge_route):
	global map_edge_id_to_u_v, map_u_v_to_edge_id
	route = [map_u_v_to_edge_id[tuple(map_edge_id_to_u_v[e])] for e in edge_route]
	return route

def fetch_map_fid_to_zero_indexed(data):
	s = set()
	for _,t,_ in data:
		s.update(set(t))
	return {el:i for i,el in enumerate(s)}

def relabel_trips(data, mapping):
	return [(idx, [mapping[e] for e in trip], timestamps) for (idx, trip, timestamps) in data]

def remove_loops(path):
	reduced = []
	last_occ = {p:-1 for p in path}
	for i in range(len(path)-1,-1,-1):
		if last_occ[path[i]] == -1:
			last_occ[path[i]] = i
	current = 0
	while(current < len(path)):
		reduced.append(path[current])
		current = last_occ[path[current]] + 1
	return reduced

def nbrs_sanity_check(node_nbrs, data):
	print("SANITY CHECK 1")
	for _, t, _ in tqdm(data, dynamic_ncols=True):
		for i in range(len(t)-1):
			assert t[i+1] in node_nbrs[t[i]], "How did this happen?"
	print('Cleared :)')

def create_node_nbrs(forward):
	start_nodes = defaultdict(set)
	for e in forward:
		u,v = map_edge_id_to_u_v[e]
		start_nodes[u].add(forward[e])
	node_nbrs = {}	# here nodes are actually edges of the road network
	for e in forward:
		_,v = map_edge_id_to_u_v[e]
		node_nbrs[forward[e]] = list(start_nodes[v])
	return node_nbrs

def edge_id_to_node_id(path):
	return [map_edge_id_to_u_v[path[0]][0], map_edge_id_to_u_v[path[0]][1]] + [map_edge_id_to_u_v[edge][1] for edge in path[1:]]

def load_data_init(fname, less=False, sample=1000):
	print("Loading map matched trajectories")
	f = open(fname, "rb")
	data = pickle.load(f)
	f.close()
			
	if less:
		data = data[:sample]

	data = [(idx, condense_edges(t), timestamps) for (idx, t, timestamps) in tqdm(data, dynamic_ncols=True)]
	
	data = [(idx, remove_loops(t),timestamps) for (idx,t,timestamps) in tqdm(data, dynamic_ncols=True)]
		
    # ignoring very small trips   
	data = [(idx,t,timestamps) for (idx,t,timestamps) in tqdm(data, dynamic_ncols=True) if len(t) >= 5]	
	
	forward = fetch_map_fid_to_zero_indexed(data)	

	data = relabel_trips(data, forward)
	return data, forward

def load_data(fname, less=False, number_of_samples=5):
	cprint("\nLoading map matched trajectories", "blue")
	f = open(fname, "rb")
	data = pickle.load(f)
	f.close()

	random.shuffle(data) 

	if less:	
		data = data[:number_of_samples]

	data = [(idx, condense_edges(t), timestamps) for (idx, t, timestamps) in tqdm(data, dynamic_ncols=True)]
	
	data = [(idx, remove_loops(t),timestamps) for (idx,t,timestamps) in tqdm(data, dynamic_ncols=True)]
		
    # ignoring very small trips   
	data = [t for (idx,t,timestamps) in tqdm(data, dynamic_ncols=True) if len(t) >= 5]	

	# Each sample data is now a sequence of nodes ID, we will use these nodes to compute new paths later
	data = [edge_id_to_node_id(path) for path in tqdm(data, dynamic_ncols=True)]

	return data

def get_path_with_road_names(path, map_uv_to_road_names):
	start, destination = 0, 1
	path_with_road_names = []
	while start < destination and destination < len(path):
		road_name = map_uv_to_road_names[(path[start], path[destination])]
		if road_name is not None:
			# some nodes may be the intersection of two or more roads, split to find the different roads and format the strings retrieved
			road_names = road_name.split(',')
			for name in road_names:
				if place_name == 'porto':
					formatted_name = re.sub(r"[\']", "", name).replace(" ", ' ')
					formatted_name = formatted_name.strip()
				else:
					formatted_name = re.sub(r"[\']", "", name).replace(" ", '')
				path_with_road_names.append(formatted_name)
		start=destination
		destination += 1
	# remove duplicates (a road(edge) may pass through different intersections(node))
	path_with_road_names = list(dict.fromkeys(path_with_road_names))
	return path_with_road_names

def extract_road_names():
	cprint('\nExtracting road names...', 'light_yellow')
	# We find the sreeet/road name associated to each edge ID and map the starting and destination node of an edge to its corresponding road name
	edges_uv_road_names = edges_df[['u', 'v', 'name']].to_numpy()
	road_names = {(u,v): re.sub(r"[\[\]]", "", name) if name is not None else None for (u,v,name) in edges_uv_road_names}
	# f = open(f"backup/new_{place_name}_map_uv_to_road_names", 'rb')
	# road_names = pickle.load(f)
	# f.close()
	cprint('Roads names extracted successfully!', 'green')
	return road_names

def load_graph():
	cprint('\nLoading graph...', 'light_yellow')
	f = open(PICKLED_GRAPH,'rb')
	graph = pickle.load(f)
	f.close()
	# Add edge speeds and travel times as weights to road network graph in order to compute fastest and shortest paths
	ox.add_edge_speeds(graph)
	ox.add_edge_travel_times(graph)
	cprint('Graph loaded successfully!', 'green')
	return graph

def compute_paths(graph,
				  start_node,
				  destination_node,
				  road_names,
				  weight= None,
				  original_path = []):
	
	if weight is None:
		path_with_road_names = get_path_with_road_names(original_path, road_names)
	
		if len(path_with_road_names) < 2:
			return [], []
		
		return original_path, path_with_road_names
	
	path = nx.shortest_path(graph,start_node,destination_node,weight)
	path_with_road_names = get_path_with_road_names(path, road_names)
	
	if len(path_with_road_names) < 2:
		return [], []
	
	return path, path_with_road_names

def augment_data(graph, data, road_names):
	
	cprint('\nAugmenting  data...', 'light_yellow')
	
	dataset = []
	for original_path in tqdm(data, dynamic_ncols=True):

		start_node = original_path[0]
		destination_node = original_path[-1]

		_, original_path_with_road_names = compute_paths(graph, start_node, destination_node, road_names, weight=None, original_path=original_path)
		fastest_path, fastest_path_with_road_names =  compute_paths(graph, start_node, destination_node, road_names, 'travel_time')
		shortest_path, shortest_path_with_road_names = compute_paths(graph, start_node, destination_node, road_names, 'length')

		skip_path_flag = len(original_path_with_road_names)*len(fastest_path_with_road_names)*len(shortest_path_with_road_names)
		if skip_path_flag == 0:
			continue
		else:
			path_collection = {}
			path_collection['original_path'] = original_path 
			path_collection['fastest_path'] = fastest_path
			path_collection['shortest_path'] = shortest_path
			path_collection['original_path_with_road_names'] = original_path_with_road_names
			path_collection['fastest_path_with_road_names'] =  fastest_path_with_road_names
			path_collection['shortest_path_with_road_names'] = shortest_path_with_road_names
			dataset.append(path_collection)

	cprint('Data augmentation done!', 'green')
	return dataset

def augment_test_data(graph, data, road_names):
	print("Creating dataset...\n")
	
	dataset = []
	for original_path in tqdm(data, dynamic_ncols=True):

		start_node = original_path[0]
		destination_node = original_path[-1]

		_, original_path_with_road_names = compute_paths(graph, start_node, destination_node, road_names, None, original_path=original_path)

		if len(original_path_with_road_names) == 0:
			continue
		else:
			path_collection = {}
			path_collection['original_path'] = original_path 
			path_collection['original_path_with_road_names'] = original_path_with_road_names
			dataset.append(path_collection)

	print(len(dataset))

	return dataset

def save_file(file_path, file):
	make_dir(file_path)
	with open(f"{file_path}/{place_name}_data", "wb") as f:
		pickle.dump(file, f)

	cprint(f'File saved at {file_path}/{place_name}_data.', 'green')
	


if __name__ == "__main__" :

	if dataset_usage == 'train':
		data = load_data(fname = TRAIN_TRIP_DATA_PICKLED_WITH_TIMESTAMPS)
	elif dataset_usage == 'test':
		data = load_data(fname= TEST_TRIP_SMALL_FIXED_DATA_PICKLED_WITH_TIMESTAMPS)		
	else:
		raise ValueError("This usage of data is not supported, support only train and test.")
	
	welcome_text()
	graph = load_graph()
	road_names = extract_road_names()
	augmented_data= augment_data(
				graph=graph,
				data = data,
				road_names=road_names)
	file_path = dataset_usage
	save_file(file_path, file=augmented_data)




