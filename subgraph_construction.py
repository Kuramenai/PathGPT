import pickle
import variables
import geopandas as gpd
from termcolor import cprint
from tqdm import tqdm
from collections import defaultdict
from generate_custom_dataset import edge_id_to_node_id
from typing import List, Dict, Any, Set, Tuple

def construct_local_subgraphs(dataset: List[Dict[str, Any]]) -> Dict[Tuple[int, int], Dict[int, Set[int]]]:
    """
    Constructs a local subgraph (adjacency list) for each Origin-Destination pair 
    by merging historical, fastest, shortest, and custom paths.
    """
    
    # Nested defaultdict: OD_pair -> { current_node -> {next_node_1, next_node_2} }
    # This automatically handles missing keys and prevents KeyErrors.
    subgraphs = defaultdict(lambda: defaultdict(set))
    
    for path_collection in tqdm(dataset, total=len(dataset), dynamic_ncols=True, desc="Generating local subgraphs"):
        
        # Safely get paths (in case some are missing)
        historical_path = path_collection.get("historical_path_edges", [])
        fastest_path = path_collection.get("fastest_path_edges",[])
        shortest_path = path_collection.get("shortest_path_edges",[])
        custom_path = path_collection.get(f"{variables.path_type}_path_edges",[])
        
        # Skip if there's no historical path to define the OD pair
        if not historical_path:
            continue
            
        start_edge, destination_edge = historical_path[0], historical_path[-1]
        # u_1, v_1, k_1 = edge_id_to_uvk(start_edge)
        # u_n, v_n, k_n = edge_id_to_uvk(destination_edge)
        od_pair = (start_edge, destination_edge)
        
        # Merge all edges from all 4 paths into the subgraph
        for path in [historical_path, fastest_path, shortest_path, custom_path]:
            if not path:
                continue # Skip empty paths
                
            for i in range(len(path) - 1):
                u = path[i]       # Current edge ID
                v = path[i + 1]   # Next edge ID
                
                # Because we used defaultdict, we can directly add to the set
                subgraphs[od_pair][u].add(v)
                
    # Convert back to standard dict for normal usage/printing
    return dict(subgraphs)

def compress_edge_subgraph(
    subgraph: Dict[int, Set[int]], 
    start_edge: int
) -> Dict[Tuple[int, ...], Set[Tuple[int, ...]]]:
    """
    Compresses an edge-based subgraph by merging linear edge sequences.
    """
    # 1. Calculate in-degrees
    in_degree = {edge: 0 for edge in subgraph}
    for u, neighbors in subgraph.items():
        for v in neighbors:
            if v not in in_degree:
                in_degree[v] = 0
            in_degree[v] += 1

    def traverse_and_compress(current_edge: int) -> Tuple[int, ...]:
        """Walks forward until it hits a decision boundary."""
        segment = [current_edge]
        curr = current_edge
        
        while True:
            neighbors = list(subgraph.get(curr, []))
            
            # Stop if no neighbors (dead end / destination)
            if not neighbors:
                break
                
            # Stop if fork in the road
            if len(neighbors) > 1:
                break
                
            nxt = neighbors[0]
            
            # Stop if the next edge is a merge point
            if in_degree.get(nxt, 0) > 1:
                break
                
            # Otherwise, it's a 1-to-1 connection. Add and continue.
            segment.append(nxt)
            curr = nxt
            
        return tuple(segment)

    # 2. Graph Building Logic (BFS Approach)
    compressed_graph = defaultdict(set)
    visited_segments = set()
    queue = [start_edge] # Start building from the origin

    while queue:
        curr_start = queue.pop(0)
        
        # Get the compressed segment starting at this edge
        segment = traverse_and_compress(curr_start)
        
        if segment in visited_segments:
            continue
            
        visited_segments.add(segment)
        
        # Look at the last edge in our newly formed segment
        last_edge = segment[-1]
        neighbors = subgraph.get(last_edge, [])
        
        # For every neighbor, find its compressed segment and link them
        for nxt in neighbors:
            nxt_segment = traverse_and_compress(nxt)
            compressed_graph[segment].add(nxt_segment)
            
            # Queue the next segment's start edge if we haven't processed it
            if nxt_segment not in visited_segments:
                queue.append(nxt)

    # Convert inner sets back to standard dict for clean output
    return {k: set(v) for k, v in compressed_graph.items()}
        
            
        
if __name__ == "__main__":
    
    edges_df = gpd.read_file(variables.EDGE_DATA)

    edge_id_to_uvk = {i: (row.u, row.v, row.key) for i, row in edges_df.iterrows()}
    
    filtered_train_data = f"filtered_train_data/{variables.path_type}/{variables.place_name}_data"
    try:
        with open(filtered_train_data, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        cprint(f"Test data not found at {data}! Please run generate_custom_dataset.py first.", "red")
        exit(1)

    cprint(f"Loaded {len(data)} test samples from.", "cyan")
    
    # Local graph construction and compression below.
    
    # 1. Build the uncompressed subgraphs 
    cprint("Constructing raw local subgraphs...", "yellow")
    uncompressed_subgraphs = construct_local_subgraphs(data)
    
    # 2. Loop through and compress each one
    final_compressed_graphs = {}
    
    cprint("Compressing subgraphs...", "yellow")

    for od_pair, subgraph in tqdm(uncompressed_subgraphs.items(), desc="Compressing", dynamic_ncols=True):
        start_edge, dest_edge = od_pair
        
        compressed_subgraph = compress_edge_subgraph(subgraph, start_edge)
        
        final_compressed_graphs[od_pair] = compressed_subgraph

    cprint(f"Successfully generated {len(final_compressed_graphs)} compressed subgraphs!", "green")
    
    with open(f"compressed_subgraphs/{variables.path_type}/{variables.place_name}_data", "wb") as f:
        pickle.dump(final_compressed_graphs, f)
    
    