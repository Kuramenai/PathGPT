import sys
from utils import get_args

# sys.argv = [""]
args = get_args()
place_name = args.place_name
# dataset_usage = args.dataset_usage
path_type = args.path_type
map_path_type = {
    "fastest": "最快",
    "shortest": "最短",
    "most_used": "",
    "fuel_efficient": "不经过高速公路",
    "poi_aware": "经过景点最多的路线",
}
save_as = args.save_as
use_context = args.use_context
llm = args.llm
llm_task = args.llm_task
corridor_graph_form = args.corridor_graph_form
context_name_suffix = "_uncompressed" if corridor_graph_form == "uncompressed" else ""
symbolic_subgraph_root = (
    "symbolic_subgraphs_uncompressed"
    if corridor_graph_form == "uncompressed"
    else "symbolic_subgraphs"
)
embedding_model = args.embedding_model
embedding_model_formatted_name = embedding_model.split("/")[1]
model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": True}
number_of_docs_to_retrieve = args.top_k
retrieval_type = args.retrieval
spatial_candidate_k = args.spatial_candidate_k
spatial_weight = args.spatial_weight
bm25_weight = args.bm25_weight
evaluation_remarks = args.evaluation_remarks
top_k_shortest = args.top_k_shortest
k_shortest = args.k_shortest
soft_retrieved_discount = args.soft_retrieved_discount
soft_anchor_discount = args.soft_anchor_discount
random_anchor_seed = args.random_anchor_seed

# fmt: off
PREFIX_PATH = f"preprocessed_data/{place_name}_data/"
PICKLED_GRAPH = PREFIX_PATH + "map/graph_with_haversine.pkl"
TRAIN_TRIP_DATA_PICKLED_WITH_TIMESTAMPS = PREFIX_PATH + "preprocessed_train_trips_all.pkl"
TEST_TRIP_DATA_PICKLED_WITH_TIMESTAMPS = PREFIX_PATH + "preprocessed_test_trips_all.pkl"
TEST_TRIP_SMALL_FIXED_DATA_PICKLED_WITH_TIMESTAMPS = PREFIX_PATH + "preprocessed_test_trips_small.pkl"
VAL_TRIP_DATA_PICKLED_WITH_TIMESTAMPS = PREFIX_PATH + "preprocessed_validation_trips_all.pkl"
EDGE_DATA = PREFIX_PATH + "map/edges.shp"
NODE_DATA = PREFIX_PATH + "map/nodes.shp"

# edges_df, nodes_df = gpd.read_file(EDGE_DATA), gpd.read_file(NODE_DATA)
# map_edge_id_to_u_v = edges_df[['u', 'v']].to_numpy()
# map_u_v_to_edge_id = {(u,v):i for i,(u,v) in enumerate(map_edge_id_to_u_v)}

chinese_cities = ['beijing', "chengdu", "harbin"]
other_cities = ["porto", "cityindia"]
if place_name == 'beijing':
    city_name = '北京'
elif place_name == 'chengdu':
    city_name = '成都'
elif place_name == 'harbin':
    city_name = '哈尔滨'
else:
    city_name = place_name
