from utils import get_args

args = get_args()
place_name = args.place_name
dataset_usage = args.dataset_usage
path_type = args.path_type
save_as = args.save_as
use_context = args.use_context
llm = args.llm
embedding_model = args.embedding_model
embedding_model_formatted_name = embedding_model.split('/')[1]
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True }
number_of_docs_to_retrieve = args.retrieval_docs_no

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


if place_name == 'beijing':
    city_name = '北京'
elif place_name == 'chengdu':
    city_name = '成都'
elif place_name == 'harbin':
    city_name = '哈尔滨'
else:
    city_name = place_name

