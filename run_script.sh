python create_documents.py -place_name $1
python populate_database.py -place_name  $1
python inference.py -place_name $1 -path_type $2 $3
python evaluate.py -place_name $1 -path_type $2  -$3
