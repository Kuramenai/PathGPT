python enhance_dataset.py -place_name $1
python create_json_file.py -place_name $1
python inference.py -place_name $1 -path_type $2 $3
python evaluate.py -place_name $1 -path_type $2  -$3
