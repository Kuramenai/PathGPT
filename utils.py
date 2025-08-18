import os
from pathlib import Path
from argparse import ArgumentParser


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Iteration timed out")


def make_dir(path):
    if not os.path.isdir(path):
        Path(path).mkdir(parents=True, exist_ok=True)
        print("Folder %s created!" % path)
    else:
        print("Folder %s already exists" % path)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("-place_name", default="harbin", type=str)
    # parser.add_argument("-dataset_usage", default="test", type=str)
    parser.add_argument("-save_as", default="all_paths_one_doc", type=str)
    parser.add_argument("-path_type", default="touristic", type=str)
    parser.add_argument("-use-context", default=False, action="store_true")
    parser.add_argument("-llm", default="qwen2.5-14b", type=str)
    parser.add_argument(
        "-embedding_model", default="intfloat/multilingual-e5-large-instruct", type=str
    )
    parser.add_argument("-top_k", default=9, type=int)
    parser.add_argument("-reset", action="store_true", help="Reset the database.")
    parser.add_argument("-evaluation_remarks", default="", type=str)
    args = parser.parse_args()
    return args


# tencentBAC/conan-embedding-v1  Alibaba-NLP/gte-multilingual-base intfloat/multilingual-e5-large-instruct
