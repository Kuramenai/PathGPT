import os
import shutil
from pathlib import Path
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader, UnstructuredMarkdownLoader, TextLoader, UnstructuredFileLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.schema.document import Document
from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from termcolor import cprint
from tqdm import tqdm

from variables import *

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ[" HF_HUB_ENABLE_HF_TRANSFER'"] = "1"
  

MAX_BATCH_SIZE= 41666

document = f"text_docs/{place_name}_paths.txt"
chroma_path = f"chroma_db/{embedding_model_formatted_name}/{place_name}/"

cprint('\nGENERATING EMBEDDINGS FOR :', 'light_yellow', attrs=['bold'])
cprint(f'-PATHS IN : {place_name}', 'green')
cprint(f'-USING : {embedding_model}\n', 'green')

 
def load_documents(document):
    cprint("Loading document(s)...", "light_yellow")
    document_loader = TextLoader(document)
    cprint("Loading document(s) completed!\n", "green")
    return document_loader.load()

def split_doc_by_md_header(documents: list[Document]):
    headers_to_split_on = [("###", "Header 3")]
    markdown_splitter = MarkdownHeaderTextSplitter(
                        headers_to_split_on=headers_to_split_on, 
                        strip_headers=False)
    txt = " ".join([d.page_content for d in documents])
    return markdown_splitter.split_text(txt)

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=256,
        chunk_overlap=32,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def get_embeddings():
    cprint("Calculating embeddings...", "light_yellow")
    return HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        show_progress = True)    

def add_to_chroma(chunks: list[Document]):
    cprint("Adding documents to database...", "light_yellow")
    # Create directory
    if os.path.isdir(chroma_path):
        cprint("Folder %s already exists" % chroma_path, "blue")
    else:
        Path(chroma_path).mkdir(parents=True, exist_ok=True)
        cprint("Folder %s created!" % chroma_path, "green")

   
    embeddings = get_embeddings()
    
    cprint("Indexing...", "light_yellow")
    # Load the existing database.
    db = Chroma(
        persist_directory=chroma_path, embedding_function=embeddings
    )

   # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        for i in tqdm(range(0, len(new_chunks), MAX_BATCH_SIZE)):
            batch = chunks[i:i + MAX_BATCH_SIZE]
            db.add_documents(batch, ids=new_chunk_ids[i:i + MAX_BATCH_SIZE])
        cprint(f"New documents added to {chroma_path}.", "green")
    else:
        print("✅ No new documents to add")
            

def calculate_chunk_ids(chunks):

    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)
        cprint('✅ Database cleared', 'green')
    else:
        cprint("No Chroma Database found at this location", 'red')


if __name__ == "__main__":

    # Check if the database should be cleared (using the --clear flag).
    if args.reset:
        cprint('✨ Clearing database...', 'yellow')
        clear_database()

    # Create (or update) the data store.
    documents = load_documents(document)
    chunks = split_documents(documents)
    add_to_chroma(chunks)

