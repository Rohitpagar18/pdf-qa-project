import argparse
import os
import shutil
from pathlib import Path
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document  
from langchain_chroma import Chroma  
from get_embedding_function import get_embedding_function  

CHROMA_PATH = "chroma"
DATA_PATH = "data"

def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)
    
    for chunk in chunks_with_ids:
        content = chunk.page_content.lower()
        # Improved category detection with multiple keywords
        if any(word in content for word in ["education", "degree", "university","post graduate", "pg diploma", "bachelor", "diploma", "engineering", "polytechnic", "school", "college"]):
            chunk.metadata["category"] = "education"
        elif any(word in content for word in ["skill", "proficient", "expert", "programming", "language", "tool", "technology", "framework", "library", "database", "software", "hardware", "platform", "cloud", "service", "methodology", "framework", "tool", "technology", "library", "database", "software", "hardware", "platform", "cloud", "service", "methodology"]):
            chunk.metadata["category"] = "skills"
        elif any(word in content for word in ["project", "case study", "portfolio", "Academic", ]):
            chunk.metadata["category"] = "projects"
        else:
            chunk.metadata["category"] = "general"

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
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        #db.persist() 
    else:
        print("No new documents to add")


def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        # Get or default the source first
        source = chunk.metadata.get("source", "unknown_source")
        
        # Then process it
        source = str(Path(source).name)  # Keep only filename
        
        page = chunk.metadata.get("page", 0)  # Default to page 0 if missing
        current_page_id = f"{source}:{page}"

        # Rest of the code remains the same...
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
