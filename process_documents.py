import os
import glob
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

# Define paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
DB_DIR = os.path.join(DATA_DIR, "chroma_db")
LOG_DIR = os.path.join(DATA_DIR, "logs")

# Make sure directories exist
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def process_documents():
    """Process all documents in the raw directory and create embeddings."""
    # Create a log file for chunking details
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chunk_log_path = os.path.join(LOG_DIR, f"chunk_log_{timestamp}.txt")
    
    with open(chunk_log_path, "w", encoding="utf-8") as log_file:
        log_file.write("=== DOCUMENT CHUNKING LOG ===\n")
        log_file.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
        
        # Initialize embedding model
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        log_file.write(f"Embedding model: {embeddings.model}\n\n")
        
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Larger chunks for legal text
            chunk_overlap=300,  # More overlap to preserve context
            separators=["\n\n", "\n", ".", " "]
        )
        
        log_file.write("Chunking configuration:\n")
        log_file.write(f"- Chunk size: 1000 characters\n")
        log_file.write(f"- Chunk overlap: 200 characters\n")
        log_file.write(f"- Separators: [\"\\n\\n\", \"\\n\", \".\", \" \"]\n\n")
        
        # Initialize vector store
        db = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings
        )
        
        # Get all files in the raw directory
        all_files = (
            glob.glob(os.path.join(RAW_DIR, "*.pdf")) +
            glob.glob(os.path.join(RAW_DIR, "*.txt"))
        )
        
        # Check if there are any files
        if not all_files:
            print(f"No PDF or text files found in {RAW_DIR}")
            log_file.write(f"No PDF or text files found in {RAW_DIR}\n")
            return
        
        log_file.write(f"Found {len(all_files)} files to process:\n")
        for file in all_files:
            log_file.write(f"- {os.path.basename(file)}\n")
        log_file.write("\n")
        
        # Process each file
        all_chunks = []
        file_chunk_counts = {}
        
        for file_path in all_files:
            file_name = os.path.basename(file_path)
            print(f"Processing {file_name}...")
            log_file.write(f"\n==== PROCESSING FILE: {file_name} ====\n")
            
            # Load the document
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                log_file.write("File type: PDF\n")
            else:
                loader = TextLoader(file_path)
                log_file.write("File type: Text\n")
            
            documents = loader.load()
            log_file.write(f"Document loaded: {len(documents)} pages/sections\n")
            
            # Add source metadata
            for doc in documents:
                doc.metadata["source"] = file_name
            
            # Split into chunks
            chunks = text_splitter.split_documents(documents)
            file_chunk_counts[file_name] = len(chunks)
            
            log_file.write(f"Created {len(chunks)} chunks\n\n")
            
            # Log detailed information about chunks
            for i, chunk in enumerate(chunks):
                log_file.write(f"--- CHUNK {i+1}/{len(chunks)} ---\n")
                log_file.write(f"Length: {len(chunk.page_content)} characters\n")
                log_file.write(f"Metadata: {chunk.metadata}\n")
                
                # Show full content for first chunk, summary for others
                if i == 0:
                    log_file.write("FULL CONTENT:\n")
                    log_file.write(chunk.page_content)
                    log_file.write("\n")
                else:
                    log_file.write(f"Preview: {chunk.page_content[:100]}...\n")
                
                log_file.write("-" * 50 + "\n")
                
                # Print some chunk information to console for visibility
                if i == 0 or i == len(chunks) - 1 or i % 10 == 0:
                    print(f"  Chunk {i+1}/{len(chunks)}: {len(chunk.page_content)} chars")
                    if i == 0:
                        print(f"  First 100 chars: {chunk.page_content[:100]}...")
            
            all_chunks.extend(chunks)
            print(f"  Created {len(chunks)} chunks from {file_name}")
        
        # Add documents to vector store
        print(f"Adding {len(all_chunks)} chunks to vector store...")
        log_file.write(f"\n\n==== VECTOR STORE SUMMARY ====\n")
        log_file.write(f"Total chunks being added: {len(all_chunks)}\n")
        log_file.write("Chunks per file:\n")
        for file_name, count in file_chunk_counts.items():
            log_file.write(f"- {file_name}: {count} chunks\n")
        
        # Add to vector store
        db.add_documents(all_chunks)
        print(f"Chunks added to vector store. Collection now has {db._collection.count()} documents.")
        
        log_file.write("\nProcessing complete!\n")
        log_file.write(f"Vector store saved to: {DB_DIR}\n")
    
    print(f"Processing complete! Chunk log saved to: {chunk_log_path}")
    return chunk_log_path

if __name__ == "__main__":
    log_path = process_documents()
    print(f"You can view detailed chunking information in: {log_path}")