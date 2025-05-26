import os
import re
import json
import time
from datetime import datetime
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from tqdm import tqdm
import logging
from dotenv import load_dotenv
load_dotenv()

# Check for API key
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Define paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
GDPR_DB_DIR = os.path.join(DATA_DIR, "gdpr_db")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Set up logging
LOG_DIR = os.path.join(DATA_DIR, "process_gdpr_logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Create a timestamped log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f"gdpr_processing_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs(GDPR_DB_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

def ensure_complete_sentences(text):
    """Ensure text ends with proper sentence termination."""
    if text and text.strip() and not re.search(r'[.!?;]\s*$', text.strip()):
        text = text.strip() + "..."
    return text.strip()

def extract_gdpr_structure(text):
    """Extract chapters, articles, paragraphs from GDPR text."""
    logger.info("Extracting GDPR structure from text...")
    
    # Extract chapters
    chapter_pattern = re.compile(r'CHAPTER (\w+)[.\s]*([^\n]+)')
    chapters = chapter_pattern.findall(text)
    logger.info(f"Found {len(chapters)} chapters in the GDPR text")
    
    # Extract articles
    article_pattern = re.compile(r'Article (\d+)[.\s]*([^\n]+)(?:\s*\n\s*)?(.*?)(?=Article \d+|\Z)', re.DOTALL)
    articles = article_pattern.findall(text)
    logger.info(f"Found {len(articles)} articles in the GDPR text")
    
    # Structure the document
    structured_gdpr = {
        "title": "General Data Protection Regulation (GDPR)",
        "chapters": [],
        "articles": []
    }
    
    # Process chapters
    for ch_num, ch_title in chapters:
        structured_gdpr["chapters"].append({
            "number": ch_num,
            "title": ch_title.strip(),
        })
    
    # Count total paragraphs and subpoints for logging
    total_paragraphs = 0
    total_subpoints = 0
    
    # Process articles with their paragraphs
    for art_num, art_title, art_content in articles:
        # Extract paragraphs (numbered sections)
        paragraph_pattern = re.compile(r'(\d+)\.\s*(.*?)(?=\d+\.\s+|\Z)', re.DOTALL)
        paragraphs = paragraph_pattern.findall(art_content)
        
        paragraphs_structured = []
        for para_num, para_text in paragraphs:
            # Clean paragraph text
            clean_para_text = para_text.strip()
            clean_para_text = ensure_complete_sentences(clean_para_text)
            
            # Extract subpoints if any (patterns like (a), (b), (c))
            subpoint_pattern = re.compile(r'\(([a-z])\)\s*(.*?)(?=\([a-z]\)|\Z)', re.DOTALL)
            subpoints = subpoint_pattern.findall(clean_para_text)
            
            subpoints_structured = []
            for sp_letter, sp_text in subpoints:
                sp_text = sp_text.strip()
                sp_text = ensure_complete_sentences(sp_text)
                subpoints_structured.append({
                    "letter": sp_letter,
                    "text": sp_text
                })
            
            total_subpoints += len(subpoints_structured)
            
            paragraphs_structured.append({
                "number": para_num,
                "text": clean_para_text,
                "subpoints": subpoints_structured
            })
            
            total_paragraphs += 1
        
        # Find chapter this article belongs to
        article_chapter = None
        for ch in structured_gdpr["chapters"]:
            chapter_pos = text.find(f"CHAPTER {ch['number']}")
            article_pos = text.find(f"Article {art_num}")
            if chapter_pos < article_pos and chapter_pos != -1:
                article_chapter = ch["number"]
        
        structured_gdpr["articles"].append({
            "number": art_num,
            "title": art_title.strip(),
            "content": art_content.strip(),
            "paragraphs": paragraphs_structured,
            "chapter": article_chapter
        })
    
    logger.info(f"Successfully structured GDPR data: {len(structured_gdpr['chapters'])} chapters, " +
                f"{len(structured_gdpr['articles'])} articles, {total_paragraphs} paragraphs, and {total_subpoints} subpoints")
    
    return structured_gdpr

def create_document_chunks(structured_gdpr, source_file):
    """Create document chunks for embedding with rich metadata."""
    logger.info("Creating document chunks for embedding...")
    docs = []
    
    # Statistics for logging
    chunk_types = {
        "chapter": 0,
        "article_full": 0,
        "article_paragraph": 0
    }
    
    # Create chapter overview documents
    for chapter in structured_gdpr["chapters"]:
        doc = Document(
            page_content=f"CHAPTER {chapter['number']}: {chapter['title']}",
            metadata={
                "source": source_file,
                "type": "chapter",
                "chapter_number": chapter["number"],
                "law": "gdpr"
            }
        )
        docs.append(doc)
        chunk_types["chapter"] += 1
    
    # Create article documents
    for article in structured_gdpr["articles"]:
        # Full article with title
        full_article_content = f"Article {article['number']}: {article['title']}\n\n{article['content']}"
        
        doc = Document(
            page_content=full_article_content,
            metadata={
                "source": source_file,
                "type": "article_full",
                "article_number": article["number"],
                "article_title": article["title"],
                "chapter": article["chapter"],
                "law": "gdpr"
            }
        )
        docs.append(doc)
        chunk_types["article_full"] += 1
        
        # Article paragraphs as separate chunks
        for paragraph in article["paragraphs"]:
            para_content = paragraph["text"]
            
            # Add subpoints to content if they exist
            if paragraph["subpoints"]:
                subpoints_text = " ".join([f"({sp['letter']}) {sp['text']}" for sp in paragraph["subpoints"]])
                para_content = f"{para_content} Subpoints: {subpoints_text}"
            
            doc = Document(
                page_content=f"Article {article['number']} ({article['title']}), Paragraph {paragraph['number']}: {para_content}",
                metadata={
                    "source": source_file,
                    "type": "article_paragraph",
                    "article_number": article["number"],
                    "article_title": article["title"],
                    "paragraph_number": paragraph["number"],
                    "chapter": article["chapter"],
                    "law": "gdpr"
                }
            )
            docs.append(doc)
            chunk_types["article_paragraph"] += 1
    
    logger.info(f"Created {len(docs)} document chunks in total:")
    logger.info(f"  - Chapter overviews: {chunk_types['chapter']}")
    logger.info(f"  - Full articles: {chunk_types['article_full']}")
    logger.info(f"  - Article paragraphs: {chunk_types['article_paragraph']}")
    
    return docs

def save_document_chunks_as_json(docs, output_path):
    """Save document chunks as JSON for reference."""
    serializable_docs = []
    
    for doc in docs:
        serializable_docs.append({
            "page_content": doc.page_content,
            "metadata": doc.metadata
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_docs, f, indent=2)
    
    logger.info(f"Saved {len(serializable_docs)} document chunks to {output_path}")

def process_gdpr():
    """Main function to process GDPR from JSON file."""
    start_time = time.time()
    logger.info("Starting GDPR processing from JSON file...")
    
    # Updated file path to use gdpr_main.json
    gdpr_json_file = os.path.join(RAW_DIR, "gdpr_main.json")
    logger.info(f"Reading from: {gdpr_json_file}")
    
    # Load JSON file
    with open(gdpr_json_file, 'r', encoding='utf-8') as f:
        gdpr_pages = json.load(f)
    
    logger.info(f"Loaded GDPR JSON with {len(gdpr_pages)} pages")
    
    # Log some statistics about the input data
    total_chars = sum(len(page["content"]) for page in gdpr_pages)
    logger.info(f"Total content size: {total_chars} characters")
    
    # Concatenate all pages into a single text
    logger.info("Concatenating pages into a single document...")
    full_text = ""
    for page in gdpr_pages:
        full_text += page["content"] + "\n\n"
    
    # Save clean text for reference
    clean_text_path = os.path.join(PROCESSED_DIR, "gdpr_clean.txt")
    with open(clean_text_path, 'w', encoding='utf-8') as f:
        f.write(full_text)
    logger.info(f"Clean text saved to: {clean_text_path}")
    
    # Extract structure
    structured_gdpr = extract_gdpr_structure(full_text)
    
    # Save structured data
    structured_path = os.path.join(PROCESSED_DIR, "gdpr_structured.json")
    with open(structured_path, 'w', encoding='utf-8') as f:
        json.dump(structured_gdpr, f, indent=2)
    logger.info(f"Structured data saved to: {structured_path}")
    
    # Create document chunks
    docs = create_document_chunks(structured_gdpr, "gdpr_main.json")
    
    # Save the final processed data that will be embedded
    final_json_path = os.path.join(PROCESSED_DIR, "gdpr_final.json")
    save_document_chunks_as_json(docs, final_json_path)
    
    # Create embeddings and store in vector DB
    logger.info("Creating embeddings and storing in vector database...")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key
    )

    # Create a separate DB for GDPR
    db = Chroma(
        persist_directory=GDPR_DB_DIR,
        embedding_function=embeddings,
        collection_name="gdpr_regulation"
    )
    
    # Add documents in batches with error handling and rate limiting
    batch_size = 25  # Smaller batch size to avoid rate limits
    logger.info(f"Adding documents to vector store in batches of {batch_size}...")
    
    total_batches = (len(docs) - 1) // batch_size + 1
    
    for i in tqdm(range(0, len(docs), batch_size)):
        batch = docs[i:i+batch_size]
        batch_num = i // batch_size + 1
        
        logger.info(f"Processing batch {batch_num}/{total_batches} with {len(batch)} documents")
        
        # Implement exponential backoff for API rate limits
        max_retries = 5
        retry_count = 0
        while retry_count < max_retries:
            try:
                db.add_documents(batch)
                logger.info(f"Successfully added batch {batch_num}/{total_batches} to vector store")
                break
            except Exception as e:
                retry_count += 1
                wait_time = 2 ** retry_count  # Exponential backoff
                logger.warning(f"Error adding batch {batch_num}, retrying in {wait_time}s: {str(e)}")
                time.sleep(wait_time)
                if retry_count == max_retries:
                    logger.error(f"Failed to add batch {batch_num} after {max_retries} retries: {str(e)}")
    
    # Log execution time
    execution_time = time.time() - start_time
    logger.info(f"GDPR processing complete in {execution_time:.2f} seconds")
    logger.info(f"Added {len(docs)} chunks to vector store at {GDPR_DB_DIR}")
    
    return db

if __name__ == "__main__":
    try:
        process_gdpr()
        logger.info("GDPR processing completed successfully")
    except Exception as e:
        logger.error(f"Error processing GDPR: {str(e)}", exc_info=True)