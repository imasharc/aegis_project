import os
import re
import json
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

# Check for API key
import os
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Define paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PL_DB_DIR = os.path.join(DATA_DIR, "polish_law_db")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
LOG_DIR = os.path.join(DATA_DIR, "logs")

os.makedirs(PL_DB_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def extract_text_from_pdf(file_path):
    """Extract text from PDF using PyPDF2 with improved cleaning."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    # Improved text cleaning
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'Journal of Laws – \d+ – Item \d+', '', text)  # Remove page headers
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'(\w)\s+(\w)', r'\1 \2', text)  # Fix spaced words
    text = re.sub(r'(\w)\s+([.,;:])', r'\1\2', text)  # Fix spaced punctuation
    
    return text

def ensure_complete_sentences(text):
    """Ensure text ends with proper sentence termination."""
    # If text ends with incomplete sentence, add ellipsis
    if text and text.strip() and not re.search(r'[.!?;]\s*$', text.strip()):
        text = text.strip() + "..."
    return text.strip()

def extract_polish_law_structure(text):
    """Extract chapters, articles, paragraphs from Polish law text with improved patterns."""
    # Extract chapters with improved pattern
    chapter_pattern = re.compile(r'Chapter (\d+)\s*\n?\s*([^\n]+)')
    chapters = chapter_pattern.findall(text)
    
    # Extract articles with improved pattern to capture complete content
    article_pattern = re.compile(r'Article (\d+[a-z]?)\.?\s*(.+?)(?=Article \d+[a-z]?\.|\Z)', re.DOTALL)
    articles = article_pattern.findall(text)
    
    # Structure the document
    structured_law = {
        "title": "Act on the Protection of Personal Data",
        "chapters": [],
        "articles": []
    }
    
    # Process chapters
    for ch_num, ch_title in chapters:
        structured_law["chapters"].append({
            "number": ch_num,
            "title": ch_title.strip(),
        })
    
    # Process articles with their paragraphs
    for art_num, art_content in articles:
        # Improved paragraph pattern to capture numbered sections
        paragraph_pattern = re.compile(r'(\d+)\.\s*([^0-9]+?)(?=\d+\.\s+|\Z)', re.DOTALL)
        paragraphs = paragraph_pattern.findall(art_content)
        
        paragraphs_structured = []
        for para_num, para_text in paragraphs:
            # Clean paragraph text
            clean_para_text = para_text.strip()
            clean_para_text = ensure_complete_sentences(clean_para_text)
            
            # Extract subpoints if any (patterns like a), b), c))
            subpoint_pattern = re.compile(r'([a-z])\)\s*([^a-z\)]+?)(?=[a-z]\)|\Z)', re.DOTALL)
            subpoints = subpoint_pattern.findall(clean_para_text)
            
            subpoints_structured = []
            for sp_letter, sp_text in subpoints:
                sp_text = sp_text.strip()
                sp_text = ensure_complete_sentences(sp_text)
                subpoints_structured.append({
                    "letter": sp_letter,
                    "text": sp_text
                })
            
            paragraphs_structured.append({
                "number": para_num,
                "text": clean_para_text,
                "subpoints": subpoints_structured
            })
        
        # Clean article content
        art_content = art_content.strip()
        art_content = ensure_complete_sentences(art_content)
        
        # Find chapter this article belongs to
        article_chapter = None
        for ch in structured_law["chapters"]:
            if f"Chapter {ch['number']}" in text[:text.find(f"Article {art_num}")]:
                article_chapter = ch["number"]
                break
        
        structured_law["articles"].append({
            "number": art_num,
            "content": art_content,
            "paragraphs": paragraphs_structured,
            "chapter": article_chapter
        })
    
    return structured_law

def map_polish_to_gdpr(article_num):
    """Map Polish law articles to corresponding GDPR articles."""
    # Preliminary mapping based on content analysis
    mapping = {
        "1": "1, 2 (subject matter and scope)",
        "2": "85 (processing and freedom of expression)",
        "3": "13 (information provision)",
        "4": "14 (information when data not obtained from subject)",
        "5": "15 (right of access)",
        "6": "2, 3 (material and territorial scope)",
        "7": "77, 78 (right to lodge a complaint)",
        "8": "37, 38, 39 (data protection officer)",
        "9": "51-54 (supervisory authority)",
        "10": "37, 38 (designation of data protection officer)",
        "11": "38 (position of data protection officer)",
        "12": "43 (certification bodies)",
        "13": "42 (certification)",
        "15": "42 (certification)",
        "27": "40 (codes of conduct)",
        "28": "41 (monitoring of approved codes)",
        "29": "41 (accreditation of monitoring bodies)",
        "34": "51, 52 (supervisory authority)",
        "35": "35 (data protection impact assessment)",
        "57": "36 (prior consultation)",
        "58": "58 (investigative powers)",
        "69": "83 (administrative fines)",
        "70": "58 (corrective powers)",
        "71": "65 (dispute resolution)",
        "83": "33, 34 (data breach notification)",
        "100": "83 (administrative fines)",
        "101": "83 (administrative fines)",
        "102": "83 (general conditions for fines)"
    }
    return mapping.get(article_num, "")

def create_document_chunks(structured_law, source_file):
    """Create document chunks for embedding with improved metadata."""
    docs = []
    
    # Create chapter overview documents
    for chapter in structured_law["chapters"]:
        doc = Document(
            page_content=f"Chapter {chapter['number']}: {chapter['title']}",
            metadata={
                "source": source_file,
                "type": "chapter",
                "chapter_number": chapter["number"],
                "law": "polish_data_protection",
                "gdpr_mapping": ""  # Chapters don't map directly to GDPR
            }
        )
        docs.append(doc)
    
    # Create article documents
    for article in structured_law["articles"]:
        # Map to GDPR articles
        gdpr_mapping = map_polish_to_gdpr(article["number"])
        
        # Full article
        doc = Document(
            page_content=f"Article {article['number']}: {article['content']}",
            metadata={
                "source": source_file,
                "type": "article_full",
                "article_number": article["number"],
                "chapter": article["chapter"],
                "law": "polish_data_protection",
                "gdpr_mapping": gdpr_mapping
            }
        )
        docs.append(doc)
        
        # Article paragraphs
        for paragraph in article["paragraphs"]:
            para_content = paragraph["text"]
            
            # Add subpoints to content if they exist
            if paragraph["subpoints"]:
                for sp in paragraph["subpoints"]:
                    para_content += f" Subpoint {sp['letter']}) {sp['text']}"
            
            doc = Document(
                page_content=f"Article {article['number']}, Paragraph {paragraph['number']}: {para_content}",
                metadata={
                    "source": source_file,
                    "type": "article_paragraph",
                    "article_number": article["number"],
                    "paragraph_number": paragraph["number"],
                    "chapter": article["chapter"],
                    "law": "polish_data_protection",
                    "gdpr_mapping": gdpr_mapping
                }
            )
            docs.append(doc)
    
    return docs

def process_polish_law():
    """Main function to process Polish law document."""
    polish_law_file = os.path.join(RAW_DIR, "Act on the Protection of Personal Data_May 2018.pdf")
    
    print(f"Processing: {polish_law_file}")
    
    # Extract text
    text = extract_text_from_pdf(polish_law_file)
    
    # Save clean text for reference
    clean_text_path = os.path.join(PROCESSED_DIR, "polish_law_clean.txt")
    with open(clean_text_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Clean text saved to: {clean_text_path}")
    
    # Extract structure
    structured_law = extract_polish_law_structure(text)
    
    # Save structured data
    structured_path = os.path.join(PROCESSED_DIR, "polish_law_structured.json")
    with open(structured_path, 'w', encoding='utf-8') as f:
        json.dump(structured_law, f, indent=2)
    print(f"Structured data saved to: {structured_path}")
    
    # Create document chunks
    docs = create_document_chunks(structured_law, "Act on the Protection of Personal Data_May 2018.pdf")
    print(f"Created {len(docs)} document chunks")
    
    # Create embeddings and store in vector DB
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key
    )

    # Create a separate DB for Polish law
    db = Chroma(
        persist_directory=PL_DB_DIR,
        embedding_function=embeddings,
        collection_name="polish_data_protection_law"
    )
    
    # Add documents in batches
    batch_size = 100
    for i in tqdm(range(0, len(docs), batch_size)):
        batch = docs[i:i+batch_size]
        db.add_documents(batch)
    
    print(f"Polish law processing complete. Added {len(docs)} chunks to vector store.")
    return db

if __name__ == "__main__":
    process_polish_law()