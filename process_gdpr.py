import os
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
PROCESS_GDPR_LOGS = os.path.join(DATA_DIR, "process_gdpr_logs")

# Set up logging with detailed formatting for better debugging
LOG_DIR = os.path.join(DATA_DIR, "process_gdpr_logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Create a timestamped log file to track this processing session
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f"enhanced_gdpr_processing_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs(GDPR_DB_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_enhanced_gdpr_json(file_path):
    """
    Load the enhanced GDPR JSON file with sophisticated structural metadata.
    
    This function reads your carefully crafted GDPR JSON that contains both content
    and rich structural metadata about the regulation's organization. We validate
    the structure to ensure it contains the enhanced metadata we expect for
    creating precise citations.
    """
    logger.info(f"Loading enhanced GDPR JSON file from: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Log document metadata for transparency and validation
        metadata = data.get('document', {}).get('metadata', {})
        logger.info(f"Document loaded: {metadata.get('official_title', 'Unknown title')}")
        logger.info(f"Source: {metadata.get('source', 'Unknown source')}")
        logger.info(f"Effective date: {metadata.get('effective_date', 'Unknown date')}")
        logger.info(f"Total chapters: {metadata.get('total_chapters', 'Unknown')}")
        logger.info(f"Total articles: {metadata.get('total_articles', 'Unknown')}")
        
        chunks = data.get('chunks', [])
        logger.info(f"Found {len(chunks)} pre-processed chunks in the document")
        
        # Validate that we have enhanced structural metadata in our chunks
        enhanced_chunks = 0
        for chunk in chunks:
            if chunk.get('metadata', {}).get('article_structure'):
                enhanced_chunks += 1
        
        logger.info(f"Enhanced structural metadata found in {enhanced_chunks}/{len(chunks)} chunks")
        
        if enhanced_chunks == 0:
            logger.warning("No enhanced structural metadata found - system will work but with limited precision")
        else:
            logger.info("Enhanced structural metadata validated successfully")
        
        return data, chunks
        
    except Exception as e:
        logger.error(f"Error loading enhanced GDPR JSON file: {str(e)}")
        raise

def flatten_article_structure(article_structure):
    """
    Intelligently flatten complex GDPR article structure metadata for vector database compatibility.
    
    This function applies the same "lossless flattening" approach we use for Polish law.
    We extract essential information as simple key-value pairs while preserving the
    complete structure for full access when needed. This solves the core challenge of
    representing sophisticated legal document organization within technical constraints.
    
    The beauty of this approach is that we don't lose any information - we just transform
    it into a format that works within vector database limitations while maintaining
    all the sophisticated capabilities your citation system needs.
    """
    logger.debug("Flattening complex GDPR article structure metadata...")
    
    # Initialize flattened structure with safe defaults
    flattened = {
        'has_enhanced_structure': False,
        'paragraph_count': 0,
        'has_sub_paragraphs': False,
        'numbering_style': '',
        'complexity_level': 'simple',  # simple, mixed, complex
        'article_structure_json': ''   # Complete structure preserved as string
    }
    
    # If no structure provided, return minimal metadata
    if not article_structure or not isinstance(article_structure, dict):
        logger.debug("No complex GDPR structure to flatten")
        return flattened
    
    try:
        # Extract basic structural indicators
        flattened['has_enhanced_structure'] = True
        flattened['paragraph_count'] = article_structure.get('paragraph_count', 0)
        
        # Analyze paragraph structure to understand GDPR-specific patterns
        paragraphs_info = article_structure.get('paragraphs', {})
        # Handle the case where paragraphs is explicitly set to null
        if paragraphs_info is None:
            paragraphs_info = {}
        has_any_sub_paragraphs = False
        numbering_styles = set()
        complexity_indicators = []
        
        for para_key, para_info in paragraphs_info.items():
            if isinstance(para_info, dict):
                # Check if this paragraph has sub-paragraphs (like (a), (b), (c) in GDPR)
                if para_info.get('has_sub_paragraphs', False):
                    has_any_sub_paragraphs = True
                    complexity_indicators.append('sub_paragraphs')
                    
                    # Collect GDPR-specific numbering styles (alphabetical vs numeric)
                    style = para_info.get('numbering_style', '')
                    if style:
                        numbering_styles.add(style)
                    
                    # Track sub-paragraph complexity for GDPR provisions
                    sub_count = para_info.get('sub_paragraph_count', 0)
                    if sub_count > 4:  # GDPR often has many sub-provisions
                        complexity_indicators.append('many_sub_paragraphs')
        
        # Store extracted indicators specific to GDPR structure
        flattened['has_sub_paragraphs'] = has_any_sub_paragraphs
        flattened['numbering_style'] = list(numbering_styles)[0] if numbering_styles else ''
        
        # Determine complexity level for quick filtering by GDPR agents
        if len(complexity_indicators) == 0:
            flattened['complexity_level'] = 'simple'
        elif len(complexity_indicators) <= 2:
            flattened['complexity_level'] = 'mixed'
        else:
            flattened['complexity_level'] = 'complex'
        
        # Most importantly: preserve complete GDPR structure as JSON string
        # This ensures no information is lost while maintaining compatibility
        flattened['article_structure_json'] = json.dumps(article_structure)
        
        logger.debug(f"Flattened GDPR structure: {flattened['paragraph_count']} paragraphs, "
                    f"sub-paragraphs: {has_any_sub_paragraphs}, "
                    f"style: {flattened['numbering_style']}, "
                    f"complexity: {flattened['complexity_level']}")
        
        return flattened
        
    except Exception as e:
        logger.warning(f"Error flattening GDPR article structure: {e}")
        # Return minimal structure to ensure processing continues
        flattened['article_structure_json'] = json.dumps(article_structure) if article_structure else ''
        return flattened

def create_documents_from_enhanced_gdpr_chunks(chunks, source_metadata):
    """
    Convert enhanced GDPR JSON chunks to LangChain Document objects with intelligent metadata flattening.
    
    This function mirrors the approach used for Polish law but adapts it specifically for
    GDPR's structure. We transform your sophisticated GDPR metadata into a format that
    Chroma can store while preserving all the information needed for precise citations
    like "Article 1, paragraph 2(c) (Chapter 1: General provisions)".
    
    The key insight is that we're creating documents that can "speak" both the vector
    database language (simple key-value pairs) and the sophisticated legal analysis
    language (complex nested structures) simultaneously.
    """
    logger.info("Converting enhanced GDPR JSON chunks to LangChain Document objects...")
    logger.info("Implementing intelligent GDPR metadata flattening for vector database compatibility...")
    
    docs = []
    processing_stats = {
        'total_chunks': len(chunks),
        'successful_conversions': 0,
        'enhanced_structure_count': 0,
        'chunk_types': {},
        'complexity_levels': {},
        'errors': 0
    }
    
    for i, chunk in enumerate(chunks):
        try:
            content = chunk.get('content', '')
            metadata = chunk.get('metadata', {})
            
            # Track chunk types for GDPR-specific statistical analysis
            chunk_type = metadata.get('type', 'unknown')
            processing_stats['chunk_types'][chunk_type] = processing_stats['chunk_types'].get(chunk_type, 0) + 1
            
            # Start building enhanced but flattened GDPR metadata
            # We preserve all essential GDPR information while ensuring compatibility
            enhanced_metadata = {
                # Basic GDPR document structure (always simple values)
                'type': metadata.get('type', ''),
                'chapter_number': metadata.get('chapter_number', ''),
                'chapter_title': metadata.get('chapter_title', ''),
                'section_number': metadata.get('section_number', ''),
                'section_title': metadata.get('section_title', ''),
                'article_number': metadata.get('article_number', ''),
                'article_title': metadata.get('article_title', ''),
                'page': metadata.get('page', ''),
                
                # GDPR-specific context for legal research
                'law': 'gdpr',
                'source': source_metadata.get('source', ''),
                'official_title': source_metadata.get('official_title', ''),
                'effective_date': source_metadata.get('effective_date', ''),
                'jurisdiction': source_metadata.get('jurisdiction', ''),
                'regulation_type': 'eu_regulation',
                
                # Processing metadata for debugging and optimization
                'chunk_index': i,
                'processing_timestamp': timestamp
            }
            
            # Handle enhanced GDPR article structure with intelligent flattening
            article_structure = metadata.get('article_structure', {})
            if article_structure:
                # Apply our sophisticated flattening algorithm adapted for GDPR
                flattened_structure = flatten_article_structure(article_structure)
                
                # Merge flattened structure into metadata
                enhanced_metadata.update(flattened_structure)
                
                # Track statistics about enhanced GDPR structures
                processing_stats['enhanced_structure_count'] += 1
                complexity = flattened_structure.get('complexity_level', 'unknown')
                processing_stats['complexity_levels'][complexity] = processing_stats['complexity_levels'].get(complexity, 0) + 1
                
                logger.debug(f"Enhanced GDPR chunk {i}: Article {enhanced_metadata.get('article_number', 'N/A')} "
                           f"with {flattened_structure.get('paragraph_count', 0)} paragraphs, "
                           f"complexity: {complexity}")
            else:
                # No enhanced structure - set basic indicators for GDPR
                enhanced_metadata.update({
                    'has_enhanced_structure': False,
                    'paragraph_count': 0,
                    'has_sub_paragraphs': False,
                    'numbering_style': '',
                    'complexity_level': 'simple',
                    'article_structure_json': ''
                })
            
            # Validate content quality before creating document
            if not content or not content.strip():
                logger.warning(f"Empty content in GDPR chunk {i}, skipping...")
                continue
            
            # Create the document with flattened but complete GDPR metadata
            doc = Document(
                page_content=content.strip(),
                metadata=enhanced_metadata
            )
            
            docs.append(doc)
            processing_stats['successful_conversions'] += 1
            
            # Log sample GDPR processing for verification
            if i < 3:
                has_structure = enhanced_metadata.get('has_enhanced_structure', False)
                logger.info(f"Sample GDPR chunk {i}: {chunk_type} - "
                          f"Article {enhanced_metadata.get('article_number', 'N/A')} - "
                          f"Enhanced: {'✓' if has_structure else '✗'} - "
                          f"Complexity: {enhanced_metadata.get('complexity_level', 'unknown')} - "
                          f"Content: {len(content)} chars")
                
        except Exception as e:
            logger.error(f"Error processing GDPR chunk {i}: {str(e)}")
            processing_stats['errors'] += 1
            continue
    
    # Log comprehensive GDPR processing statistics
    logger.info("=" * 60)
    logger.info("ENHANCED GDPR METADATA PROCESSING COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Total chunks processed: {processing_stats['total_chunks']}")
    logger.info(f"Successful conversions: {processing_stats['successful_conversions']}")
    logger.info(f"Enhanced structures: {processing_stats['enhanced_structure_count']}")
    logger.info(f"Processing errors: {processing_stats['errors']}")
    
    logger.info("GDPR chunk type distribution:")
    for chunk_type, count in sorted(processing_stats['chunk_types'].items()):
        logger.info(f"  - {chunk_type}: {count} chunks")
    
    logger.info("GDPR complexity level distribution:")
    for complexity, count in sorted(processing_stats['complexity_levels'].items()):
        logger.info(f"  - {complexity}: {count} chunks")
    
    enhancement_rate = (processing_stats['enhanced_structure_count'] / processing_stats['successful_conversions'] * 100) if processing_stats['successful_conversions'] > 0 else 0
    logger.info(f"GDPR enhanced structure rate: {enhancement_rate:.1f}%")
    
    return docs

def embed_gdpr_documents_with_flattened_metadata(docs):
    """
    Embed GDPR documents into the Chroma vector store using flattened metadata.
    
    This function handles the GDPR embedding process with our proven flattened metadata
    approach. The flattened metadata is now compatible with Chroma's constraints while
    preserving all the sophisticated structural information your GDPR citation system needs.
    
    We include comprehensive error handling and logging to ensure you can monitor
    the success of our metadata flattening solution for GDPR specifically.
    """
    logger.info("Creating GDPR embeddings and storing in vector database with flattened metadata...")
    logger.info("Testing GDPR compatibility with vector database metadata constraints...")
    
    # Initialize embeddings with the same model used throughout your system
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=api_key
    )
    
    # Create or connect to the GDPR vector store with enhanced error detection
    db = Chroma(
        persist_directory=GDPR_DB_DIR,
        embedding_function=embeddings,
        collection_name="gdpr_regulation"
    )
    
    # Clear existing collection to ensure clean start with new GDPR metadata format
    try:
        existing_count = db._collection.count()
        if existing_count > 0:
            logger.info(f"Clearing existing GDPR collection with {existing_count} documents...")
            db._collection.delete(where={})
            logger.info("GDPR collection cleared successfully")
    except Exception as e:
        logger.info(f"No existing GDPR collection found or error checking: {e}")
    
    # Test GDPR metadata compatibility with a small sample first
    if docs:
        logger.info("Testing GDPR metadata compatibility with sample document...")
        try:
            test_doc = docs[0]
            # Attempt to add one GDPR document to test our flattening approach
            db.add_documents([test_doc])
            logger.info("✅ GDPR metadata compatibility test successful - flattening approach works!")
            
            # Clear the test document
            try:
                all_docs = db._collection.get()
                if all_docs['ids']:
                    db._collection.delete(ids=all_docs['ids'])
                    logger.info("GDPR test collection cleared successfully")
            except Exception as e:
                logger.info(f"GDPR collection was already empty or error clearing: {e}")

        except Exception as e:
            logger.error(f"❌ GDPR metadata compatibility test failed: {e}")
            logger.error("This suggests our GDPR flattening approach needs adjustment")
            raise Exception(f"GDPR metadata flattening failed compatibility test: {e}")
    
    # Proceed with full GDPR embedding using proven compatible metadata
    batch_size = 25  # Conservative batch size for reliability
    logger.info(f"Adding {len(docs)} GDPR documents to vector store in batches of {batch_size}...")
    
    total_batches = (len(docs) - 1) // batch_size + 1
    successful_batches = 0
    total_embedded = 0
    metadata_errors = 0
    
    for i in tqdm(range(0, len(docs), batch_size), desc="Embedding enhanced GDPR documents"):
        batch = docs[i:i+batch_size]
        batch_num = i // batch_size + 1
        
        logger.info(f"Processing GDPR batch {batch_num}/{total_batches} with {len(batch)} documents")
        
        # Log sample of enhanced GDPR metadata being processed
        sample_articles = []
        for doc in batch[:3]:
            article_num = doc.metadata.get('article_number', 'N/A')
            chunk_type = doc.metadata.get('type', 'unknown')
            has_enhanced = doc.metadata.get('has_enhanced_structure', False)
            complexity = doc.metadata.get('complexity_level', 'unknown')
            sample_articles.append(f"Article {article_num} ({chunk_type}, {'enhanced' if has_enhanced else 'basic'}, {complexity})")
        
        logger.info(f"GDPR batch {batch_num} sample: {', '.join(sample_articles)}")
        
        # Implement exponential backoff for API rate limits
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                db.add_documents(batch)
                logger.info(f"✅ Successfully embedded GDPR batch {batch_num}/{total_batches}")
                successful_batches += 1
                total_embedded += len(batch)
                break
                
            except Exception as e:
                retry_count += 1
                wait_time = 2 ** retry_count
                
                # Check if this is a GDPR metadata-related error
                if "metadata" in str(e).lower():
                    metadata_errors += 1
                    logger.error(f"❌ GDPR metadata error in batch {batch_num}: {str(e)}")
                    logger.error("This suggests our GDPR flattening approach may have missed some complex metadata")
                    break  # Don't retry metadata errors
                else:
                    logger.warning(f"⚠️  Error embedding GDPR batch {batch_num}, retry {retry_count}/{max_retries} "
                                 f"in {wait_time}s: {str(e)}")
                    time.sleep(wait_time)
                
                if retry_count == max_retries:
                    logger.error(f"❌ Failed to embed GDPR batch {batch_num} after {max_retries} retries")
    
    # Comprehensive final GDPR statistics and validation
    logger.info("=" * 60)
    logger.info("ENHANCED GDPR EMBEDDING PROCESS COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Successful batches: {successful_batches}/{total_batches}")
    logger.info(f"GDPR documents embedded: {total_embedded}/{len(docs)}")
    logger.info(f"GDPR metadata errors: {metadata_errors}")
    
    # Verify the GDPR vector store contains our enhanced documents
    try:
        final_count = db._collection.count()
        logger.info(f"GDPR vector store final count: {final_count} documents")
        
        # Test retrieval of enhanced GDPR metadata
        if final_count > 0:
            test_docs = db.similarity_search("Article 1", k=1)
            if test_docs:
                test_metadata = test_docs[0].metadata
                has_enhanced = test_metadata.get('has_enhanced_structure', False)
                logger.info(f"GDPR verification: Retrieved document has enhanced structure: {'✓' if has_enhanced else '✗'}")
                
                if has_enhanced:
                    logger.info(f"GDPR enhanced metadata preserved: paragraph_count={test_metadata.get('paragraph_count', 0)}, "
                               f"complexity={test_metadata.get('complexity_level', 'unknown')}")
            
    except Exception as e:
        logger.warning(f"Could not verify final GDPR embedding results: {e}")
    
    if metadata_errors > 0:
        logger.warning(f"⚠️  {metadata_errors} GDPR metadata errors occurred - review flattening approach")
    else:
        logger.info("✅ No GDPR metadata errors - flattening approach completely successful!")
    
    return db

def save_enhanced_gdpr_processing_summary(docs, output_path):
    """
    Save a comprehensive summary of enhanced GDPR document processing.
    
    This summary helps you understand how well the enhanced metadata flattening
    worked for GDPR and provides statistics about the structural complexity of your
    processed GDPR regulation documents.
    """
    logger.info(f"Saving enhanced GDPR processing summary to: {output_path}")
    
    summary = {
        "processing_timestamp": datetime.now().isoformat(),
        "document_type": "gdpr_regulation",
        "total_documents": len(docs),
        "enhanced_metadata_stats": {
            "enhanced_structure_count": 0,
            "simple_documents": 0,
            "complexity_distribution": {},
            "numbering_styles": {},
            "paragraph_statistics": {
                "min_paragraphs": float('inf'),
                "max_paragraphs": 0,
                "total_paragraphs": 0,
                "articles_with_sub_paragraphs": 0
            }
        },
        "document_types": {},
        "articles_processed": set(),
        "chapters_processed": set(),
        "sample_enhanced_documents": []
    }
    
    # Analyze enhanced GDPR metadata in processed documents
    for i, doc in enumerate(docs):
        metadata = doc.metadata
        
        # Basic GDPR document analysis
        doc_type = metadata.get('type', 'unknown')
        summary["document_types"][doc_type] = summary["document_types"].get(doc_type, 0) + 1
        
        if metadata.get('article_number'):
            summary["articles_processed"].add(metadata['article_number'])
        if metadata.get('chapter_number'):
            summary["chapters_processed"].add(metadata['chapter_number'])
        
        # Enhanced GDPR metadata analysis
        if metadata.get('has_enhanced_structure', False):
            summary["enhanced_metadata_stats"]["enhanced_structure_count"] += 1
            
            # GDPR complexity analysis
            complexity = metadata.get('complexity_level', 'unknown')
            summary["enhanced_metadata_stats"]["complexity_distribution"][complexity] = \
                summary["enhanced_metadata_stats"]["complexity_distribution"].get(complexity, 0) + 1
            
            # GDPR numbering style analysis (alphabetical vs numeric)
            numbering_style = metadata.get('numbering_style', '')
            if numbering_style:
                summary["enhanced_metadata_stats"]["numbering_styles"][numbering_style] = \
                    summary["enhanced_metadata_stats"]["numbering_styles"].get(numbering_style, 0) + 1
            
            # GDPR paragraph statistics
            para_count = metadata.get('paragraph_count', 0)
            if para_count > 0:
                stats = summary["enhanced_metadata_stats"]["paragraph_statistics"]
                stats["min_paragraphs"] = min(stats["min_paragraphs"], para_count)
                stats["max_paragraphs"] = max(stats["max_paragraphs"], para_count)
                stats["total_paragraphs"] += para_count
                
                if metadata.get('has_sub_paragraphs', False):
                    stats["articles_with_sub_paragraphs"] += 1
            
            # Save sample enhanced GDPR documents
            if len(summary["sample_enhanced_documents"]) < 3:
                summary["sample_enhanced_documents"].append({
                    "article_number": metadata.get('article_number', 'N/A'),
                    "content_preview": doc.page_content[:150] + "...",
                    "enhanced_metadata": {
                        "paragraph_count": metadata.get('paragraph_count', 0),
                        "has_sub_paragraphs": metadata.get('has_sub_paragraphs', False),
                        "complexity_level": metadata.get('complexity_level', 'unknown'),
                        "numbering_style": metadata.get('numbering_style', '')
                    }
                })
        else:
            summary["enhanced_metadata_stats"]["simple_documents"] += 1
    
    # Calculate final GDPR statistics
    if summary["enhanced_metadata_stats"]["paragraph_statistics"]["min_paragraphs"] == float('inf'):
        summary["enhanced_metadata_stats"]["paragraph_statistics"]["min_paragraphs"] = 0
    
    # Convert sets to sorted lists for JSON serialization
    summary["articles_processed"] = sorted(list(summary["articles_processed"]))
    summary["chapters_processed"] = sorted(list(summary["chapters_processed"]))
    
    # Calculate GDPR enhancement rate
    total_docs = summary["total_documents"]
    enhanced_docs = summary["enhanced_metadata_stats"]["enhanced_structure_count"]
    enhancement_rate = (enhanced_docs / total_docs * 100) if total_docs > 0 else 0
    summary["enhancement_rate_percent"] = round(enhancement_rate, 1)
    
    # Save comprehensive GDPR summary
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Enhanced GDPR processing summary saved:")
    logger.info(f"  - Total documents: {total_docs}")
    logger.info(f"  - Enhanced documents: {enhanced_docs} ({enhancement_rate:.1f}%)")
    logger.info(f"  - Articles processed: {len(summary['articles_processed'])}")
    logger.info(f"  - Chapters processed: {len(summary['chapters_processed'])}")

def process_enhanced_gdpr():
    """
    Main function to process enhanced GDPR JSON with sophisticated metadata flattening.
    
    This represents the complete solution adapted for GDPR, taking your enhanced JSON
    structure and transforming it into a format that works within vector database
    constraints while preserving all the sophisticated capabilities your legal
    analysis system needs for uniform citation creation.
    """
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("STARTING ENHANCED GDPR PROCESSING WITH METADATA FLATTENING")
    logger.info("=" * 80)
    
    # Updated path to look for your enhanced GDPR JSON structure
    gdpr_json_file = os.path.join(PROCESSED_DIR, "gdpr_final_manual.json")
    
    if not os.path.exists(gdpr_json_file):
        # Also check RAW_DIR as backup location
        backup_path = os.path.join(RAW_DIR, "gdpr_final_manual.json")
        if os.path.exists(backup_path):
            gdpr_json_file = backup_path
            logger.info(f"Using backup location: {backup_path}")
        else:
            logger.error(f"Enhanced GDPR JSON file not found in either location:")
            logger.error(f"  Primary: {os.path.join(PROCESSED_DIR, 'gdpr_final_manual.json')}")
            logger.error(f"  Backup: {backup_path}")
            raise FileNotFoundError(f"Required enhanced GDPR JSON file not found")
    
    try:
        # Step 1: Load enhanced GDPR JSON with structural validation
        logger.info("STEP 1: Loading enhanced GDPR JSON with structural validation...")
        data, chunks = load_enhanced_gdpr_json(gdpr_json_file)
        source_metadata = data.get('document', {}).get('metadata', {})
        
        # Step 2: Convert chunks with intelligent GDPR metadata flattening
        logger.info("STEP 2: Converting GDPR chunks with intelligent metadata flattening...")
        docs = create_documents_from_enhanced_gdpr_chunks(chunks, source_metadata)
        
        if not docs:
            logger.error("No valid documents created from enhanced GDPR chunks")
            return None
        
        # Step 3: Embed GDPR documents with flattened but complete metadata
        logger.info("STEP 3: Embedding GDPR documents with flattened metadata...")
        db = embed_gdpr_documents_with_flattened_metadata(docs)
        
        # Step 4: Save comprehensive GDPR processing summary
        logger.info("STEP 4: Saving comprehensive GDPR processing summary...")
        summary_path = os.path.join(PROCESS_GDPR_LOGS, f"enhanced_gdpr_processing_summary_{timestamp}.json")
        save_enhanced_gdpr_processing_summary(docs, summary_path)
        
        # Final completion statistics
        execution_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info("ENHANCED GDPR PROCESSING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Execution time: {execution_time:.2f} seconds")
        logger.info(f"GDPR documents processed: {len(docs)}")
        logger.info(f"Vector store location: {GDPR_DB_DIR}")
        logger.info(f"Enhanced log file: {log_file}")
        logger.info(f"Processing summary: {summary_path}")
        logger.info("Enhanced GDPR metadata flattening approach successful!")
        logger.info("Vector database compatibility achieved while preserving all GDPR functionality!")
        logger.info("=" * 80)
        
        return db
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error("=" * 80)
        logger.error("ENHANCED GDPR PROCESSING FAILED")
        logger.error(f"Error after {execution_time:.2f} seconds: {str(e)}")
        logger.error("=" * 80)
        raise

if __name__ == "__main__":
    try:
        process_enhanced_gdpr()
        logger.info("Enhanced GDPR processing completed successfully!")
    except Exception as e:
        logger.error(f"Fatal error in enhanced GDPR processing: {str(e)}")
        exit(1)