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
INTERNAL_SEC_DB_DIR = os.path.join(DATA_DIR, "internal_security_db")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
PROCESS_INTERNAL_SEC_LOGS = os.path.join(DATA_DIR, "process_internal_security_logs")

# Set up comprehensive logging for internal security procedure processing
LOG_DIR = os.path.join(DATA_DIR, "process_internal_security_logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Create timestamped log file to track this processing session
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f"enhanced_internal_security_processing_{timestamp}.log")

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
os.makedirs(INTERNAL_SEC_DB_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_enhanced_internal_security_json(file_path):
    """
    Load the enhanced internal security procedures JSON file with sophisticated procedural metadata.
    
    This function reads your carefully structured internal security procedures that contain
    both procedural content and rich implementation metadata. We validate the structure
    to ensure it contains the procedural information our security agent needs for precise
    procedure identification and citation.
    """
    logger.info(f"Loading enhanced internal security JSON file from: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Log document metadata for transparency and validation
        doc_metadata = data.get('document_metadata', {})
        logger.info(f"Document loaded: {doc_metadata.get('document_title', 'Unknown title')}")
        logger.info(f"Document ID: {doc_metadata.get('document_id', 'Unknown ID')}")
        logger.info(f"Version: {doc_metadata.get('version', 'Unknown version')}")
        logger.info(f"Last updated: {doc_metadata.get('last_updated', 'Unknown date')}")
        logger.info(f"Total sections: {doc_metadata.get('total_sections', 'Unknown')}")
        logger.info(f"Total procedures: {doc_metadata.get('total_procedures', 'Unknown')}")
        
        sections = data.get('sections', [])
        logger.info(f"Found {len(sections)} procedural sections in the document")
        
        # Validate enhanced procedural metadata in sections
        enhanced_sections = 0
        procedure_count = 0
        for section in sections:
            section_metadata = section.get('metadata', {})
            if section_metadata.get('implementation_steps'):
                enhanced_sections += 1
                steps = section_metadata.get('implementation_steps', [])
                procedure_count += len(steps)
        
        logger.info(f"Enhanced procedural metadata found in {enhanced_sections}/{len(sections)} sections")
        logger.info(f"Total implementation steps identified: {procedure_count}")
        
        if enhanced_sections == 0:
            logger.warning("No enhanced procedural metadata found - system will work but with limited precision")
        else:
            logger.info("Enhanced procedural metadata validated successfully")
        
        return data, sections
        
    except Exception as e:
        logger.error(f"Error loading enhanced internal security JSON file: {str(e)}")
        raise

def flatten_procedure_structure(procedure_metadata):
    """
    Intelligently flatten complex procedure structure metadata for vector database compatibility.
    
    This function solves a similar challenge to the Polish law system but for internal procedures:
    your sophisticated nested procedure metadata is perfect for representing implementation
    workflows, but vector databases need simple key-value pairs. We extract essential
    procedural information while preserving complete structure for full access when needed.
    
    This demonstrates "procedural metadata flattening" - transforming complex implementation
    steps into searchable, simple metadata while maintaining all information for precise citations.
    """
    logger.debug("Flattening complex procedure structure metadata...")
    
    # Initialize flattened structure with safe defaults for procedures
    flattened = {
        'has_enhanced_procedure': False,
        'implementation_step_count': 0,
        'has_sub_steps': False,
        'required_tools_count': 0,
        'procedure_complexity': 'simple',  # simple, moderate, complex
        'responsible_roles_count': 0,
        'procedure_structure_json': ''   # Complete structure preserved as string
    }
    
    # If no procedure structure provided, return minimal metadata
    if not procedure_metadata or not isinstance(procedure_metadata, dict):
        logger.debug("No complex procedure structure to flatten")
        return flattened
    
    try:
        # Check if this section has implementation steps (indicates enhanced procedure)
        implementation_steps = procedure_metadata.get('implementation_steps', [])
        if implementation_steps and len(implementation_steps) > 0:
            flattened['has_enhanced_procedure'] = True
            flattened['implementation_step_count'] = len(implementation_steps)
            
            # Analyze procedural complexity indicators
            complexity_indicators = []
            total_tools = set()
            has_sub_steps = False
            
            for step in implementation_steps:
                if isinstance(step, dict):
                    # Check for sub-step complexity
                    if any(key.startswith('step_') and isinstance(step.get(key), dict) 
                          for key in step.keys()):
                        has_sub_steps = True
                        complexity_indicators.append('sub_steps')
                    
                    # Collect required tools
                    step_tools = step.get('required_tools', [])
                    if isinstance(step_tools, list):
                        total_tools.update(step_tools)
                    elif isinstance(step_tools, str):
                        total_tools.add(step_tools)
                    
                    # Check for complex configurations
                    if step.get('configuration_by_level') or step.get('configuration_settings'):
                        complexity_indicators.append('complex_configuration')
                    
                    # Check for automation and monitoring
                    if step.get('automation_tools') or step.get('monitoring'):
                        complexity_indicators.append('automation_monitoring')
            
            # Store procedural indicators
            flattened['has_sub_steps'] = has_sub_steps
            flattened['required_tools_count'] = len(total_tools)
            
            # Collect responsible roles
            responsible_roles = set()
            if isinstance(procedure_metadata.get('responsible_roles'), list):
                responsible_roles.update(procedure_metadata['responsible_roles'])
            flattened['responsible_roles_count'] = len(responsible_roles)
            
            # Determine procedure complexity for quick filtering
            if len(complexity_indicators) == 0 and len(implementation_steps) <= 2:
                flattened['procedure_complexity'] = 'simple'
            elif len(complexity_indicators) <= 2 and len(implementation_steps) <= 5:
                flattened['procedure_complexity'] = 'moderate'
            else:
                flattened['procedure_complexity'] = 'complex'
            
            logger.debug(f"Analyzed procedure: {len(implementation_steps)} steps, "
                        f"tools: {len(total_tools)}, complexity: {flattened['procedure_complexity']}")
        
        # Most importantly: preserve complete structure as JSON string
        # This ensures no procedural information is lost while maintaining compatibility
        flattened['procedure_structure_json'] = json.dumps(procedure_metadata)
        
        logger.debug(f"Flattened procedure: {flattened['implementation_step_count']} steps, "
                    f"tools: {flattened['required_tools_count']}, "
                    f"complexity: {flattened['procedure_complexity']}")
        
        return flattened
        
    except Exception as e:
        logger.warning(f"Error flattening procedure structure: {e}")
        # Return minimal structure to ensure processing continues
        flattened['procedure_structure_json'] = json.dumps(procedure_metadata) if procedure_metadata else ''
        return flattened

def create_documents_from_enhanced_security_sections(sections, source_metadata):
    """
    Convert enhanced internal security sections to LangChain Document objects with intelligent metadata flattening.
    
    This function adapts the sophisticated metadata flattening approach for internal security
    procedures. Unlike legal documents which have articles and paragraphs, security procedures
    have sections, procedures, and implementation steps. We preserve this organizational
    structure while making it compatible with vector database constraints.
    
    The approach creates "bilingual documents" that speak both vector database language
    (simple key-value pairs) and sophisticated procedural language (complex implementation workflows).
    """
    logger.info("Converting enhanced security sections to LangChain Document objects...")
    logger.info("Implementing intelligent procedural metadata flattening for vector database compatibility...")
    
    docs = []
    processing_stats = {
        'total_sections': len(sections),
        'successful_conversions': 0,
        'enhanced_procedure_count': 0,
        'section_types': {},
        'complexity_levels': {},
        'errors': 0
    }
    
    for i, section in enumerate(sections):
        try:
            content = section.get('content', '')
            metadata = section.get('metadata', {})
            
            # Track section types for statistical analysis
            section_type = metadata.get('type', 'unknown')
            processing_stats['section_types'][section_type] = processing_stats['section_types'].get(section_type, 0) + 1
            
            # Start building enhanced but flattened metadata for security procedures
            # We preserve all essential procedural information while ensuring compatibility
            enhanced_metadata = {
                # Basic procedural structure (always simple values)
                'type': metadata.get('type', ''),
                'section_number': metadata.get('section_number', ''),
                'section_title': metadata.get('section_title', ''),
                'procedure_number': metadata.get('procedure_number', ''),
                'procedure_title': metadata.get('procedure_title', ''),
                'policy_reference': metadata.get('policy_reference', ''),
                'subsection_count': metadata.get('subsection_count', 0),
                
                # Document-level context for security research
                'document_type': 'internal_security_procedures',
                'document_title': source_metadata.get('document_title', ''),
                'document_id': source_metadata.get('document_id', ''),
                'version': source_metadata.get('version', ''),
                'last_updated': source_metadata.get('last_updated', ''),
                'approved_by': source_metadata.get('approved_by', ''),
                'classification_level': source_metadata.get('classification_level', ''),
                
                # Processing metadata for debugging and optimization
                'section_index': i,
                'processing_timestamp': timestamp
            }
            
            # Handle enhanced procedure structure with intelligent flattening
            if metadata:  # All sections have some metadata
                # Apply sophisticated flattening algorithm for procedures
                flattened_procedure = flatten_procedure_structure(metadata)
                
                # Merge flattened structure into metadata
                enhanced_metadata.update(flattened_procedure)
                
                # Track statistics about enhanced procedures
                if flattened_procedure.get('has_enhanced_procedure', False):
                    processing_stats['enhanced_procedure_count'] += 1
                    complexity = flattened_procedure.get('procedure_complexity', 'unknown')
                    processing_stats['complexity_levels'][complexity] = processing_stats['complexity_levels'].get(complexity, 0) + 1
                    
                    logger.debug(f"Enhanced section {i}: {section_type} - "
                               f"Procedure {enhanced_metadata.get('procedure_number', 'N/A')} "
                               f"with {flattened_procedure.get('implementation_step_count', 0)} steps, "
                               f"complexity: {complexity}")
                else:
                    # No enhanced procedure structure - set basic indicators
                    enhanced_metadata.update({
                        'has_enhanced_procedure': False,
                        'implementation_step_count': 0,
                        'has_sub_steps': False,
                        'required_tools_count': 0,
                        'procedure_complexity': 'simple',
                        'responsible_roles_count': 0,
                        'procedure_structure_json': ''
                    })
            
            # Validate content quality before creating document
            if not content or not content.strip():
                logger.warning(f"Empty content in section {i}, skipping...")
                continue
            
            # Create the document with flattened but complete metadata
            doc = Document(
                page_content=content.strip(),
                metadata=enhanced_metadata
            )
            
            docs.append(doc)
            processing_stats['successful_conversions'] += 1
            
            # Log sample processing for verification
            if i < 3:
                has_procedure = enhanced_metadata.get('has_enhanced_procedure', False)
                logger.info(f"Sample section {i}: {section_type} - "
                          f"Procedure {enhanced_metadata.get('procedure_number', 'N/A')} - "
                          f"Enhanced: {'✓' if has_procedure else '✗'} - "
                          f"Complexity: {enhanced_metadata.get('procedure_complexity', 'unknown')} - "
                          f"Steps: {enhanced_metadata.get('implementation_step_count', 0)} - "
                          f"Content: {len(content)} chars")
                
        except Exception as e:
            logger.error(f"Error processing section {i}: {str(e)}")
            processing_stats['errors'] += 1
            continue
    
    # Log comprehensive processing statistics
    logger.info("=" * 60)
    logger.info("ENHANCED PROCEDURAL METADATA PROCESSING COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Total sections processed: {processing_stats['total_sections']}")
    logger.info(f"Successful conversions: {processing_stats['successful_conversions']}")
    logger.info(f"Enhanced procedures: {processing_stats['enhanced_procedure_count']}")
    logger.info(f"Processing errors: {processing_stats['errors']}")
    
    logger.info("Section type distribution:")
    for section_type, count in sorted(processing_stats['section_types'].items()):
        logger.info(f"  - {section_type}: {count} sections")
    
    logger.info("Procedure complexity distribution:")
    for complexity, count in sorted(processing_stats['complexity_levels'].items()):
        logger.info(f"  - {complexity}: {count} procedures")
    
    enhancement_rate = (processing_stats['enhanced_procedure_count'] / processing_stats['successful_conversions'] * 100) if processing_stats['successful_conversions'] > 0 else 0
    logger.info(f"Enhanced procedure rate: {enhancement_rate:.1f}%")
    
    return docs

def embed_security_documents_with_flattened_metadata(docs):
    """
    Embed internal security procedure documents into the Chroma vector store using flattened metadata.
    
    This function handles the embedding process with our procedural metadata flattening approach.
    The flattened metadata should now be compatible with Chroma's constraints while preserving
    all the sophisticated procedural information your security analysis system needs.
    
    We include comprehensive error handling and logging to ensure you can monitor the success
    of our procedural metadata flattening solution.
    """
    logger.info("Creating embeddings and storing in vector database with flattened procedural metadata...")
    logger.info("Testing compatibility with vector database metadata constraints...")
    
    # Initialize embeddings with the same model used throughout your system
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=api_key
    )
    
    # Create or connect to the vector store with enhanced error detection
    db = Chroma(
        persist_directory=INTERNAL_SEC_DB_DIR,
        embedding_function=embeddings,
        collection_name="internal_security_procedures"
    )
    
    # Clear existing collection to ensure clean start with new metadata format
    try:
        existing_count = db._collection.count()
        if existing_count > 0:
            logger.info(f"Clearing existing collection with {existing_count} documents...")
            db._collection.delete(where={})
            logger.info("Collection cleared successfully")
    except Exception as e:
        logger.info(f"No existing collection found or error checking: {e}")
    
    # Test metadata compatibility with a small sample first
    if docs:
        logger.info("Testing procedural metadata compatibility with sample document...")
        try:
            test_doc = docs[0]
            # Attempt to add one document to test our flattening approach
            db.add_documents([test_doc])
            logger.info("✅ Procedural metadata compatibility test successful - flattening approach works!")
            
            # Clear the test document
            try:
                # Get all document IDs first
                all_docs = db._collection.get()
                if all_docs['ids']:
                    # Delete by IDs if any exist
                    db._collection.delete(ids=all_docs['ids'])
                    logger.info("Test collection cleared successfully")
            except Exception as e:
                logger.info(f"Collection was already empty or error clearing: {e}")

        except Exception as e:
            logger.error(f"❌ Procedural metadata compatibility test failed: {e}")
            logger.error("This suggests our procedural flattening approach needs adjustment")
            raise Exception(f"Procedural metadata flattening failed compatibility test: {e}")
    
    # Proceed with full embedding using proven compatible metadata
    batch_size = 25  # Conservative batch size for reliability
    logger.info(f"Adding {len(docs)} documents to vector store in batches of {batch_size}...")
    
    total_batches = (len(docs) - 1) // batch_size + 1
    successful_batches = 0
    total_embedded = 0
    metadata_errors = 0
    
    for i in tqdm(range(0, len(docs), batch_size), desc="Embedding enhanced security procedures"):
        batch = docs[i:i+batch_size]
        batch_num = i // batch_size + 1
        
        logger.info(f"Processing batch {batch_num}/{total_batches} with {len(batch)} documents")
        
        # Log sample of enhanced metadata being processed
        sample_procedures = []
        for doc in batch[:3]:
            procedure_num = doc.metadata.get('procedure_number', 'N/A')
            section_type = doc.metadata.get('type', 'unknown')
            has_enhanced = doc.metadata.get('has_enhanced_procedure', False)
            complexity = doc.metadata.get('procedure_complexity', 'unknown')
            sample_procedures.append(f"Procedure {procedure_num} ({section_type}, {'enhanced' if has_enhanced else 'basic'}, {complexity})")
        
        logger.info(f"Batch {batch_num} sample: {', '.join(sample_procedures)}")
        
        # Implement exponential backoff for API rate limits
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                db.add_documents(batch)
                logger.info(f"✅ Successfully embedded batch {batch_num}/{total_batches}")
                successful_batches += 1
                total_embedded += len(batch)
                break
                
            except Exception as e:
                retry_count += 1
                wait_time = 2 ** retry_count
                
                # Check if this is a metadata-related error
                if "metadata" in str(e).lower():
                    metadata_errors += 1
                    logger.error(f"❌ Metadata error in batch {batch_num}: {str(e)}")
                    logger.error("This suggests our procedural flattening approach may have missed some complex metadata")
                    break  # Don't retry metadata errors
                else:
                    logger.warning(f"⚠️  Error embedding batch {batch_num}, retry {retry_count}/{max_retries} "
                                 f"in {wait_time}s: {str(e)}")
                    time.sleep(wait_time)
                
                if retry_count == max_retries:
                    logger.error(f"❌ Failed to embed batch {batch_num} after {max_retries} retries")
    
    # Comprehensive final statistics and validation
    logger.info("=" * 60)
    logger.info("ENHANCED SECURITY PROCEDURE EMBEDDING COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Successful batches: {successful_batches}/{total_batches}")
    logger.info(f"Documents embedded: {total_embedded}/{len(docs)}")
    logger.info(f"Metadata errors: {metadata_errors}")
    
    # Verify the vector store contains our enhanced documents
    try:
        final_count = db._collection.count()
        logger.info(f"Vector store final count: {final_count} documents")
        
        # Test retrieval of enhanced procedural metadata
        if final_count > 0:
            test_docs = db.similarity_search("access control", k=1)
            if test_docs:
                test_metadata = test_docs[0].metadata
                has_enhanced = test_metadata.get('has_enhanced_procedure', False)
                logger.info(f"Verification: Retrieved document has enhanced procedure: {'✓' if has_enhanced else '✗'}")
                
                if has_enhanced:
                    logger.info(f"Enhanced procedural metadata preserved: step_count={test_metadata.get('implementation_step_count', 0)}, "
                               f"complexity={test_metadata.get('procedure_complexity', 'unknown')}")
            
    except Exception as e:
        logger.warning(f"Could not verify final embedding results: {e}")
    
    if metadata_errors > 0:
        logger.warning(f"⚠️  {metadata_errors} metadata errors occurred - review procedural flattening approach")
    else:
        logger.info("✅ No metadata errors - procedural flattening approach completely successful!")
    
    return db

def save_enhanced_security_processing_summary(docs, output_path):
    """
    Save a comprehensive summary of enhanced security procedure processing.
    
    This summary helps you understand how well the enhanced procedural metadata flattening
    worked and provides statistics about the implementation complexity of your processed
    internal security procedures.
    """
    logger.info(f"Saving enhanced security processing summary to: {output_path}")
    
    summary = {
        "processing_timestamp": datetime.now().isoformat(),
        "total_documents": len(docs),
        "enhanced_procedure_stats": {
            "enhanced_procedure_count": 0,
            "simple_sections": 0,
            "complexity_distribution": {},
            "tool_usage_stats": {},
            "implementation_statistics": {
                "min_steps": float('inf'),
                "max_steps": 0,
                "total_steps": 0,
                "procedures_with_sub_steps": 0
            }
        },
        "section_types": {},
        "procedures_processed": set(),
        "sections_processed": set(),
        "sample_enhanced_procedures": []
    }
    
    # Analyze enhanced procedural metadata in processed documents
    for i, doc in enumerate(docs):
        metadata = doc.metadata
        
        # Basic document analysis
        section_type = metadata.get('type', 'unknown')
        summary["section_types"][section_type] = summary["section_types"].get(section_type, 0) + 1
        
        if metadata.get('procedure_number'):
            summary["procedures_processed"].add(metadata['procedure_number'])
        if metadata.get('section_number'):
            summary["sections_processed"].add(metadata['section_number'])
        
        # Enhanced procedural metadata analysis
        if metadata.get('has_enhanced_procedure', False):
            summary["enhanced_procedure_stats"]["enhanced_procedure_count"] += 1
            
            # Complexity analysis
            complexity = metadata.get('procedure_complexity', 'unknown')
            summary["enhanced_procedure_stats"]["complexity_distribution"][complexity] = \
                summary["enhanced_procedure_stats"]["complexity_distribution"].get(complexity, 0) + 1
            
            # Tool usage analysis
            tool_count = metadata.get('required_tools_count', 0)
            if tool_count > 0:
                summary["enhanced_procedure_stats"]["tool_usage_stats"][str(tool_count)] = \
                    summary["enhanced_procedure_stats"]["tool_usage_stats"].get(str(tool_count), 0) + 1
            
            # Implementation step statistics
            step_count = metadata.get('implementation_step_count', 0)
            if step_count > 0:
                stats = summary["enhanced_procedure_stats"]["implementation_statistics"]
                stats["min_steps"] = min(stats["min_steps"], step_count)
                stats["max_steps"] = max(stats["max_steps"], step_count)
                stats["total_steps"] += step_count
                
                if metadata.get('has_sub_steps', False):
                    stats["procedures_with_sub_steps"] += 1
            
            # Save sample enhanced procedures
            if len(summary["sample_enhanced_procedures"]) < 20:
                summary["sample_enhanced_procedures"].append({
                    "procedure_number": metadata.get('procedure_number', 'N/A'),
                    "section_title": metadata.get('section_title', 'N/A'),
                    "content_preview": doc.page_content[:150] + "...",
                    "enhanced_metadata": {
                        "implementation_step_count": metadata.get('implementation_step_count', 0),
                        "has_sub_steps": metadata.get('has_sub_steps', False),
                        "procedure_complexity": metadata.get('procedure_complexity', 'unknown'),
                        "required_tools_count": metadata.get('required_tools_count', 0)
                    }
                })
        else:
            summary["enhanced_procedure_stats"]["simple_sections"] += 1
    
    # Calculate final statistics
    if summary["enhanced_procedure_stats"]["implementation_statistics"]["min_steps"] == float('inf'):
        summary["enhanced_procedure_stats"]["implementation_statistics"]["min_steps"] = 0
    
    # Convert sets to sorted lists for JSON serialization
    summary["procedures_processed"] = sorted(list(summary["procedures_processed"]))
    summary["sections_processed"] = sorted(list(summary["sections_processed"]))
    
    # Calculate enhancement rate
    total_docs = summary["total_documents"]
    enhanced_docs = summary["enhanced_procedure_stats"]["enhanced_procedure_count"]
    enhancement_rate = (enhanced_docs / total_docs * 100) if total_docs > 0 else 0
    summary["enhancement_rate_percent"] = round(enhancement_rate, 1)
    
    # Save comprehensive summary
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Enhanced security processing summary saved:")
    logger.info(f"  - Total documents: {total_docs}")
    logger.info(f"  - Enhanced procedures: {enhanced_docs} ({enhancement_rate:.1f}%)")
    logger.info(f"  - Procedures processed: {len(summary['procedures_processed'])}")
    logger.info(f"  - Sections processed: {len(summary['sections_processed'])}")

def process_enhanced_internal_security():
    """
    Main function to process enhanced internal security procedures JSON with sophisticated metadata flattening.
    
    This represents the complete solution adapted for internal security procedures from the
    Polish law approach. We take your enhanced procedural JSON structure and transform it
    into a format that works within technical limitations while preserving all the
    sophisticated capabilities your security procedure analysis system needs.
    """
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("STARTING ENHANCED INTERNAL SECURITY PROCESSING WITH PROCEDURAL METADATA FLATTENING")
    logger.info("=" * 80)
    
    # Path to your internal security procedures JSON file
    security_json_file = os.path.join(PROCESSED_DIR, "internal_security_procedures_final_manual.json")
    
    if not os.path.exists(security_json_file):
        # Also check RAW_DIR as backup location
        backup_path = os.path.join(RAW_DIR, "internal_security_procedures_final_manual.json")
        if os.path.exists(backup_path):
            security_json_file = backup_path
            logger.info(f"Using backup location: {backup_path}")
        else:
            logger.error(f"Enhanced security JSON file not found in either location:")
            logger.error(f"  Primary: {os.path.join(PROCESSED_DIR, 'internal_security_procedures_final_manual.json')}")
            logger.error(f"  Backup: {backup_path}")
            raise FileNotFoundError(f"Required enhanced security JSON file not found")
    
    try:
        # Step 1: Load enhanced JSON with procedural validation
        logger.info("STEP 1: Loading enhanced security JSON with procedural validation...")
        data, sections = load_enhanced_internal_security_json(security_json_file)
        source_metadata = data.get('document_metadata', {})
        
        # Step 2: Convert sections with intelligent procedural metadata flattening
        logger.info("STEP 2: Converting sections with intelligent procedural metadata flattening...")
        docs = create_documents_from_enhanced_security_sections(sections, source_metadata)
        
        if not docs:
            logger.error("No valid documents created from enhanced security sections")
            return None
        
        # Step 3: Embed documents with flattened but complete procedural metadata
        logger.info("STEP 3: Embedding documents with flattened procedural metadata...")
        db = embed_security_documents_with_flattened_metadata(docs)
        
        # Step 4: Save comprehensive processing summary
        logger.info("STEP 4: Saving comprehensive procedural processing summary...")
        summary_path = os.path.join(PROCESS_INTERNAL_SEC_LOGS, f"enhanced_security_processing_summary_{timestamp}.json")
        save_enhanced_security_processing_summary(docs, summary_path)
        
        # Final completion statistics
        execution_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info("ENHANCED INTERNAL SECURITY PROCESSING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Execution time: {execution_time:.2f} seconds")
        logger.info(f"Documents processed: {len(docs)}")
        logger.info(f"Vector store location: {INTERNAL_SEC_DB_DIR}")
        logger.info(f"Enhanced log file: {log_file}")
        logger.info(f"Processing summary: {summary_path}")
        logger.info("Enhanced procedural metadata flattening approach successful!")
        logger.info("Vector database compatibility achieved while preserving all functionality!")
        logger.info("=" * 80)
        
        return db
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error("=" * 80)
        logger.error("ENHANCED INTERNAL SECURITY PROCESSING FAILED")
        logger.error(f"Error after {execution_time:.2f} seconds: {str(e)}")
        logger.error("=" * 80)
        raise

if __name__ == "__main__":
    try:
        process_enhanced_internal_security()
        logger.info("Enhanced security processing completed successfully!")
    except Exception as e:
        logger.error(f"Fatal error in enhanced security processing: {str(e)}")
        exit(1)