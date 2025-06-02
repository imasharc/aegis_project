"""
Enhanced GDPR Processing - Unified Main Script

This combines the orchestration logic with the entry point for a cleaner,
more direct solution. Since we only have one way to run GDPR
processing, we don't need a separate orchestrator class.
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv

from gdpr_loader import GDPRDocumentLoader, create_gdpr_loader
from gdpr_metadata_flattener import GDPRMetadataFlattener, create_gdpr_metadata_flattener
from gdpr_document_converter import GDPRDocumentConverter, create_gdpr_document_converter
from gdpr_embedder import GDPRVectorStoreEmbedder, create_gdpr_embedder
from gdpr_summary_generator import GDPRProcessingSummaryGenerator, create_gdpr_summary_generator

# Load environment variables
load_dotenv()

def setup_comprehensive_logging() -> logging.Logger:
    """
    Set up comprehensive logging for the enhanced GDPR processing pipeline.
    
    This creates a logger with detailed formatting and file handling,
    providing complete visibility into the processing pipeline operations.
    """
    # Define paths
    current_dir = os.path.dirname(__file__)
    if 'backend' in current_dir:
        project_root = os.path.join(current_dir, "..", "..", "..")
        project_root = os.path.abspath(project_root)
        data_dir = os.path.join(project_root, "data")
    else:
        # We're at project root
        data_dir = os.path.join(current_dir, "data")
    
    log_dir = os.path.join(data_dir, "process_gdpr_logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"enhanced_gdpr_processing_{timestamp}.log")
    
    # Configure logger with detailed formatting
    logger = logging.getLogger("EnhancedGDPRProcessing")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # Clear any existing handlers
    
    # File handler for persistent detailed logs
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Console handler for immediate feedback
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Detailed formatter for debugging and monitoring
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Enhanced GDPR processing logging initialized. Log file: {log_file}")
    return logger

def create_processing_config(openai_api_key: str) -> Dict[str, Any]:
    """
    Create configuration with paths that work from any directory structure.
    
    This function automatically detects whether we're running from the project
    root or from within a backend directory structure and adjusts paths accordingly.
    """
    current_dir = os.path.dirname(__file__)
    
    if 'backend' in current_dir:
        # We're in backend/processing/gdpr/, calculate path to project root
        project_root = os.path.join(current_dir, "..", "..", "..")
        project_root = os.path.abspath(project_root)
    else:
        # We're at project root
        project_root = current_dir
    
    # Build all paths relative to project root
    data_dir = os.path.join(project_root, "data")
    
    config = {
        # Directory paths
        'data_dir': data_dir,
        'raw_dir': os.path.join(data_dir, "raw"),
        'processed_dir': os.path.join(data_dir, "processed"),
        'gdpr_db_path': os.path.join(data_dir, "gdpr_db"),
        'logs_dir': os.path.join(data_dir, "process_gdpr_logs"),
        
        # File names
        'gdpr_filename': 'gdpr_final_manual.json',
        
        # API configuration
        'openai_api_key': openai_api_key,
        
        # Processing settings
        'batch_size': 25,
        'max_retries': 5,
        'embedding_model': 'text-embedding-3-large',
        'collection_name': 'gdpr_regulation'
    }
    
    # Ensure all directories exist
    for dir_path in [config['raw_dir'], config['processed_dir'], 
                     config['gdpr_db_path'], config['logs_dir']]:
        os.makedirs(dir_path, exist_ok=True)
    
    return config

def validate_environment() -> str:
    """
    Validate that all required environment variables are available.
    
    Returns:
        OpenAI API key if validation succeeds
        
    Raises:
        ValueError: If required environment variables are missing
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables. "
            "Please set this environment variable to proceed with GDPR processing."
        )
    
    return api_key

def execute_loading_stage(loader: GDPRDocumentLoader, config: Dict[str, Any], 
                         logger: logging.Logger) -> Tuple[Optional[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    """
    Execute the document loading and validation stage.
    
    This stage finds the GDPR JSON file and validates its structure
    to ensure it contains the enhanced metadata needed for the pipeline.
    """
    stage_name = "Document Loading and Validation"
    logger.info(f"STAGE 1: {stage_name}")
    
    try:
        # Find the GDPR JSON file
        gdpr_file_path = loader.find_gdpr_file(
            config['processed_dir'],
            config['raw_dir'],
            config['gdpr_filename']
        )
        
        # Load and validate the document
        data, chunks = loader.load_and_validate_gdpr_json(gdpr_file_path)
        
        logger.info(f"✅ {stage_name} completed successfully")
        logger.info(f"   - Loaded {len(chunks)} chunks from {gdpr_file_path}")
        
        return data, chunks
        
    except Exception as e:
        logger.error(f"❌ {stage_name} failed: {str(e)}")
        return None, None

def execute_conversion_stage(converter: GDPRDocumentConverter, chunks: List[Dict[str, Any]], 
                           data: Dict[str, Any], processing_timestamp: str,
                           logger: logging.Logger) -> Optional[List[Any]]:
    """
    Execute the document conversion with metadata flattening stage.
    
    This stage converts the enhanced JSON chunks into LangChain Documents
    while applying the sophisticated metadata flattening approach.
    """
    stage_name = "Document Conversion with Metadata Flattening"
    logger.info(f"STAGE 2: {stage_name}")
    
    try:
        source_metadata = data.get('document', {}).get('metadata', {})
        
        docs = converter.convert_chunks_to_documents(
            chunks, 
            source_metadata, 
            processing_timestamp
        )
        
        if not docs:
            raise Exception("No documents were successfully converted")
        
        logger.info(f"✅ {stage_name} completed successfully")
        logger.info(f"   - Converted {len(docs)} documents with flattened metadata")
        
        return docs
        
    except Exception as e:
        logger.error(f"❌ {stage_name} failed: {str(e)}")
        return None

def execute_embedding_stage(embedder: GDPRVectorStoreEmbedder, docs: List[Any],
                          logger: logging.Logger) -> Optional[Any]:
    """
    Execute the vector store embedding with validation stage.
    
    This stage embeds the converted documents into the Chroma vector store
    while validating that the flattened metadata approach works correctly.
    """
    stage_name = "Vector Store Embedding with Validation"
    logger.info(f"STAGE 3: {stage_name}")
    
    try:
        db = embedder.embed_documents_with_validation(docs)
        
        if not db:
            raise Exception("Vector store embedding failed")
        
        logger.info(f"✅ {stage_name} completed successfully")
        logger.info(f"   - Embedded {len(docs)} documents into vector store")
        
        return db
        
    except Exception as e:
        logger.error(f"❌ {stage_name} failed: {str(e)}")
        return None

def execute_summary_stage(summary_generator: GDPRProcessingSummaryGenerator, docs: List[Any],
                        processing_timestamp: str, config: Dict[str, Any],
                        flattener_stats: Dict[str, Any], converter_stats: Dict[str, Any],
                        embedder_stats: Dict[str, Any], logger: logging.Logger) -> None:
    """
    Execute the summary generation and statistics stage.
    
    This stage aggregates statistics from all processing stages and
    generates a comprehensive summary of the pipeline results.
    """
    stage_name = "Summary Generation and Statistics"
    logger.info(f"STAGE 4: {stage_name}")
    
    try:
        # Generate comprehensive summary
        summary = summary_generator.generate_comprehensive_summary(
            docs=docs,
            processing_timestamp=processing_timestamp,
            loader_stats=None,  # Loader doesn't currently expose detailed stats
            flattener_stats=flattener_stats,
            converter_stats=converter_stats,
            embedder_stats=embedder_stats
        )
        
        # Save summary to file
        summary_path = os.path.join(
            config['logs_dir'],
            f"enhanced_gdpr_processing_summary_{processing_timestamp}.json"
        )
        summary_generator.save_summary_to_file(summary, summary_path)
        
        logger.info(f"✅ {stage_name} completed successfully")
        logger.info(f"   - Summary saved to {summary_path}")
        
    except Exception as e:
        logger.warning(f"⚠️  {stage_name} failed: {str(e)}")
        # Don't re-raise - summary generation failure shouldn't fail the pipeline

def process_enhanced_gdpr() -> bool:
    """
    Execute the complete enhanced GDPR processing pipeline.
    
    This function combines orchestration logic with the main entry point,
    creating a clean, direct solution that handles the entire pipeline
    from environment validation through final summary generation.
    
    Returns:
        True if processing completed successfully, False otherwise
    """
    # Initialize timing and session tracking
    start_time = time.time()
    processing_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Step 1: Set up logging infrastructure
    logger = setup_comprehensive_logging()
    logger.info("Starting enhanced GDPR processing with combined orchestration")
    
    logger.info("=" * 80)
    logger.info("ENHANCED GDPR PROCESSING WITH METADATA FLATTENING")
    logger.info(f"Session ID: {processing_timestamp}")
    logger.info("=" * 80)
    
    try:
        # Step 2: Validate environment and create configuration
        logger.info("Validating environment and creating configuration...")
        api_key = validate_environment()
        config = create_processing_config(api_key)
        logger.info("Environment validation and configuration completed successfully")
        
        # Step 3: Initialize all processing modules with dependency injection
        logger.info("Initializing all processing modules...")
        loader = create_gdpr_loader(logger)
        metadata_flattener = create_gdpr_metadata_flattener(logger)
        converter = create_gdpr_document_converter(metadata_flattener, logger)
        embedder = create_gdpr_embedder(config['gdpr_db_path'], api_key, logger)
        summary_generator = create_gdpr_summary_generator(logger)
        logger.info("All processing modules initialized successfully")
        
        # Step 4: Execute the complete processing pipeline
        logger.info("Executing complete enhanced GDPR processing pipeline...")
        
        # Stage 1: Document Loading and Validation
        data, chunks = execute_loading_stage(loader, config, logger)
        if not chunks:
            logger.error("Loading stage failed - cannot continue processing")
            return False
        
        # Stage 2: Document Conversion with Metadata Flattening
        docs = execute_conversion_stage(converter, chunks, data, processing_timestamp, logger)
        if not docs:
            logger.error("Conversion stage failed - cannot continue processing")
            return False
        
        # Stage 3: Vector Store Embedding with Validation
        db = execute_embedding_stage(embedder, docs, logger)
        if not db:
            logger.error("Embedding stage failed - cannot continue processing")
            return False
        
        # Stage 4: Summary Generation and Statistics
        flattener_stats = metadata_flattener.get_flattening_statistics()
        converter_stats = converter.get_conversion_statistics()
        embedder_stats = embedder.get_embedding_statistics()
        
        execute_summary_stage(summary_generator, docs, processing_timestamp, config,
                            flattener_stats, converter_stats, embedder_stats, logger)
        
        # Step 5: Log successful completion
        execution_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info("ENHANCED GDPR PROCESSING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Total execution time: {execution_time:.2f} seconds")
        logger.info(f"Processing session: {processing_timestamp}")
        logger.info(f"Documents processed: {len(docs)}")
        logger.info(f"Vector store location: {config['gdpr_db_path']}")
        logger.info("Enhanced metadata flattening approach successful!")
        logger.info("Vector database compatibility achieved while preserving all functionality!")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error("=" * 80)
        logger.error("CRITICAL PIPELINE ERROR - PROCESSING TERMINATED")
        logger.error(f"Error after {execution_time:.2f} seconds: {str(e)}")
        logger.error("=" * 80)
        return False

def main():
    """
    Entry point for the enhanced GDPR processing script.
    
    This provides a clean command-line interface that handles the complete
    processing pipeline with comprehensive error handling and user feedback.
    """
    print("Enhanced GDPR Processing with Metadata Flattening")
    print("=" * 50)
    print("Processing GDPR documents with sophisticated structural preservation...")
    print()
    
    try:
        success = process_enhanced_gdpr()
        
        if success:
            print()
            print("✅ Enhanced GDPR processing completed successfully!")
            print("Your vector database is ready with flattened metadata that preserves")
            print("all the sophisticated structural information for precise citations.")
            print()
            print("The combined architecture demonstrates:")
            print("  • Clean separation of concerns within a single, focused file")
            print("  • Sophisticated metadata flattening with vector database compatibility")
            print("  • Comprehensive error handling and recovery")
            print("  • Detailed logging and statistics throughout the pipeline")
            print("  • Simplified deployment with fewer moving parts")
            
        else:
            print()
            print("❌ Enhanced GDPR processing encountered errors.")
            print("Please check the log files for detailed error information.")
            exit(1)
            
    except KeyboardInterrupt:
        print()
        print("⚠️  Processing interrupted by user.")
        exit(1)
    except Exception as e:
        print()
        print(f"💥 Unexpected error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()