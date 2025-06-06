"""
Enhanced Internal Security Procedure Processing - Unified Main Script

This follows the same clean architecture as the GDPR and Polish law processing, combining 
orchestration logic with the entry point for a focused, maintainable solution. The refactored 
approach demonstrates how breaking down a monolithic script into focused, single-responsibility 
modules creates a much more maintainable and reliable system.

The main script's role has been dramatically simplified - it's now focused purely on:
- Environment setup and validation
- Component initialization with proper dependency injection  
- Pipeline orchestration with comprehensive error handling
- Result aggregation and reporting

This transformation shows the power of good software architecture. What was once a complex,
hard-to-maintain monolithic script has become a clean orchestration layer that coordinates
well-designed, focused components. Each component can be developed, tested, and maintained
independently, while the main script ensures they work together harmoniously.

The architecture demonstrates several key principles:
- Single Responsibility Principle: Each module has one clear job
- Dependency Injection: Components receive their dependencies rather than creating them
- Fail-Fast Validation: Problems are caught early when they're easier to understand
- Comprehensive Error Handling: Each stage can fail gracefully without breaking the pipeline
- Clear Separation of Concerns: Orchestration logic is separate from implementation details

The procedural focus brings unique orchestration challenges around implementation workflow
validation and security procedure citation creation that differ from legal document processing.
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv

from internal_security_loader import InternalSecurityDocumentLoader, create_internal_security_loader
from internal_security_metadata_flattener import InternalSecurityMetadataFlattener, create_internal_security_metadata_flattener
from internal_security_document_converter import InternalSecurityDocumentConverter, create_internal_security_document_converter
from internal_security_embedder import InternalSecurityVectorStoreEmbedder, create_internal_security_embedder
from internal_security_summary_generator import InternalSecurityProcessingSummaryGenerator, create_internal_security_summary_generator

# Load environment variables
load_dotenv()

def setup_comprehensive_logging() -> logging.Logger:
    """
    Set up comprehensive logging for the enhanced internal security procedure processing pipeline.
    
    This creates a logger with detailed formatting and file handling, providing complete visibility 
    into the processing pipeline operations. The logging setup follows the same proven pattern as 
    the GDPR and Polish law processing, ensuring consistency across the entire system.
    
    Comprehensive logging is crucial for complex procedural data processing pipelines because it 
    provides the visibility needed to understand what's happening, debug issues, and optimize 
    performance. The logging setup demonstrates how to create production-ready logging infrastructure 
    that balances detail with usability for procedural document processing.
    """
    # Define paths with flexible directory structure support
    # This approach works whether the script is run from the project root or from a subdirectory
    current_dir = os.path.dirname(__file__)
    if 'backend' in current_dir:
        project_root = os.path.join(current_dir, "..", "..")
        project_root = os.path.abspath(project_root)
        data_dir = os.path.join(project_root, "data")
    else:
        # We're at project root
        data_dir = os.path.join(current_dir, "data")
    
    log_dir = os.path.join(data_dir, "process_internal_security_logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamped log file for this processing session
    # The timestamp helps track different processing runs and makes debugging easier
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"enhanced_internal_security_processing_{timestamp}.log")
    
    # Configure logger with detailed formatting for debugging and monitoring
    logger = logging.getLogger("EnhancedInternalSecurityProcessing")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # Clear any existing handlers to prevent conflicts
    
    # File handler for persistent detailed logs
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Console handler for immediate feedback during processing
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Detailed formatter that includes function context for debugging
    # This level of detail is invaluable when troubleshooting complex procedural processing issues
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Enhanced internal security procedure processing logging initialized. Log file: {log_file}")
    return logger

def create_processing_config(openai_api_key: str) -> Dict[str, Any]:
    """
    Create configuration with paths that work from any directory structure.
    
    This function automatically detects whether we're running from the project root or from within 
    a backend directory structure and adjusts paths accordingly. This flexibility makes the system 
    much easier to deploy and test in different environments.
    
    The configuration approach demonstrates how to create robust systems that work reliably regardless 
    of how they're deployed or where they're executed from. The procedural focus requires specific 
    configuration for security procedure document patterns and organizational structures.
    """
    current_dir = os.path.dirname(__file__)
    
    if 'backend' in current_dir:
        # We're in backend/processing/internal_security/, calculate path to project root
        project_root = os.path.join(current_dir, "..", "..")
        project_root = os.path.abspath(project_root)
    else:
        # We're at project root
        project_root = current_dir
    
    # Build all paths relative to project root for consistency
    data_dir = os.path.join(project_root, "data")
    
    config = {
        # Directory paths organized for clarity and maintainability
        'data_dir': data_dir,
        'raw_dir': os.path.join(data_dir, "raw"),
        'processed_dir': os.path.join(data_dir, "processed"),
        'internal_security_db_path': os.path.join(data_dir, "internal_security_db"),
        'logs_dir': os.path.join(data_dir, "process_internal_security_logs"),
        
        # File names for internal security procedure processing
        'security_procedures_filename': 'internal_security_procedures_final_manual.json',
        
        # API configuration
        'openai_api_key': openai_api_key,
        
        # Processing settings optimized for internal security procedure documents
        'batch_size': 25,
        'max_retries': 5,
        'embedding_model': 'text-embedding-3-large',
        'collection_name': 'internal_security_procedures'
    }
    
    # Ensure all directories exist before processing begins
    # This proactive approach prevents errors during processing
    for dir_path in [config['raw_dir'], config['processed_dir'], 
                     config['internal_security_db_path'], config['logs_dir']]:
        os.makedirs(dir_path, exist_ok=True)
    
    return config

def validate_environment() -> str:
    """
    Validate that all required environment variables are available.
    
    This validation follows the "fail-fast" principle - we want to discover environment issues 
    immediately rather than failing deep into the processing pipeline. Early validation saves 
    time and provides clear feedback about what needs to be fixed.
    
    Returns:
        OpenAI API key if validation succeeds
        
    Raises:
        ValueError: If required environment variables are missing
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables. "
            "Please set this environment variable to proceed with internal security procedure processing."
        )
    
    return api_key

def execute_loading_stage(loader: InternalSecurityDocumentLoader, config: Dict[str, Any], 
                         logger: logging.Logger) -> Tuple[Optional[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    """
    Execute the document loading and validation stage.
    
    This stage finds the internal security procedure JSON file and validates its structure to ensure 
    it contains the enhanced procedural metadata needed for the pipeline. The loading stage is critical 
    because all subsequent stages depend on having valid, well-structured input data with proper 
    implementation step information.
    
    The stage demonstrates how to structure processing steps with clear inputs, outputs, and error 
    handling that provides useful feedback when things go wrong in procedural document processing.
    """
    stage_name = "Document Loading and Validation"
    logger.info(f"STAGE 1: {stage_name}")
    
    try:
        # Find the internal security procedure JSON file using the loader's file discovery logic
        security_file_path = loader.find_security_procedure_file(
            config['processed_dir'],
            config['raw_dir'],
            config['security_procedures_filename']
        )
        
        # Load and validate the document with comprehensive error handling
        data, sections = loader.load_and_validate_security_json(security_file_path)
        
        logger.info(f"✅ {stage_name} completed successfully")
        logger.info(f"   - Loaded {len(sections)} sections from {security_file_path}")
        
        return data, sections
        
    except Exception as e:
        logger.error(f"❌ {stage_name} failed: {str(e)}")
        return None, None

def execute_conversion_stage(converter: InternalSecurityDocumentConverter, sections: List[Dict[str, Any]], 
                           data: Dict[str, Any], processing_timestamp: str,
                           logger: logging.Logger) -> Optional[List[Any]]:
    """
    Execute the document conversion with procedural metadata flattening stage.
    
    This stage converts the enhanced JSON sections into LangChain Documents while applying the 
    sophisticated procedural metadata flattening approach. The conversion stage is where the magic 
    of the "bilingual documents" happens for procedural documents - creating documents that work 
    with both simple vector databases and sophisticated procedure citation systems.
    
    The stage demonstrates how to handle complex procedural transformations with comprehensive error 
    handling and clear success/failure reporting. The procedural focus requires different validation 
    patterns compared to legal documents, emphasizing implementation workflow preservation.
    """
    stage_name = "Document Conversion with Procedural Metadata Flattening"
    logger.info(f"STAGE 2: {stage_name}")
    
    try:
        source_metadata = data.get('document_metadata', {})
        
        docs = converter.convert_sections_to_documents(
            sections, 
            source_metadata, 
            processing_timestamp
        )
        
        if not docs:
            raise Exception("No documents were successfully converted")
        
        logger.info(f"✅ {stage_name} completed successfully")
        logger.info(f"   - Converted {len(docs)} documents with flattened procedural metadata")
        
        return docs
        
    except Exception as e:
        logger.error(f"❌ {stage_name} failed: {str(e)}")
        return None

def execute_embedding_stage(embedder: InternalSecurityVectorStoreEmbedder, docs: List[Any],
                          logger: logging.Logger) -> Optional[Any]:
    """
    Execute the vector store embedding with validation stage.
    
    This stage embeds the converted documents into the Chroma vector store while validating that 
    the flattened procedural metadata approach works correctly. The embedding stage is particularly 
    complex for procedural documents because it must handle the intersection of three challenging 
    systems: sophisticated procedural metadata, vector database constraints, and unreliable network APIs.
    
    The stage demonstrates how to handle complex, multi-step operations with comprehensive validation 
    and retry logic specifically adapted for procedural document complexity patterns.
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

def execute_summary_stage(summary_generator: InternalSecurityProcessingSummaryGenerator, docs: List[Any],
                        processing_timestamp: str, config: Dict[str, Any],
                        flattener_stats: Dict[str, Any], converter_stats: Dict[str, Any],
                        embedder_stats: Dict[str, Any], logger: logging.Logger) -> None:
    """
    Execute the summary generation and statistics stage.
    
    This stage aggregates statistics from all processing stages and generates a comprehensive summary 
    of the pipeline results. The summary stage provides valuable insights that help understand system 
    performance and identify optimization opportunities specifically for procedural document processing.
    
    The stage demonstrates how to aggregate information from multiple sources to create actionable 
    insights about system performance and reliability for security procedure workflows.
    """
    stage_name = "Summary Generation and Statistics"
    logger.info(f"STAGE 4: {stage_name}")
    
    try:
        # Generate comprehensive summary by aggregating information from all stages
        summary = summary_generator.generate_comprehensive_summary(
            docs=docs,
            processing_timestamp=processing_timestamp,
            loader_stats=None,  # Loader doesn't currently expose detailed stats
            flattener_stats=flattener_stats,
            converter_stats=converter_stats,
            embedder_stats=embedder_stats
        )
        
        # Save summary to file for permanent record keeping
        summary_path = os.path.join(
            config['logs_dir'],
            f"enhanced_internal_security_processing_summary_{processing_timestamp}.json"
        )
        summary_generator.save_summary_to_file(summary, summary_path)
        
        logger.info(f"✅ {stage_name} completed successfully")
        logger.info(f"   - Summary saved to {summary_path}")
        
    except Exception as e:
        logger.warning(f"⚠️  {stage_name} failed: {str(e)}")
        # Don't re-raise - summary generation failure shouldn't fail the entire pipeline
        # This demonstrates graceful degradation where non-critical failures don't stop processing

def process_enhanced_internal_security() -> bool:
    """
    Execute the complete enhanced internal security procedure processing pipeline.
    
    This function represents the culmination of the refactoring effort. What was once a monolithic 
    script with mixed concerns has become a clean orchestration function that coordinates well-designed, 
    focused components. Each component handles one aspect of the processing pipeline, making the entire 
    system much more maintainable and reliable.
    
    The function demonstrates several important software engineering principles:
    - Clear separation of concerns between orchestration and implementation
    - Comprehensive error handling that provides useful feedback
    - Fail-fast validation to catch problems early
    - Graceful degradation where non-critical failures don't stop processing
    - Comprehensive logging and reporting for operational visibility
    
    The procedural focus brings unique orchestration challenges around implementation workflow validation 
    and security procedure citation creation that differ from legal document processing.
    
    Returns:
        True if processing completed successfully, False otherwise
    """
    # Initialize timing and session tracking for comprehensive monitoring
    start_time = time.time()
    processing_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Step 1: Set up logging infrastructure for comprehensive visibility
    logger = setup_comprehensive_logging()
    logger.info("Starting enhanced internal security procedure processing with modular architecture")
    
    logger.info("=" * 80)
    logger.info("ENHANCED INTERNAL SECURITY PROCEDURE PROCESSING WITH PROCEDURAL METADATA FLATTENING")
    logger.info(f"Session ID: {processing_timestamp}")
    logger.info("Demonstrating the power of modular, maintainable architecture for procedural documents")
    logger.info("=" * 80)
    
    try:
        # Step 2: Validate environment and create configuration
        # This early validation follows the fail-fast principle
        logger.info("Validating environment and creating configuration...")
        api_key = validate_environment()
        config = create_processing_config(api_key)
        logger.info("Environment validation and configuration completed successfully")
        
        # Step 3: Initialize all processing modules with dependency injection
        # The dependency injection pattern makes the system much more testable and maintainable
        logger.info("Initializing all processing modules with dependency injection...")
        loader = create_internal_security_loader(logger)
        metadata_flattener = create_internal_security_metadata_flattener(logger)
        converter = create_internal_security_document_converter(metadata_flattener, logger)
        embedder = create_internal_security_embedder(config['internal_security_db_path'], api_key, logger)
        summary_generator = create_internal_security_summary_generator(logger)
        logger.info("All processing modules initialized successfully")
        
        # Step 4: Execute the complete processing pipeline with comprehensive error handling
        # Each stage can fail gracefully without breaking the entire pipeline
        logger.info("Executing complete enhanced internal security procedure processing pipeline...")
        
        # Stage 1: Document Loading and Validation
        data, sections = execute_loading_stage(loader, config, logger)
        if not sections:
            logger.error("Loading stage failed - cannot continue processing")
            return False
        
        # Stage 2: Document Conversion with Procedural Metadata Flattening
        docs = execute_conversion_stage(converter, sections, data, processing_timestamp, logger)
        if not docs:
            logger.error("Conversion stage failed - cannot continue processing")
            return False
        
        # Stage 3: Vector Store Embedding with Validation
        db = execute_embedding_stage(embedder, docs, logger)
        if not db:
            logger.error("Embedding stage failed - cannot continue processing")
            return False
        
        # Stage 4: Summary Generation and Statistics (non-critical)
        # This stage demonstrates graceful degradation - its failure doesn't stop the pipeline
        flattener_stats = metadata_flattener.get_flattening_statistics()
        converter_stats = converter.get_conversion_statistics()
        embedder_stats = embedder.get_embedding_statistics()
        
        execute_summary_stage(summary_generator, docs, processing_timestamp, config,
                            flattener_stats, converter_stats, embedder_stats, logger)
        
        # Step 5: Log successful completion with comprehensive metrics
        execution_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info("ENHANCED INTERNAL SECURITY PROCEDURE PROCESSING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Total execution time: {execution_time:.2f} seconds")
        logger.info(f"Processing session: {processing_timestamp}")
        logger.info(f"Documents processed: {len(docs)}")
        logger.info(f"Vector store location: {config['internal_security_db_path']}")
        logger.info("Enhanced procedural metadata flattening approach successful!")
        logger.info("Vector database compatibility achieved while preserving all functionality!")
        logger.info("Modular architecture demonstrates the power of well-designed components!")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error("=" * 80)
        logger.error("CRITICAL PIPELINE ERROR - PROCESSING TERMINATED")
        logger.error(f"Error after {execution_time:.2f} seconds: {str(e)}")
        logger.error("Check individual component logs for detailed error information")
        logger.error("=" * 80)
        return False

def main():
    """
    Entry point for the enhanced internal security procedure processing script.
    
    This provides a clean command-line interface that handles the complete processing pipeline 
    with comprehensive error handling and user feedback. The main function demonstrates how to 
    create user-friendly interfaces for complex processing systems.
    
    The interface approach balances technical detail with user-friendly feedback, providing enough 
    information to understand what's happening without overwhelming users with excessive technical 
    details. The procedural focus brings unique user communication challenges around implementation 
    workflow explanation.
    """
    print("Enhanced Internal Security Procedure Processing with Procedural Metadata Flattening")
    print("=" * 75)
    print("Processing internal security procedure documents with sophisticated workflow preservation...")
    print("Demonstrating modular, maintainable architecture patterns for procedural documents...")
    print()
    
    try:
        success = process_enhanced_internal_security()
        
        if success:
            print()
            print("✅ Enhanced internal security procedure processing completed successfully!")
            print("Your vector database is ready with flattened procedural metadata that preserves")
            print("all the sophisticated implementation information for precise procedure citations.")
            print()
            print("The modular architecture demonstrates:")
            print("  • Clean separation of concerns with single-responsibility modules")
            print("  • Sophisticated procedural metadata flattening with vector database compatibility")
            print("  • Comprehensive error handling and graceful failure recovery")
            print("  • Detailed logging and statistics throughout the pipeline")
            print("  • Maintainable code that's easy to test and modify")
            print("  • Dependency injection for flexible, testable components")
            print("  • Procedural workflow preservation for actionable implementation guidance")
            print()
            print("This refactoring shows how good architecture makes procedural systems:")
            print("  • Easier to understand and maintain")
            print("  • More reliable and robust for complex workflows")
            print("  • Simpler to test and debug")
            print("  • Flexible and adaptable to changing security requirements")
            print("  • Capable of preserving complex implementation details")
            
        else:
            print()
            print("❌ Enhanced internal security procedure processing encountered errors.")
            print("Please check the log files for detailed error information.")
            print("The modular architecture makes it easier to identify and fix issues.")
            exit(1)
            
    except KeyboardInterrupt:
        print()
        print("⚠️  Processing interrupted by user.")
        exit(1)
    except Exception as e:
        print()
        print(f"💥 Unexpected error: {str(e)}")
        print("Check the logs for detailed debugging information.")
        exit(1)

if __name__ == "__main__":
    main()