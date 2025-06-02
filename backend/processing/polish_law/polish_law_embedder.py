"""
Polish Law Vector Store Embedder with Comprehensive Validation

This module handles the complex process of embedding Polish law documents into the 
Chroma vector store using the flattened metadata approach. Following the same proven 
pattern as the GDPR embedder, it includes sophisticated error handling, batch processing, 
retry logic, and comprehensive validation to ensure reliable operation.

The embedder represents one of the most technically challenging components because it must
handle the intersection of three complex systems: the sophisticated metadata we've created,
the vector database's constraints, and the unpredictable nature of network APIs. The
solution demonstrates how to build robust systems that gracefully handle failures and
provide comprehensive feedback about what's happening during processing.

This component showcases several important software engineering principles:
- Fail-fast validation to catch problems early
- Comprehensive error handling with intelligent retry logic
- Batch processing optimization for large datasets
- Detailed monitoring and reporting for operational visibility
"""

import os
import time
import logging
from typing import List, Dict, Any, Tuple
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from tqdm import tqdm


class PolishLawVectorStoreEmbedder:
    """
    Handles embedding Polish law documents into Chroma vector store with flattened metadata validation.
    
    This class encapsulates all the complexity of working with vector databases while ensuring 
    that our flattened metadata approach works correctly for Polish law documents. The design 
    follows the principle of "hope for the best, prepare for the worst" - we expect things to 
    work smoothly, but we're prepared to handle any issues that arise.
    
    The embedder implements a sophisticated validation and processing pipeline:
    1. Test metadata compatibility with a single document first
    2. Process documents in manageable batches with progress tracking
    3. Implement intelligent retry logic for transient failures
    4. Provide comprehensive statistics and error reporting
    
    This approach ensures that even when dealing with large document collections and 
    unreliable network conditions, the system behaves predictably and provides useful 
    feedback about any issues that occur.
    """
    
    def __init__(self, db_path: str, api_key: str, logger: logging.Logger):
        """
        Initialize the Polish law vector store embedder.
        
        The initialization sets up all the infrastructure needed for reliable embedding
        operations while establishing comprehensive statistics tracking. This preparation
        is crucial for handling the complex, multi-step embedding process that follows.
        
        Args:
            db_path: Path to the Chroma database directory
            api_key: OpenAI API key for embeddings
            logger: Configured logger for tracking operations
        """
        self.db_path = db_path
        self.api_key = api_key
        self.logger = logger
        self.db = None
        
        # Initialize comprehensive embedding statistics
        # These statistics provide complete visibility into the embedding process
        # and help identify patterns, optimization opportunities, and potential issues
        self.embedding_stats = {
            'total_documents': 0,
            'successful_batches': 0,
            'failed_batches': 0,
            'metadata_compatibility_tested': False,
            'metadata_errors': 0,
            'retry_attempts': 0,
            'polish_specific_metrics': {
                'sections_processed': 0,
                'gazette_references_found': 0,
                'enhanced_articles': 0
            }
        }
        
        self.logger.info("Polish Law Vector Store Embedder initialized")
    
    def embed_documents_with_validation(self, docs: List[Document]) -> Chroma:
        """
        Embed Polish law documents into the vector store with comprehensive validation.
        
        This method implements a careful, step-by-step approach to ensure that our flattened 
        metadata works correctly with the vector database. The process demonstrates how to 
        handle complex, multi-stage operations where each stage depends on the success of 
        the previous stages.
        
        The validation-first approach is crucial because it catches metadata compatibility 
        issues before we invest time and resources in processing thousands of documents. 
        This "fail-fast" principle saves time and provides clear feedback about any problems.
        
        Args:
            docs: List of LangChain Document objects ready for embedding
            
        Returns:
            Configured Chroma vector store instance
            
        Raises:
            Exception: If metadata compatibility fails or critical errors occur
        """
        self.logger.info("Starting Polish law document embedding with comprehensive validation...")
        self.logger.info("Testing compatibility with vector database metadata constraints...")
        
        self.embedding_stats['total_documents'] = len(docs)
        
        # The embedding process follows a carefully designed sequence where each step
        # validates the success of the previous step before proceeding. This approach
        # ensures that problems are caught early and provide clear diagnostic information.
        
        # Step 1: Initialize the vector store and embeddings infrastructure
        self._initialize_vector_store()
        
        # Step 2: Test metadata compatibility with a sample document (fail-fast approach)
        self._test_metadata_compatibility(docs)
        
        # Step 3: Process documents in batches with comprehensive error handling
        self._process_documents_in_batches(docs)
        
        # Step 4: Validate final results and provide comprehensive reporting
        self._validate_final_embedding_results()
        
        self.logger.info("Polish law document embedding completed successfully")
        return self.db
    
    def _initialize_vector_store(self) -> None:
        """
        Initialize the Chroma vector store and embeddings with proper configuration.
        
        This sets up the vector database connection and clears any existing data to ensure 
        a clean start with the new flattened metadata format. The clean start is important 
        because it prevents confusion between old and new metadata formats.
        
        The initialization process demonstrates how to set up complex external dependencies
        while providing clear feedback about what's happening at each step.
        """
        self.logger.info("Initializing Chroma vector store and OpenAI embeddings for Polish law...")
        
        # Initialize embeddings with the same model used throughout your system
        # Consistency in embedding models is crucial for system reliability
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=self.api_key
        )
        
        # Create or connect to the Polish law vector store
        # The collection name clearly identifies this as Polish law data
        self.db = Chroma(
            persist_directory=self.db_path,
            embedding_function=embeddings,
            collection_name="polish_data_protection_law"
        )
        
        # Clear existing collection for a clean start with new metadata format
        # This ensures we don't have any confusion between old and new data
        self._clear_existing_collection()
        
        self.logger.info("Vector store initialization completed")
    
    def _clear_existing_collection(self) -> None:
        """
        Clear any existing documents from the collection for a clean start.
        
        This ensures we're working with fresh data that uses the new flattened metadata 
        format throughout. While it might seem wasteful to delete existing data, this 
        approach prevents subtle bugs that can occur when mixing different metadata formats.
        
        The clearing process demonstrates how to safely reset system state while providing
        clear feedback about what's being removed and why.
        """
        try:
            existing_count = self.db._collection.count()
            if existing_count > 0:
                self.logger.info(f"Clearing existing Polish law collection with {existing_count} documents...")
                self.db._collection.delete(where={})
                self.logger.info("Polish law collection cleared successfully")
            else:
                self.logger.info("No existing Polish law collection found - starting fresh")
        except Exception as e:
            self.logger.info(f"No existing Polish law collection found or error checking: {e}")
    
    def _test_metadata_compatibility(self, docs: List[Document]) -> None:
        """
        Test metadata compatibility with a sample document before full processing.
        
        This critical step validates that our flattened metadata approach works correctly 
        with the vector database. The test-first approach is essential because metadata 
        compatibility issues can be subtle and difficult to debug once you're processing 
        thousands of documents.
        
        The testing approach demonstrates the "fail-fast" principle - we want to discover 
        problems as early as possible when they're easier to understand and fix. Testing 
        with a single document gives us confidence that the full processing will work correctly.
        
        Args:
            docs: List of documents to test with
            
        Raises:
            Exception: If metadata compatibility test fails
        """
        if not docs:
            self.logger.warning("No Polish law documents available for metadata compatibility testing")
            return
        
        self.logger.info("Testing Polish law metadata compatibility with sample document...")
        
        try:
            test_doc = docs[0]
            
            # Log what we're testing for transparency and debugging
            # This information is invaluable when diagnosing compatibility issues
            self._log_test_document_metadata(test_doc)
            
            # Attempt to add one Polish law document to test our flattening approach
            # This is the critical test - if this fails, we know there's a fundamental
            # problem with our metadata structure
            self.db.add_documents([test_doc])
            
            self.logger.info("✅ Polish law metadata compatibility test successful - flattening approach works!")
            self.embedding_stats['metadata_compatibility_tested'] = True
            
            # Clean up the test document to start fresh for the actual processing
            self._cleanup_test_document()
            
        except Exception as e:
            error_msg = f"❌ Polish law metadata compatibility test failed: {e}"
            self.logger.error(error_msg)
            self.logger.error("This suggests our Polish law flattening approach needs adjustment")
            raise Exception(f"Polish law metadata flattening failed compatibility test: {e}")
    
    def _log_test_document_metadata(self, test_doc: Document) -> None:
        """
        Log details about the test document metadata for transparency.
        
        This helps understand exactly what metadata structure we're testing and provides 
        debugging information if the test fails. The detailed logging is particularly 
        important for complex metadata structures because it helps identify exactly 
        which fields or patterns are causing compatibility issues.
        
        The logging demonstrates how to provide just enough detail for debugging without
        overwhelming the logs with excessive information.
        """
        metadata = test_doc.metadata
        
        self.logger.info("Test Polish law document metadata structure:")
        self.logger.info(f"  - Article: {metadata.get('article_number', 'N/A')}")
        self.logger.info(f"  - Type: {metadata.get('type', 'unknown')}")
        self.logger.info(f"  - Enhanced structure: {metadata.get('has_enhanced_structure', False)}")
        self.logger.info(f"  - Complexity: {metadata.get('complexity_level', 'unknown')}")
        self.logger.info(f"  - Content length: {len(test_doc.page_content)} characters")
        
        # Log Polish law-specific metadata fields
        if metadata.get('section_number'):
            self.logger.info(f"  - Section: {metadata['section_number']}")
        if metadata.get('gazette_reference'):
            self.logger.info(f"  - Gazette reference: {metadata['gazette_reference']}")
    
    def _cleanup_test_document(self) -> None:
        """
        Remove the test document from the collection after successful testing.
        
        This ensures we start with a clean collection for the actual embedding process.
        The cleanup step is important because it prevents the test document from appearing
        in the final results, which could be confusing and affect system behavior.
        """
        try:
            all_docs = self.db._collection.get()
            if all_docs['ids']:
                self.db._collection.delete(ids=all_docs['ids'])
                self.logger.info("Test document cleaned up successfully")
        except Exception as e:
            self.logger.info(f"Collection was already empty or error during cleanup: {e}")
    
    def _process_documents_in_batches(self, docs: List[Document]) -> None:
        """
        Process documents in batches with comprehensive error handling and retry logic.
        
        Batch processing is essential for large document collections because it provides 
        better control over resource usage, enables progress tracking, and allows for 
        more sophisticated error handling. The approach demonstrates how to balance 
        efficiency with reliability in data processing systems.
        
        The batching strategy includes several important considerations:
        - Conservative batch sizes to avoid overwhelming the API
        - Comprehensive progress tracking for long-running operations
        - Intelligent retry logic for transient failures
        - Clear distinction between retryable and non-retryable errors
        
        Args:
            docs: List of documents to embed
        """
        batch_size = 25  # Conservative batch size for reliability with Polish law documents
        total_batches = (len(docs) - 1) // batch_size + 1
        
        self.logger.info(f"Processing {len(docs)} Polish law documents in {total_batches} batches of {batch_size}...")
        
        # Process each batch with progress tracking and comprehensive error handling
        # The progress tracking is important for long-running operations because it
        # provides feedback to users and helps estimate completion times
        for i in tqdm(range(0, len(docs), batch_size), desc="Embedding enhanced Polish law documents"):
            batch = docs[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            success = self._process_single_batch(batch, batch_num, total_batches)
            
            # Track batch success/failure for comprehensive statistics
            if success:
                self.embedding_stats['successful_batches'] += 1
            else:
                self.embedding_stats['failed_batches'] += 1
    
    def _process_single_batch(self, batch: List[Document], batch_num: int, total_batches: int) -> bool:
        """
        Process a single batch of documents with retry logic and error handling.
        
        This method implements sophisticated error handling that distinguishes between 
        different types of failures and responds appropriately to each. The approach 
        demonstrates how to build resilient systems that can handle the unpredictable 
        nature of network APIs and external services.
        
        The retry logic is particularly important for API-based operations because 
        temporary failures (like rate limits or network issues) are common and 
        usually resolve themselves quickly. However, we need to distinguish these 
        from permanent failures (like metadata incompatibility) that won't be 
        fixed by retrying.
        
        Args:
            batch: Documents in this batch
            batch_num: Current batch number for logging
            total_batches: Total number of batches for progress tracking
            
        Returns:
            True if batch was processed successfully, False otherwise
        """
        self.logger.info(f"Processing Polish law batch {batch_num}/{total_batches} with {len(batch)} documents")
        
        # Log sample of metadata being processed for verification and debugging
        # This helps identify patterns in successful vs. failed batches
        self._log_batch_sample(batch, batch_num)
        
        # Implement exponential backoff for API rate limits and transient failures
        # This is a standard pattern for dealing with rate-limited APIs
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                self.db.add_documents(batch)
                self.logger.info(f"✅ Successfully embedded Polish law batch {batch_num}/{total_batches}")
                return True
                
            except Exception as e:
                retry_count += 1
                self.embedding_stats['retry_attempts'] += 1
                
                # Intelligent error classification: distinguish between retryable and non-retryable errors
                # This is crucial for providing appropriate responses to different types of failures
                if self._is_metadata_error(e):
                    self.embedding_stats['metadata_errors'] += 1
                    self.logger.error(f"❌ Polish law metadata error in batch {batch_num}: {str(e)}")
                    self.logger.error("This suggests our Polish law flattening approach may have missed some complex metadata")
                    return False  # Don't retry metadata errors - they won't be fixed by retrying
                else:
                    # Handle other types of errors (likely network/API issues) with retry
                    wait_time = 2 ** retry_count
                    self.logger.warning(f"⚠️  Error embedding Polish law batch {batch_num}, "
                                       f"retry {retry_count}/{max_retries} in {wait_time}s: {str(e)}")
                    time.sleep(wait_time)
                
                if retry_count == max_retries:
                    self.logger.error(f"❌ Failed to embed Polish law batch {batch_num} after {max_retries} retries")
                    return False
        
        return False
    
    def _log_batch_sample(self, batch: List[Document], batch_num: int) -> None:
        """
        Log a sample of the enhanced metadata being processed in this batch.
        
        This provides visibility into what kind of documents we're processing and helps 
        identify any patterns in successful or failed embeddings. The sampling approach 
        provides useful information without overwhelming the logs with excessive detail.
        
        Sample logging is particularly valuable for debugging because it helps identify
        whether failures are related to specific types of documents or metadata patterns.
        """
        sample_articles = []
        polish_metrics = self.embedding_stats['polish_specific_metrics']
        
        for doc in batch[:3]:  # Show first 3 documents in batch
            metadata = doc.metadata
            article_num = metadata.get('article_number', 'N/A')
            chunk_type = metadata.get('type', 'unknown')
            has_enhanced = metadata.get('has_enhanced_structure', False)
            complexity = metadata.get('complexity_level', 'unknown')
            
            # Track Polish law-specific metrics
            if metadata.get('section_number'):
                polish_metrics['sections_processed'] += 1
            if metadata.get('gazette_reference'):
                polish_metrics['gazette_references_found'] += 1
            if has_enhanced:
                polish_metrics['enhanced_articles'] += 1
            
            sample_desc = f"Article {article_num} ({chunk_type}, " \
                         f"{'enhanced' if has_enhanced else 'basic'}, {complexity})"
            
            # Add Polish law-specific information to the sample description
            if metadata.get('section_number'):
                sample_desc += f", Section {metadata['section_number']}"
            
            sample_articles.append(sample_desc)
        
        self.logger.info(f"Polish law batch {batch_num} sample: {', '.join(sample_articles)}")
    
    def _is_metadata_error(self, error: Exception) -> bool:
        """
        Determine if an error is related to metadata structure issues.
        
        This helps distinguish between metadata problems (which shouldn't be retried because 
        they won't be fixed by trying again) and temporary issues like API rate limits 
        (which should be retried because they often resolve themselves).
        
        Proper error classification is crucial for building robust systems because it 
        determines the appropriate response to each type of failure. Retrying metadata 
        errors wastes time and resources, while not retrying transient failures causes 
        unnecessary processing failures.
        """
        error_str = str(error).lower()
        metadata_indicators = ['metadata', 'field', 'key', 'value', 'schema', 'type', 'structure']
        
        return any(indicator in error_str for indicator in metadata_indicators)
    
    def _validate_final_embedding_results(self) -> None:
        """
        Validate the final embedding results and log comprehensive statistics.
        
        This provides confirmation that the embedding process worked correctly and gives 
        detailed statistics about the success rate and any issues encountered. The 
        validation step is important because it provides closure on the embedding process 
        and confirms that the system is working as expected.
        
        Comprehensive final validation demonstrates how to provide complete feedback 
        about complex operations, helping users understand both what succeeded and 
        what may need attention.
        """
        try:
            final_count = self.db._collection.count()
            self.logger.info(f"Polish law vector store final count: {final_count} documents")
            
            # Test retrieval of enhanced Polish law metadata
            # This confirms that our flattened metadata approach works for both
            # storage and retrieval, completing the round-trip validation
            if final_count > 0:
                self._test_metadata_retrieval()
            
            # Log comprehensive embedding statistics including Polish law-specific metrics
            self._log_final_embedding_statistics(final_count)
            
        except Exception as e:
            self.logger.warning(f"Could not validate final Polish law embedding results: {e}")
    
    def _test_metadata_retrieval(self) -> None:
        """
        Test that we can successfully retrieve documents with enhanced metadata.
        
        This final validation step confirms that our flattened metadata approach not only 
        allows storage but also proper retrieval with all the enhanced information intact. 
        This round-trip test is crucial because it validates the entire metadata processing 
        pipeline from creation through storage to retrieval.
        
        The retrieval test demonstrates how to validate complex systems end-to-end rather
        than just testing individual components in isolation.
        """
        try:
            test_docs = self.db.similarity_search("Article 1", k=1)
            if test_docs:
                test_metadata = test_docs[0].metadata
                has_enhanced = test_metadata.get('has_enhanced_structure', False)
                
                self.logger.info(f"Polish law retrieval verification: Document has enhanced structure: "
                               f"{'✓' if has_enhanced else '✗'}")
                
                if has_enhanced:
                    paragraph_count = test_metadata.get('paragraph_count', 0)
                    complexity = test_metadata.get('complexity_level', 'unknown')
                    self.logger.info(f"Polish law enhanced metadata preserved: "
                                   f"paragraph_count={paragraph_count}, complexity={complexity}")
                
                # Test Polish law-specific metadata preservation
                if test_metadata.get('section_number'):
                    self.logger.info(f"Polish law section information preserved: Section {test_metadata['section_number']}")
                
                if test_metadata.get('gazette_reference'):
                    self.logger.info(f"Polish law gazette reference preserved: {test_metadata['gazette_reference']}")
                    
        except Exception as e:
            self.logger.warning(f"Could not test Polish law metadata retrieval: {e}")
    
    def _log_final_embedding_statistics(self, final_count: int) -> None:
        """
        Log comprehensive final statistics about the embedding process.
        
        This provides a complete picture of how the embedding process performed and 
        highlights any issues that need attention. The comprehensive reporting helps 
        both developers and system administrators understand system performance and 
        identify opportunities for optimization.
        
        Detailed statistics are particularly important for complex operations because 
        they provide the information needed to optimize performance and identify 
        potential issues before they become serious problems.
        """
        stats = self.embedding_stats
        
        self.logger.info("=" * 60)
        self.logger.info("ENHANCED POLISH LAW EMBEDDING PROCESS COMPLETED")
        self.logger.info("=" * 60)
        self.logger.info(f"Documents in vector store: {final_count}")
        self.logger.info(f"Successful batches: {stats['successful_batches']}")
        self.logger.info(f"Failed batches: {stats['failed_batches']}")
        self.logger.info(f"Metadata compatibility tested: {'✓' if stats['metadata_compatibility_tested'] else '✗'}")
        self.logger.info(f"Metadata errors: {stats['metadata_errors']}")
        self.logger.info(f"Total retry attempts: {stats['retry_attempts']}")
        
        # Log Polish law-specific metrics
        polish_metrics = stats['polish_specific_metrics']
        self.logger.info("Polish law-specific processing metrics:")
        self.logger.info(f"  - Sections processed: {polish_metrics['sections_processed']}")
        self.logger.info(f"  - Gazette references found: {polish_metrics['gazette_references_found']}")
        self.logger.info(f"  - Enhanced articles: {polish_metrics['enhanced_articles']}")
        
        # Provide assessment of embedding success
        if stats['metadata_errors'] > 0:
            self.logger.warning(f"⚠️  {stats['metadata_errors']} Polish law metadata errors occurred - "
                               f"review flattening approach")
        else:
            self.logger.info("✅ No Polish law metadata errors - flattening approach completely successful!")
    
    def get_embedding_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the embedding process.
        
        This method provides access to all the statistics collected during the embedding 
        process, making it possible for other components (like the summary generator) to 
        incorporate this information into their own reporting and analysis.
        
        Returns:
            Dictionary containing detailed embedding statistics
        """
        return dict(self.embedding_stats)


def create_polish_law_embedder(db_path: str, api_key: str, logger: logging.Logger) -> PolishLawVectorStoreEmbedder:
    """
    Factory function to create a configured Polish law vector store embedder.
    
    This provides a clean interface for creating embedder instances with proper 
    dependency injection of configuration and logger. The factory pattern ensures 
    consistent initialization and makes it easy to modify the creation process 
    if needed in the future.
    
    Factory functions are particularly valuable for complex objects that require 
    multiple parameters because they centralize the creation logic and make it 
    easier to maintain consistency across the application.
    """
    return PolishLawVectorStoreEmbedder(db_path, api_key, logger)