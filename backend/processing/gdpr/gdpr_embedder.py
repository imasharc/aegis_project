"""
GDPR Vector Store Embedder with Comprehensive Validation

This module handles the complex process of embedding GDPR documents into the Chroma 
vector store using the flattened metadata approach. The embedder includes sophisticated error handling, batch processing, retry logic, 
and comprehensive validation to ensure reliable operation even with large document sets.
"""

import os
import time
import logging
from typing import List, Dict, Any, Tuple
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from tqdm import tqdm


class GDPRVectorStoreEmbedder:
    """
    Handles embedding GDPR documents into Chroma vector store with flattened metadata validation.
    
    This class encapsulates all the complexity of working with vector databases while
    ensuring that our flattened metadata approach works correctly. It includes comprehensive
    error handling, batch processing, and validation to ensure reliable operation.
    
    The embedder implements a "test-first" approach - we validate that our metadata
    flattening works before processing the full document set.
    """
    
    def __init__(self, db_path: str, api_key: str, logger: logging.Logger):
        """
        Initialize the GDPR vector store embedder.
        
        Args:
            db_path: Path to the Chroma database directory
            api_key: OpenAI API key for embeddings
            logger: Configured logger for tracking operations
        """
        self.db_path = db_path
        self.api_key = api_key
        self.logger = logger
        self.db = None
        
        # Initialize embedding statistics
        self.embedding_stats = {
            'total_documents': 0,
            'successful_batches': 0,
            'failed_batches': 0,
            'metadata_compatibility_tested': False,
            'metadata_errors': 0,
            'retry_attempts': 0
        }
        
        self.logger.info("GDPR Vector Store Embedder initialized")
    
    def embed_documents_with_validation(self, docs: List[Document]) -> Chroma:
        """
        Embed GDPR documents into the vector store with comprehensive validation.
        
        This method implements a careful, step-by-step approach to ensure that our
        flattened metadata works correctly with the vector database. We test first,
        then process in batches with extensive error handling.
        
        Args:
            docs: List of LangChain Document objects ready for embedding
            
        Returns:
            Configured Chroma vector store instance
            
        Raises:
            Exception: If metadata compatibility fails or critical errors occur
        """
        self.logger.info("Starting GDPR document embedding with comprehensive validation...")
        self.logger.info("Testing compatibility with vector database metadata constraints...")
        
        self.embedding_stats['total_documents'] = len(docs)
        
        # Step 1: Initialize the vector store and embeddings
        self._initialize_vector_store()
        
        # Step 2: Test metadata compatibility with a sample document
        self._test_metadata_compatibility(docs)
        
        # Step 3: Process documents in batches with error handling
        self._process_documents_in_batches(docs)
        
        # Step 4: Validate final results
        self._validate_final_embedding_results()
        
        self.logger.info("GDPR document embedding completed successfully")
        return self.db
    
    def _initialize_vector_store(self) -> None:
        """
        Initialize the Chroma vector store and embeddings with proper configuration.
        
        This sets up the vector database connection and clears any existing data
        to ensure a clean start with the new flattened metadata format.
        """
        self.logger.info("Initializing Chroma vector store and OpenAI embeddings...")
        
        # Initialize embeddings with the same model used throughout your system
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=self.api_key
        )
        
        # Create or connect to the GDPR vector store
        self.db = Chroma(
            persist_directory=self.db_path,
            embedding_function=embeddings,
            collection_name="gdpr_regulation"
        )
        
        # Clear existing collection for a clean start
        self._clear_existing_collection()
        
        self.logger.info("Vector store initialization completed")
    
    def _clear_existing_collection(self) -> None:
        """
        Clear any existing documents from the collection for a clean start.
        
        This ensures we're working with fresh data that uses the new flattened
        metadata format throughout.
        """
        try:
            existing_count = self.db._collection.count()
            if existing_count > 0:
                self.logger.info(f"Clearing existing GDPR collection with {existing_count} documents...")
                self.db._collection.delete(where={})
                self.logger.info("GDPR collection cleared successfully")
        except Exception as e:
            self.logger.info(f"No existing GDPR collection found or error checking: {e}")
    
    def _test_metadata_compatibility(self, docs: List[Document]) -> None:
        """
        Test metadata compatibility with a sample document before full processing.
        
        This critical step validates that our flattened metadata approach works
        correctly with the vector database. If this test fails, we know there's
        an issue with our metadata structure before processing thousands of documents.
        
        Args:
            docs: List of documents to test with
            
        Raises:
            Exception: If metadata compatibility test fails
        """
        if not docs:
            self.logger.warning("No documents available for metadata compatibility testing")
            return
        
        self.logger.info("Testing GDPR metadata compatibility with sample document...")
        
        try:
            test_doc = docs[0]
            
            # Log what we're testing for transparency
            self._log_test_document_metadata(test_doc)
            
            # Attempt to add one GDPR document to test our flattening approach
            self.db.add_documents([test_doc])
            
            self.logger.info("✅ GDPR metadata compatibility test successful - flattening approach works!")
            self.embedding_stats['metadata_compatibility_tested'] = True
            
            # Clean up the test document
            self._cleanup_test_document()
            
        except Exception as e:
            error_msg = f"❌ GDPR metadata compatibility test failed: {e}"
            self.logger.error(error_msg)
            self.logger.error("This suggests our GDPR flattening approach needs adjustment")
            raise Exception(f"GDPR metadata flattening failed compatibility test: {e}")
    
    def _log_test_document_metadata(self, test_doc: Document) -> None:
        """
        Log details about the test document metadata for transparency.
        
        This helps understand exactly what metadata structure we're testing
        and provides debugging information if the test fails.
        """
        metadata = test_doc.metadata
        
        self.logger.info("Test document metadata structure:")
        self.logger.info(f"  - Article: {metadata.get('article_number', 'N/A')}")
        self.logger.info(f"  - Type: {metadata.get('type', 'unknown')}")
        self.logger.info(f"  - Enhanced structure: {metadata.get('has_enhanced_structure', False)}")
        self.logger.info(f"  - Complexity: {metadata.get('complexity_level', 'unknown')}")
        self.logger.info(f"  - Content length: {len(test_doc.page_content)} characters")
    
    def _cleanup_test_document(self) -> None:
        """
        Remove the test document from the collection after successful testing.
        
        This ensures we start with a clean collection for the actual embedding process.
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
        
        This method implements sophisticated batch processing with exponential backoff
        for API rate limits and detailed error reporting for metadata issues.
        
        Args:
            docs: List of documents to embed
        """
        batch_size = 25  # Conservative batch size for reliability
        total_batches = (len(docs) - 1) // batch_size + 1
        
        self.logger.info(f"Processing {len(docs)} GDPR documents in {total_batches} batches of {batch_size}...")
        
        # Process each batch with progress tracking
        for i in tqdm(range(0, len(docs), batch_size), desc="Embedding enhanced GDPR documents"):
            batch = docs[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            success = self._process_single_batch(batch, batch_num, total_batches)
            
            if success:
                self.embedding_stats['successful_batches'] += 1
            else:
                self.embedding_stats['failed_batches'] += 1
    
    def _process_single_batch(self, batch: List[Document], batch_num: int, total_batches: int) -> bool:
        """
        Process a single batch of documents with retry logic and error handling.
        
        Args:
            batch: Documents in this batch
            batch_num: Current batch number for logging
            total_batches: Total number of batches for progress tracking
            
        Returns:
            True if batch was processed successfully, False otherwise
        """
        self.logger.info(f"Processing GDPR batch {batch_num}/{total_batches} with {len(batch)} documents")
        
        # Log sample of metadata being processed for verification
        self._log_batch_sample(batch, batch_num)
        
        # Implement exponential backoff for API rate limits
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                self.db.add_documents(batch)
                self.logger.info(f"✅ Successfully embedded GDPR batch {batch_num}/{total_batches}")
                return True
                
            except Exception as e:
                retry_count += 1
                self.embedding_stats['retry_attempts'] += 1
                
                # Check if this is a metadata-related error
                if self._is_metadata_error(e):
                    self.embedding_stats['metadata_errors'] += 1
                    self.logger.error(f"❌ GDPR metadata error in batch {batch_num}: {str(e)}")
                    self.logger.error("This suggests our GDPR flattening approach may have missed some complex metadata")
                    return False  # Don't retry metadata errors
                else:
                    # Handle other types of errors with retry
                    wait_time = 2 ** retry_count
                    self.logger.warning(f"⚠️  Error embedding GDPR batch {batch_num}, "
                                       f"retry {retry_count}/{max_retries} in {wait_time}s: {str(e)}")
                    time.sleep(wait_time)
                
                if retry_count == max_retries:
                    self.logger.error(f"❌ Failed to embed GDPR batch {batch_num} after {max_retries} retries")
                    return False
        
        return False
    
    def _log_batch_sample(self, batch: List[Document], batch_num: int) -> None:
        """
        Log a sample of the enhanced metadata being processed in this batch.
        
        This provides visibility into what kind of documents we're processing
        and helps identify any patterns in successful or failed embeddings.
        """
        sample_articles = []
        for doc in batch[:3]:  # Show first 3 documents in batch
            metadata = doc.metadata
            article_num = metadata.get('article_number', 'N/A')
            chunk_type = metadata.get('type', 'unknown')
            has_enhanced = metadata.get('has_enhanced_structure', False)
            complexity = metadata.get('complexity_level', 'unknown')
            
            sample_desc = f"Article {article_num} ({chunk_type}, " \
                         f"{'enhanced' if has_enhanced else 'basic'}, {complexity})"
            sample_articles.append(sample_desc)
        
        self.logger.info(f"GDPR batch {batch_num} sample: {', '.join(sample_articles)}")
    
    def _is_metadata_error(self, error: Exception) -> bool:
        """
        Determine if an error is related to metadata structure issues.
        
        This helps distinguish between metadata problems (which shouldn't be retried)
        and temporary issues like API rate limits (which should be retried).
        """
        error_str = str(error).lower()
        metadata_indicators = ['metadata', 'field', 'key', 'value', 'schema', 'type']
        
        return any(indicator in error_str for indicator in metadata_indicators)
    
    def _validate_final_embedding_results(self) -> None:
        """
        Validate the final embedding results and log comprehensive statistics.
        
        This provides confirmation that the embedding process worked correctly
        and gives detailed statistics about the success rate and any issues encountered.
        """
        try:
            final_count = self.db._collection.count()
            self.logger.info(f"GDPR vector store final count: {final_count} documents")
            
            # Test retrieval of enhanced GDPR metadata
            if final_count > 0:
                self._test_metadata_retrieval()
            
            # Log comprehensive embedding statistics
            self._log_final_embedding_statistics(final_count)
            
        except Exception as e:
            self.logger.warning(f"Could not validate final GDPR embedding results: {e}")
    
    def _test_metadata_retrieval(self) -> None:
        """
        Test that we can successfully retrieve documents with enhanced metadata.
        
        This final validation step confirms that our flattened metadata approach
        not only allows storage but also proper retrieval with all the enhanced
        information intact.
        """
        try:
            test_docs = self.db.similarity_search("Article 1", k=1)
            if test_docs:
                test_metadata = test_docs[0].metadata
                has_enhanced = test_metadata.get('has_enhanced_structure', False)
                
                self.logger.info(f"GDPR retrieval verification: Document has enhanced structure: "
                               f"{'✓' if has_enhanced else '✗'}")
                
                if has_enhanced:
                    paragraph_count = test_metadata.get('paragraph_count', 0)
                    complexity = test_metadata.get('complexity_level', 'unknown')
                    self.logger.info(f"GDPR enhanced metadata preserved: "
                                   f"paragraph_count={paragraph_count}, complexity={complexity}")
        except Exception as e:
            self.logger.warning(f"Could not test GDPR metadata retrieval: {e}")
    
    def _log_final_embedding_statistics(self, final_count: int) -> None:
        """
        Log comprehensive final statistics about the embedding process.
        
        This provides a complete picture of how the embedding process performed
        and highlights any issues that need attention.
        """
        stats = self.embedding_stats
        
        self.logger.info("=" * 60)
        self.logger.info("ENHANCED GDPR EMBEDDING PROCESS COMPLETED")
        self.logger.info("=" * 60)
        self.logger.info(f"Documents in vector store: {final_count}")
        self.logger.info(f"Successful batches: {stats['successful_batches']}")
        self.logger.info(f"Failed batches: {stats['failed_batches']}")
        self.logger.info(f"Metadata compatibility tested: {'✓' if stats['metadata_compatibility_tested'] else '✗'}")
        self.logger.info(f"Metadata errors: {stats['metadata_errors']}")
        self.logger.info(f"Total retry attempts: {stats['retry_attempts']}")
        
        # Provide assessment of embedding success
        if stats['metadata_errors'] > 0:
            self.logger.warning(f"⚠️  {stats['metadata_errors']} GDPR metadata errors occurred - "
                               f"review flattening approach")
        else:
            self.logger.info("✅ No GDPR metadata errors - flattening approach completely successful!")
    
    def get_embedding_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the embedding process.
        
        Returns:
            Dictionary containing detailed embedding statistics
        """
        return dict(self.embedding_stats)


def create_gdpr_embedder(db_path: str, api_key: str, logger: logging.Logger) -> GDPRVectorStoreEmbedder:
    """
    Factory function to create a configured GDPR vector store embedder.
    
    This provides a clean interface for creating embedder instances with
    proper dependency injection of configuration and logger.
    """
    return GDPRVectorStoreEmbedder(db_path, api_key, logger)