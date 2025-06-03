"""
GDPR Vector Store Connector

This module handles all interactions with the Chroma vector store for GDPR documents.
Following the Single Responsibility Principle, this module's only job is to safely
connect to the vector store, validate its structure, and retrieve relevant documents
with their flattened metadata intact.

Key responsibilities:
- Connect to and validate the GDPR vector store
- Perform similarity searches with proper filtering
- Validate flattened metadata structure in retrieved documents
- Provide comprehensive logging and error handling for vector operations
"""

import os
import time
import logging
from typing import Dict, List, Any, Tuple
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


class GDPRVectorStoreConnector:
    """
    Handles connection and interaction with the GDPR vector store.
    
    This class encapsulates all the logic for safely connecting to the vector store
    and retrieving documents while validating that the flattened metadata approach
    is working correctly. It provides a clean interface for document retrieval
    operations while handling all the complexity of vector store management.
    """
    
    def __init__(self, db_path: str, logger: logging.Logger):
        """
        Initialize the GDPR vector store connector.
        
        Args:
            db_path: Path to the Chroma database directory
            logger: Configured logger instance for detailed operation tracking
        """
        self.db_path = db_path
        self.logger = logger
        self.gdpr_db = None
        
        # Initialize retrieval statistics
        self.retrieval_stats = {
            'total_queries': 0,
            'total_documents_retrieved': 0,
            'enhanced_metadata_documents': 0,
            'retrieval_errors': 0,
            'average_retrieval_time': 0.0
        }
        
        self.logger.info("GDPR Vector Store Connector initialized")
    
    def connect_and_validate(self) -> bool:
        """
        Connect to the GDPR vector store and validate its structure.
        
        This method establishes the connection to the vector store and performs
        comprehensive validation to ensure that the flattened metadata approach
        is working correctly with the stored documents.
        
        Returns:
            True if connection and validation succeed, False otherwise
        """
        self.logger.info("Connecting to GDPR vector store with metadata validation...")
        
        try:
            # Initialize embeddings using the same model as processing
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            self.logger.info("Embeddings model initialized: text-embedding-3-large")
            
            # Connect to the GDPR vector store
            if not os.path.exists(self.db_path):
                self.logger.error(f"GDPR vector store not found at: {self.db_path}")
                return False
            
            self.gdpr_db = Chroma(
                persist_directory=self.db_path,
                embedding_function=embeddings,
                collection_name="gdpr_regulation"
            )
            
            # Validate the connection and metadata structure
            return self._validate_store_structure()
            
        except Exception as e:
            self.logger.error(f"Error connecting to GDPR vector store: {e}")
            return False
    
    def _validate_store_structure(self) -> bool:
        """
        Validate that the vector store contains expected documents with flattened metadata.
        
        This validation ensures that the documents were processed correctly by the
        enhanced processing pipeline and contain the metadata structure needed for
        sophisticated citation creation.
        """
        try:
            collection_count = self.gdpr_db._collection.count()
            self.logger.info(f"GDPR vector store validation: {collection_count} documents found")
            
            if collection_count == 0:
                self.logger.warning("GDPR vector store is empty - no documents to validate")
                return True  # Empty store is valid, just not useful
            
            # Test retrieval and metadata structure
            test_docs = self.gdpr_db.similarity_search("Article 1", k=1)
            if test_docs:
                test_metadata = test_docs[0].metadata
                
                # Validate presence of expected flattened fields for GDPR
                expected_fields = [
                    'law', 'article_number', 'has_enhanced_structure', 
                    'paragraph_count', 'complexity_level'
                ]
                
                present_fields = []
                missing_fields = []
                
                for field in expected_fields:
                    if field in test_metadata:
                        present_fields.append(field)
                    else:
                        missing_fields.append(field)
                
                self.logger.info(f"Metadata validation: {len(present_fields)}/{len(expected_fields)} expected fields present")
                
                if missing_fields:
                    self.logger.warning(f"Missing expected metadata fields: {missing_fields}")
                    self.logger.warning("Store may be from older processing - basic functionality will work")
                
                # Test enhanced structure if available
                if test_metadata.get('has_enhanced_structure', False):
                    self._validate_enhanced_structure(test_metadata)
                
                self.logger.info("✅ GDPR vector store structure validation completed")
                return True
            else:
                self.logger.warning("Could not retrieve test documents for validation")
                return False
                
        except Exception as e:
            self.logger.error(f"Error validating GDPR vector store structure: {e}")
            return False
    
    def _validate_enhanced_structure(self, metadata: Dict[str, Any]) -> None:
        """
        Validate enhanced metadata structure in a sample document.
        
        This checks that the sophisticated flattened metadata is properly preserved
        and can be accessed for precise citation creation.
        """
        # Check for JSON structure preservation
        json_str = metadata.get('article_structure_json', '')
        if json_str:
            try:
                import json
                deserialized = json.loads(json_str)
                self.logger.info("✅ Enhanced structure JSON deserialization successful")
            except Exception as e:
                self.logger.warning(f"⚠️  Enhanced structure JSON deserialization failed: {e}")
        else:
            self.logger.info("No JSON structure found in sample document")
        
        # Log structural indicators
        paragraph_count = metadata.get('paragraph_count', 0)
        has_sub_paragraphs = metadata.get('has_sub_paragraphs', False)
        complexity = metadata.get('complexity_level', 'unknown')
        
        self.logger.info(f"Enhanced structure indicators: {paragraph_count} paragraphs, "
                        f"sub-paragraphs: {has_sub_paragraphs}, complexity: {complexity}")
    
    def retrieve_relevant_documents(self, query: str, k: int = 8) -> Tuple[List, str, List[Dict]]:
        """
        Retrieve relevant GDPR documents with comprehensive metadata tracking.
        
        This method performs similarity search while tracking statistics about
        the retrieved documents and their metadata quality. It provides detailed
        logging for debugging and optimization purposes.
        
        Args:
            query: Search query for similarity search
            k: Number of documents to retrieve
            
        Returns:
            Tuple of (documents, formatted_context, document_metadata_list)
        """
        if not self.gdpr_db:
            error_msg = "GDPR vector store is not connected"
            self.logger.error(error_msg)
            return [], f"ERROR: {error_msg}", []
        
        self.logger.info(f"Retrieving GDPR documents for query: '{query[:100]}...'")
        self.logger.info(f"Requesting top {k} documents with metadata analysis")
        
        self.retrieval_stats['total_queries'] += 1
        
        try:
            # Perform similarity search with timing
            start_time = time.time()
            docs = self.gdpr_db.similarity_search(
                query, 
                k=k,
                filter={"law": "gdpr"}  # Ensure we only get GDPR documents
            )
            retrieval_time = time.time() - start_time
            
            # Update statistics
            self.retrieval_stats['total_documents_retrieved'] += len(docs)
            self.retrieval_stats['average_retrieval_time'] = \
                (self.retrieval_stats['average_retrieval_time'] * (self.retrieval_stats['total_queries'] - 1) + retrieval_time) / \
                self.retrieval_stats['total_queries']
            
            self.logger.info(f"Retrieved {len(docs)} GDPR documents in {retrieval_time:.3f} seconds")
            
            # Process and analyze retrieved documents
            context_pieces, document_metadata = self._process_retrieved_documents(docs)
            
            # Create formatted context for LLM
            retrieved_context = "\n\n" + "="*80 + "\n\n".join(context_pieces)
            
            # Log comprehensive retrieval statistics
            self._log_retrieval_statistics(docs, document_metadata)
            
            return docs, retrieved_context, document_metadata
            
        except Exception as e:
            self.retrieval_stats['retrieval_errors'] += 1
            error_msg = f"Error during GDPR document retrieval: {e}"
            self.logger.error(error_msg)
            return [], f"ERROR: {error_msg}", []
    
    def _process_retrieved_documents(self, docs: List) -> Tuple[List[str], List[Dict]]:
        """
        Process retrieved documents and extract metadata for analysis.
        
        This method analyzes each retrieved document to understand its metadata
        quality and creates formatted context pieces for LLM processing.
        """
        context_pieces = []
        document_metadata = []
        
        for i, doc in enumerate(docs):
            metadata = doc.metadata
            
            # Extract and validate metadata structure
            has_enhanced = metadata.get('has_enhanced_structure', False)
            complexity = metadata.get('complexity_level', 'unknown')
            
            # Track enhanced metadata statistics
            if has_enhanced:
                self.retrieval_stats['enhanced_metadata_documents'] += 1
            
            # Create document metadata entry for further processing
            doc_metadata = {
                'index': i,
                'metadata': metadata,
                'content': doc.page_content,
                'has_enhanced_structure': has_enhanced,
                'complexity_level': complexity
            }
            document_metadata.append(doc_metadata)
            
            # Build enhanced reference for context display
            reference = self._build_document_reference(metadata, has_enhanced, complexity)
            
            # Create formatted context piece
            context_piece = f"[Document {i+1} - {reference}]\n{doc.page_content}"
            context_pieces.append(context_piece)
            
            # Log document details for verification
            self._log_document_details(i, metadata, has_enhanced, complexity, doc.page_content)
        
        return context_pieces, document_metadata
    
    def _build_document_reference(self, metadata: Dict[str, Any], 
                                 has_enhanced: bool, complexity: str) -> str:
        """
        Build a comprehensive reference string for a document.
        
        This creates a human-readable reference that shows the document's
        structure and metadata quality for logging and debugging purposes.
        """
        article_num = metadata.get('article_number', 'N/A')
        doc_type = metadata.get('type', 'unknown')
        chapter_num = metadata.get('chapter_number', 'N/A')
        paragraph_count = metadata.get('paragraph_count', 0)
        
        reference = f"GDPR - Article {article_num}"
        
        if metadata.get('chapter_title'):
            try:
                # Convert chapter to Roman numerals for GDPR
                chapter_roman = self._convert_to_roman(int(chapter_num))
                reference += f" (Chapter {chapter_roman}: {metadata['chapter_title']})"
            except (ValueError, TypeError):
                reference += f" (Chapter {chapter_num}: {metadata['chapter_title']})"
        
        reference += f" - {doc_type}"
        
        if has_enhanced:
            reference += f" [Enhanced: {complexity}, {paragraph_count}p]"
        
        return reference
    
    def _convert_to_roman(self, num: int) -> str:
        """Convert an integer to Roman numerals for GDPR chapter references."""
        values = [10, 9, 5, 4, 1]
        numerals = ['X', 'IX', 'V', 'IV', 'I']
        result = ''
        
        for i, value in enumerate(values):
            count = num // value
            if count:
                result += numerals[i] * count
                num -= value * count
        
        return result
    
    def _log_document_details(self, index: int, metadata: Dict[str, Any], 
                             has_enhanced: bool, complexity: str, content: str) -> None:
        """
        Log detailed information about a retrieved document.
        
        This provides comprehensive visibility into what documents were retrieved
        and their metadata quality for debugging and optimization.
        """
        article_num = metadata.get('article_number', 'N/A')
        doc_type = metadata.get('type', 'unknown')
        chapter_num = metadata.get('chapter_number', 'N/A')
        paragraph_count = metadata.get('paragraph_count', 0)
        
        self.logger.info(f"GDPR Document {index+1}: Article {article_num} ({doc_type}), "
                        f"Chapter {chapter_num}, Enhanced: {'✓' if has_enhanced else '✗'}, "
                        f"Complexity: {complexity}, Paragraphs: {paragraph_count}, "
                        f"Content: {len(content)} chars")
        
        # Log content preview for verification
        if index < 3:  # Only log previews for first few documents
            content_preview = content[:150].replace('\n', ' ')
            self.logger.debug(f"GDPR Document {index+1} preview: {content_preview}...")
    
    def _log_retrieval_statistics(self, docs: List, document_metadata: List[Dict]) -> None:
        """
        Log comprehensive statistics about the retrieval operation.
        
        This provides insights into the quality and patterns of retrieved documents
        to help optimize the retrieval and citation processes.
        """
        enhanced_count = sum(1 for doc_meta in document_metadata 
                           if doc_meta.get('has_enhanced_structure', False))
        
        # Calculate complexity distribution
        complexity_distribution = {}
        for doc_meta in document_metadata:
            complexity = doc_meta.get('complexity_level', 'unknown')
            complexity_distribution[complexity] = complexity_distribution.get(complexity, 0) + 1
        
        # Calculate article and type distributions
        article_counts = {}
        type_counts = {}
        for doc in docs:
            article = doc.metadata.get('article_number', 'unknown')
            doc_type = doc.metadata.get('type', 'unknown')
            article_counts[article] = article_counts.get(article, 0) + 1
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        enhancement_rate = (enhanced_count / len(docs) * 100) if docs else 0
        
        # Log comprehensive statistics
        self.logger.info("=" * 60)
        self.logger.info("GDPR DOCUMENT RETRIEVAL STATISTICS")
        self.logger.info("=" * 60)
        self.logger.info(f"Documents retrieved: {len(docs)}")
        self.logger.info(f"Enhanced metadata: {enhanced_count}/{len(docs)} documents")
        self.logger.info(f"Enhancement rate: {enhancement_rate:.1f}%")
        self.logger.info(f"Complexity distribution: {dict(complexity_distribution)}")
        self.logger.info(f"Article distribution: {dict(article_counts)}")
        self.logger.info(f"Document type distribution: {dict(type_counts)}")
    
    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about all retrieval operations.
        
        Returns:
            Dictionary containing detailed retrieval statistics
        """
        stats = dict(self.retrieval_stats)
        
        # Calculate enhancement rate across all retrievals
        if stats['total_documents_retrieved'] > 0:
            enhancement_rate = (stats['enhanced_metadata_documents'] / stats['total_documents_retrieved']) * 100
            stats['enhancement_rate_percent'] = round(enhancement_rate, 1)
        else:
            stats['enhancement_rate_percent'] = 0
        
        return stats
    
    def is_connected(self) -> bool:
        """
        Check if the vector store connection is active.
        
        Returns:
            True if connected and ready for operations, False otherwise
        """
        return self.gdpr_db is not None


def create_gdpr_vector_store_connector(db_path: str, logger: logging.Logger) -> GDPRVectorStoreConnector:
    """
    Factory function to create a configured GDPR vector store connector.
    
    This provides a clean interface for creating connector instances with
    proper dependency injection of configuration and logger.
    """
    return GDPRVectorStoreConnector(db_path, logger)