"""
Polish Law Vector Store Connector

This module handles all interactions with the Chroma vector store for Polish law documents.
Following the Single Responsibility Principle established in the GDPR refactoring, this 
module's only job is to safely connect to the vector store, validate its structure, and 
retrieve relevant documents with their flattened metadata intact.

Polish Law Specific Features:
- Section-aware document filtering and organization
- Gazette reference validation for legal authenticity
- Polish legal document type recognition
- Parliament session metadata handling
- Enhanced logging for Polish legal terminology

This connector works seamlessly with the sophisticated Polish law processing pipeline
to provide the foundation for precise Polish legal citation creation.
"""

import os
import time
import logging
from typing import Dict, List, Any, Tuple
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


class PolishLawVectorStoreConnector:
    """
    Handles connection and interaction with the Polish law vector store.
    
    This class encapsulates all the logic for safely connecting to the vector store
    and retrieving documents while validating that the flattened metadata approach
    is working correctly for Polish law documents. It provides a clean interface for 
    document retrieval operations while handling all the complexity of vector store 
    management for Polish legal documents.
    
    The connector is specifically designed to work with the enhanced Polish law 
    processing pipeline, understanding Polish legal document structure including
    sections, gazette references, and Polish numbering patterns.
    """
    
    def __init__(self, db_path: str, logger: logging.Logger):
        """
        Initialize the Polish law vector store connector.
        
        Args:
            db_path: Path to the Chroma database directory for Polish law documents
            logger: Configured logger instance for detailed operation tracking
        """
        self.db_path = db_path
        self.logger = logger
        self.polish_law_db = None
        
        # Initialize retrieval statistics specific to Polish law document patterns
        self.retrieval_stats = {
            'total_queries': 0,
            'total_documents_retrieved': 0,
            'enhanced_metadata_documents': 0,
            'retrieval_errors': 0,
            'average_retrieval_time': 0.0,
            'sections_found': 0,  # Unique to Polish law
            'gazette_references_found': 0,  # Important for Polish law authenticity
            'polish_terminology_detected': 0
        }
        
        self.logger.info("Polish Law Vector Store Connector initialized")
    
    def connect_and_validate(self) -> bool:
        """
        Connect to the Polish law vector store and validate its structure.
        
        This method establishes the connection to the vector store and performs
        comprehensive validation to ensure that the flattened metadata approach
        is working correctly with the stored Polish law documents.
        
        Returns:
            True if connection and validation succeed, False otherwise
        """
        self.logger.info("Connecting to Polish law vector store with metadata validation...")
        
        try:
            # Initialize embeddings using the same model as processing pipeline
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            self.logger.info("Embeddings model initialized: text-embedding-3-large")
            
            # Connect to the Polish law vector store
            if not os.path.exists(self.db_path):
                self.logger.error(f"Polish law vector store not found at: {self.db_path}")
                return False
            
            self.polish_law_db = Chroma(
                persist_directory=self.db_path,
                embedding_function=embeddings,
                collection_name="polish_data_protection_law"  # Matches processing pipeline
            )
            
            # Validate the connection and Polish law metadata structure
            return self._validate_polish_law_store_structure()
            
        except Exception as e:
            self.logger.error(f"Error connecting to Polish law vector store: {e}")
            return False
    
    def _validate_polish_law_store_structure(self) -> bool:
        """
        Validate that the vector store contains expected Polish law documents with flattened metadata.
        
        This validation ensures that the documents were processed correctly by the
        enhanced Polish law processing pipeline and contain the metadata structure needed for
        sophisticated citation creation, including Polish law-specific elements.
        """
        try:
            collection_count = self.polish_law_db._collection.count()
            self.logger.info(f"Polish law vector store validation: {collection_count} documents found")
            
            if collection_count == 0:
                self.logger.warning("Polish law vector store is empty - no documents to validate")
                return True  # Empty store is valid, just not useful
            
            # Test retrieval and Polish law metadata structure
            test_docs = self.polish_law_db.similarity_search("Article 1", k=1)
            if test_docs:
                test_metadata = test_docs[0].metadata
                
                # Validate presence of expected flattened fields for Polish law
                expected_fields = [
                    'law', 'article_number', 'has_enhanced_structure', 
                    'paragraph_count', 'complexity_level', 'jurisdiction'
                ]
                
                # Polish law-specific fields
                polish_specific_fields = [
                    'section_number', 'gazette_reference', 'parliament_session'
                ]
                
                present_fields = []
                missing_fields = []
                polish_fields_present = []
                
                for field in expected_fields:
                    if field in test_metadata:
                        present_fields.append(field)
                    else:
                        missing_fields.append(field)
                
                for field in polish_specific_fields:
                    if field in test_metadata and test_metadata[field]:
                        polish_fields_present.append(field)
                
                self.logger.info(f"Polish law metadata validation: {len(present_fields)}/{len(expected_fields)} expected fields present")
                self.logger.info(f"Polish-specific fields found: {polish_fields_present}")
                
                if missing_fields:
                    self.logger.warning(f"Missing expected metadata fields: {missing_fields}")
                    self.logger.warning("Store may be from older processing - basic functionality will work")
                
                # Test enhanced structure if available
                if test_metadata.get('has_enhanced_structure', False):
                    self._validate_polish_enhanced_structure(test_metadata)
                
                # Validate Polish law-specific content
                self._validate_polish_law_content(test_docs[0].page_content)
                
                self.logger.info("✅ Polish law vector store structure validation completed")
                return True
            else:
                self.logger.warning("Could not retrieve test documents for validation")
                return False
                
        except Exception as e:
            self.logger.error(f"Error validating Polish law vector store structure: {e}")
            return False
    
    def _validate_polish_enhanced_structure(self, metadata: Dict[str, Any]) -> None:
        """
        Validate enhanced metadata structure in a sample Polish law document.
        
        This checks that the sophisticated flattened metadata is properly preserved
        and can be accessed for precise citation creation, including Polish law-specific
        structural elements like sections and gazette references.
        """
        # Check for JSON structure preservation
        json_str = metadata.get('article_structure_json', '')
        if json_str:
            try:
                import json
                deserialized = json.loads(json_str)
                self.logger.info("✅ Polish law enhanced structure JSON deserialization successful")
            except Exception as e:
                self.logger.warning(f"⚠️  Polish law enhanced structure JSON deserialization failed: {e}")
        else:
            self.logger.info("No JSON structure found in sample Polish law document")
        
        # Log structural indicators specific to Polish law
        paragraph_count = metadata.get('paragraph_count', 0)
        has_sub_paragraphs = metadata.get('has_sub_paragraphs', False)
        complexity = metadata.get('complexity_level', 'unknown')
        section_number = metadata.get('section_number', '')
        
        self.logger.info(f"Polish law enhanced structure indicators: {paragraph_count} paragraphs, "
                        f"sub-paragraphs: {has_sub_paragraphs}, complexity: {complexity}")
        
        if section_number:
            self.logger.info(f"Polish law section organization detected: Section {section_number}")
        
        # Check for Polish law-specific metadata
        gazette_ref = metadata.get('gazette_reference', '')
        if gazette_ref:
            self.logger.info(f"Gazette reference preserved: {gazette_ref}")
        
        parliament_session = metadata.get('parliament_session', '')
        if parliament_session:
            self.logger.info(f"Parliament session information: {parliament_session}")
    
    def _validate_polish_law_content(self, content: str) -> None:
        """
        Validate that content contains expected Polish legal terminology and patterns.
        
        This ensures that the retrieved documents are actually Polish law content
        and contain the expected linguistic and structural patterns.
        """
        polish_legal_terms = ['ustawa', 'artykuł', 'rozdział', 'przepis', 'dziennik ustaw']
        content_lower = content.lower()
        
        found_terms = [term for term in polish_legal_terms if term in content_lower]
        
        if found_terms:
            self.logger.info(f"Polish legal terminology detected: {found_terms}")
            self.retrieval_stats['polish_terminology_detected'] += len(found_terms)
        else:
            self.logger.warning("No Polish legal terminology detected in sample content")
    
    def retrieve_relevant_documents(self, query: str, k: int = 8) -> Tuple[List, str, List[Dict]]:
        """
        Retrieve relevant Polish law documents with comprehensive metadata tracking.
        
        This method performs similarity search while tracking statistics about
        the retrieved documents and their metadata quality, with special attention
        to Polish law-specific features like sections and gazette references.
        
        Args:
            query: Search query for similarity search
            k: Number of documents to retrieve
            
        Returns:
            Tuple of (documents, formatted_context, document_metadata_list)
        """
        if not self.polish_law_db:
            error_msg = "Polish law vector store is not connected"
            self.logger.error(error_msg)
            return [], f"ERROR: {error_msg}", []
        
        self.logger.info(f"Retrieving Polish law documents for query: '{query[:100]}...'")
        self.logger.info(f"Requesting top {k} documents with Polish law metadata analysis")
        
        self.retrieval_stats['total_queries'] += 1
        
        try:
            # Perform similarity search with timing and Polish law-specific filtering
            start_time = time.time()
            docs = self.polish_law_db.similarity_search(
                query, 
                k=k,
                filter={"law": "polish_data_protection"}  # Ensure we only get Polish law documents
            )
            retrieval_time = time.time() - start_time
            
            # Update statistics
            self.retrieval_stats['total_documents_retrieved'] += len(docs)
            self.retrieval_stats['average_retrieval_time'] = \
                (self.retrieval_stats['average_retrieval_time'] * (self.retrieval_stats['total_queries'] - 1) + retrieval_time) / \
                self.retrieval_stats['total_queries']
            
            self.logger.info(f"Retrieved {len(docs)} Polish law documents in {retrieval_time:.3f} seconds")
            
            # Process and analyze retrieved documents with Polish law specifics
            context_pieces, document_metadata = self._process_retrieved_polish_documents(docs)
            
            # Create formatted context for LLM with Polish law markers
            retrieved_context = "\n\n" + "="*80 + "\n\n".join(context_pieces)
            
            # Log comprehensive retrieval statistics including Polish law metrics
            self._log_polish_law_retrieval_statistics(docs, document_metadata)
            
            return docs, retrieved_context, document_metadata
            
        except Exception as e:
            self.retrieval_stats['retrieval_errors'] += 1
            error_msg = f"Error during Polish law document retrieval: {e}"
            self.logger.error(error_msg)
            return [], f"ERROR: {error_msg}", []
    
    def _process_retrieved_polish_documents(self, docs: List) -> Tuple[List[str], List[Dict]]:
        """
        Process retrieved Polish law documents and extract metadata for analysis.
        
        This method analyzes each retrieved document to understand its metadata
        quality and creates formatted context pieces for LLM processing, with
        special attention to Polish law-specific features.
        """
        context_pieces = []
        document_metadata = []
        
        for i, doc in enumerate(docs):
            metadata = doc.metadata
            
            # Extract and validate metadata structure with Polish law specifics
            has_enhanced = metadata.get('has_enhanced_structure', False)
            complexity = metadata.get('complexity_level', 'unknown')
            section_number = metadata.get('section_number', '')
            gazette_ref = metadata.get('gazette_reference', '')
            
            # Track enhanced metadata and Polish law-specific statistics
            if has_enhanced:
                self.retrieval_stats['enhanced_metadata_documents'] += 1
            
            if section_number:
                self.retrieval_stats['sections_found'] += 1
            
            if gazette_ref:
                self.retrieval_stats['gazette_references_found'] += 1
            
            # Create document metadata entry for further processing
            doc_metadata = {
                'index': i,
                'metadata': metadata,
                'content': doc.page_content,
                'has_enhanced_structure': has_enhanced,
                'complexity_level': complexity,
                'section_number': section_number,
                'gazette_reference': gazette_ref
            }
            document_metadata.append(doc_metadata)
            
            # Build enhanced reference for context display with Polish law elements
            reference = self._build_polish_law_document_reference(metadata, has_enhanced, complexity)
            
            # Create formatted context piece with Polish law identifier
            context_piece = f"[Polish Law Document {i+1} - {reference}]\n{doc.page_content}"
            context_pieces.append(context_piece)
            
            # Log document details for verification
            self._log_polish_law_document_details(i, metadata, has_enhanced, complexity, doc.page_content)
        
        return context_pieces, document_metadata
    
    def _build_polish_law_document_reference(self, metadata: Dict[str, Any], 
                                           has_enhanced: bool, complexity: str) -> str:
        """
        Build a comprehensive reference string for a Polish law document.
        
        This creates a human-readable reference that shows the document's
        structure and metadata quality, including Polish law-specific elements
        like sections and gazette references.
        """
        article_num = metadata.get('article_number', 'N/A')
        doc_type = metadata.get('type', 'unknown')
        chapter_num = metadata.get('chapter_number', 'N/A')
        section_num = metadata.get('section_number', '')
        paragraph_count = metadata.get('paragraph_count', 0)
        gazette_ref = metadata.get('gazette_reference', '')
        
        # Build base reference
        reference = f"Polish Data Protection Law - Article {article_num}"
        
        # Add chapter information
        if metadata.get('chapter_title'):
            reference += f" (Chapter {chapter_num}: {metadata['chapter_title']})"
        
        # Add section information (unique to Polish law structure)
        if section_num:
            section_title = metadata.get('section_title', '')
            if section_title:
                reference += f", Section {section_num}: {section_title}"
            else:
                reference += f", Section {section_num}"
        
        reference += f" - {doc_type}"
        
        # Add enhancement information
        if has_enhanced:
            reference += f" [Enhanced: {complexity}, {paragraph_count}p]"
        
        # Add gazette reference if available (important for Polish law authenticity)
        if gazette_ref:
            reference += f" [Dz.U.: {gazette_ref}]"
        
        return reference
    
    def _log_polish_law_document_details(self, index: int, metadata: Dict[str, Any], 
                                       has_enhanced: bool, complexity: str, content: str) -> None:
        """
        Log detailed information about a retrieved Polish law document.
        
        This provides comprehensive visibility into what documents were retrieved
        and their metadata quality, including Polish law-specific features.
        """
        article_num = metadata.get('article_number', 'N/A')
        doc_type = metadata.get('type', 'unknown')
        chapter_num = metadata.get('chapter_number', 'N/A')
        section_num = metadata.get('section_number', 'N/A')
        paragraph_count = metadata.get('paragraph_count', 0)
        gazette_ref = metadata.get('gazette_reference', 'N/A')
        
        self.logger.info(f"Polish Law Document {index+1}: Article {article_num} ({doc_type}), "
                        f"Chapter {chapter_num}, Section {section_num}, "
                        f"Enhanced: {'✓' if has_enhanced else '✗'}, "
                        f"Complexity: {complexity}, Paragraphs: {paragraph_count}, "
                        f"Gazette: {gazette_ref}, Content: {len(content)} chars")
        
        # Log content preview for verification (only for first few documents)
        if index < 3:
            content_preview = content[:150].replace('\n', ' ')
            self.logger.debug(f"Polish Law Document {index+1} preview: {content_preview}...")
    
    def _log_polish_law_retrieval_statistics(self, docs: List, document_metadata: List[Dict]) -> None:
        """
        Log comprehensive statistics about the Polish law retrieval operation.
        
        This provides insights into the quality and patterns of retrieved documents
        to help optimize the retrieval and citation processes for Polish law.
        """
        enhanced_count = sum(1 for doc_meta in document_metadata 
                           if doc_meta.get('has_enhanced_structure', False))
        
        sections_count = sum(1 for doc_meta in document_metadata 
                           if doc_meta.get('section_number'))
        
        gazette_refs_count = sum(1 for doc_meta in document_metadata 
                               if doc_meta.get('gazette_reference'))
        
        # Calculate complexity distribution
        complexity_distribution = {}
        for doc_meta in document_metadata:
            complexity = doc_meta.get('complexity_level', 'unknown')
            complexity_distribution[complexity] = complexity_distribution.get(complexity, 0) + 1
        
        # Calculate article and type distributions
        article_counts = {}
        type_counts = {}
        section_counts = {}
        
        for doc in docs:
            article = doc.metadata.get('article_number', 'unknown')
            doc_type = doc.metadata.get('type', 'unknown')
            section = doc.metadata.get('section_number', 'none')
            
            article_counts[article] = article_counts.get(article, 0) + 1
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
            section_counts[section] = section_counts.get(section, 0) + 1
        
        enhancement_rate = (enhanced_count / len(docs) * 100) if docs else 0
        section_coverage_rate = (sections_count / len(docs) * 100) if docs else 0
        gazette_coverage_rate = (gazette_refs_count / len(docs) * 100) if docs else 0
        
        # Log comprehensive statistics including Polish law-specific metrics
        self.logger.info("=" * 80)
        self.logger.info("POLISH LAW DOCUMENT RETRIEVAL STATISTICS")
        self.logger.info("=" * 80)
        self.logger.info(f"Documents retrieved: {len(docs)}")
        self.logger.info(f"Enhanced metadata: {enhanced_count}/{len(docs)} documents")
        self.logger.info(f"Enhancement rate: {enhancement_rate:.1f}%")
        self.logger.info(f"Section coverage: {sections_count}/{len(docs)} documents ({section_coverage_rate:.1f}%)")
        self.logger.info(f"Gazette reference coverage: {gazette_refs_count}/{len(docs)} documents ({gazette_coverage_rate:.1f}%)")
        self.logger.info(f"Complexity distribution: {dict(complexity_distribution)}")
        self.logger.info(f"Article distribution: {dict(article_counts)}")
        self.logger.info(f"Document type distribution: {dict(type_counts)}")
        self.logger.info(f"Section distribution: {dict(section_counts)}")
    
    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about all Polish law retrieval operations.
        
        Returns:
            Dictionary containing detailed retrieval statistics including Polish law-specific metrics
        """
        stats = dict(self.retrieval_stats)
        
        # Calculate enhancement rate across all retrievals
        if stats['total_documents_retrieved'] > 0:
            enhancement_rate = (stats['enhanced_metadata_documents'] / stats['total_documents_retrieved']) * 100
            stats['enhancement_rate_percent'] = round(enhancement_rate, 1)
            
            section_rate = (stats['sections_found'] / stats['total_documents_retrieved']) * 100
            stats['section_coverage_rate_percent'] = round(section_rate, 1)
            
            gazette_rate = (stats['gazette_references_found'] / stats['total_documents_retrieved']) * 100
            stats['gazette_reference_rate_percent'] = round(gazette_rate, 1)
        else:
            stats['enhancement_rate_percent'] = 0
            stats['section_coverage_rate_percent'] = 0
            stats['gazette_reference_rate_percent'] = 0
        
        return stats
    
    def is_connected(self) -> bool:
        """
        Check if the Polish law vector store connection is active.
        
        Returns:
            True if connected and ready for operations, False otherwise
        """
        return self.polish_law_db is not None


def create_polish_law_vector_store_connector(db_path: str, logger: logging.Logger) -> PolishLawVectorStoreConnector:
    """
    Factory function to create a configured Polish law vector store connector.
    
    This provides a clean interface for creating connector instances with
    proper dependency injection of configuration and logger.
    """
    return PolishLawVectorStoreConnector(db_path, logger)