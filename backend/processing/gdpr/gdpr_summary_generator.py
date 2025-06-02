"""
GDPR Processing Summary Generator

This module creates comprehensive summaries of the enhanced GDPR document processing
pipeline. It aggregates statistics from all the processing stages and generates
detailed reports about the success rate, quality metrics, and performance of the
metadata flattening approach.

The summary generator demonstrates the "Observer" pattern - it watches what happens
in the other modules and reports on the overall system health and performance.
"""

import json
import logging
from typing import Dict, List, Any, Set
from datetime import datetime
from langchain.docstore.document import Document


class GDPRProcessingSummaryGenerator:
    """
    Generates comprehensive summaries of the GDPR document processing pipeline.
    
    This class aggregates statistics and metrics from all stages of the processing
    pipeline to create detailed reports about the effectiveness of the metadata
    flattening approach and overall system performance.
    
    The summary generator provides valuable insights into:
    - How well the metadata flattening worked
    - Quality metrics for the processed documents
    - Performance statistics for each processing stage
    - Recommendations for optimization or issue resolution
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize the summary generator.
        
        Args:
            logger: Configured logger for tracking summary generation
        """
        self.logger = logger
        self.logger.info("GDPR Processing Summary Generator initialized")
    
    def generate_comprehensive_summary(self, docs: List[Document], 
                                     processing_timestamp: str,
                                     loader_stats: Dict[str, Any] = None,
                                     flattener_stats: Dict[str, Any] = None,
                                     converter_stats: Dict[str, Any] = None,
                                     embedder_stats: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of the entire GDPR processing pipeline.
        
        This method creates a detailed report that aggregates information from all
        processing stages to provide a complete picture of how well the enhanced
        processing worked.
        
        Args:
            docs: Final processed documents
            processing_timestamp: Timestamp of the processing session
            loader_stats: Statistics from the document loader (optional)
            flattener_stats: Statistics from the metadata flattener (optional)
            converter_stats: Statistics from the document converter (optional)
            embedder_stats: Statistics from the embedder (optional)
            
        Returns:
            Comprehensive summary dictionary with detailed statistics and analysis
        """
        self.logger.info("Generating comprehensive GDPR processing summary...")
        
        summary = {
            "processing_timestamp": processing_timestamp,
            "generation_timestamp": datetime.now().isoformat(),
            "document_type": "gdpr_regulation",
            "pipeline_overview": self._create_pipeline_overview(docs, loader_stats, converter_stats, embedder_stats),
            "enhanced_metadata_analysis": self._analyze_enhanced_metadata(docs),
            "document_quality_metrics": self._calculate_quality_metrics(docs),
            "processing_stage_performance": self._analyze_stage_performance(loader_stats, flattener_stats, converter_stats, embedder_stats),
            "structural_analysis": self._perform_structural_analysis(docs),
            "recommendations": self._generate_recommendations(docs, flattener_stats, embedder_stats),
            "sample_enhanced_documents": self._collect_sample_documents(docs)
        }
        
        self.logger.info("Comprehensive GDPR processing summary generated successfully")
        return summary
    
    def _create_pipeline_overview(self, docs: List[Document], 
                                 loader_stats: Dict[str, Any],
                                 converter_stats: Dict[str, Any],
                                 embedder_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a high-level overview of the entire processing pipeline.
        
        This provides key metrics that give a quick assessment of how well
        the processing worked overall.
        """
        overview = {
            "total_documents_processed": len(docs),
            "pipeline_stages_completed": 0,
            "overall_success_rate": 0.0,
            "enhanced_metadata_rate": 0.0,
            "critical_errors": []
        }
        
        # Count completed pipeline stages
        stages_completed = [
            loader_stats is not None,
            converter_stats is not None, 
            embedder_stats is not None
        ]
        overview["pipeline_stages_completed"] = sum(stages_completed)
        
        # Calculate overall success rate
        if converter_stats:
            total_chunks = converter_stats.get('total_chunks', 0)
            successful_conversions = converter_stats.get('successful_conversions', 0)
            if total_chunks > 0:
                overview["overall_success_rate"] = (successful_conversions / total_chunks) * 100
        
        # Calculate enhanced metadata rate
        enhanced_count = sum(1 for doc in docs if doc.metadata.get('has_enhanced_structure', False))
        if docs:
            overview["enhanced_metadata_rate"] = (enhanced_count / len(docs)) * 100
        
        # Check for critical errors
        if embedder_stats and embedder_stats.get('metadata_errors', 0) > 0:
            overview["critical_errors"].append("Metadata compatibility issues detected")
        
        return overview
    
    def _analyze_enhanced_metadata(self, docs: List[Document]) -> Dict[str, Any]:
        """
        Analyze the enhanced metadata across all processed documents.
        
        This provides detailed insights into how well the metadata flattening
        approach worked and what types of enhanced structures were preserved.
        """
        analysis = {
            "total_documents": len(docs),
            "enhanced_structure_count": 0,
            "simple_documents": 0,
            "complexity_distribution": {},
            "numbering_styles": {},
            "paragraph_statistics": {
                "min_paragraphs": float('inf'),
                "max_paragraphs": 0,
                "total_paragraphs": 0,
                "articles_with_sub_paragraphs": 0
            },
            "flattening_effectiveness": {
                "json_preservation_rate": 0.0,
                "structural_indicators_rate": 0.0
            }
        }
        
        # Track various metadata characteristics
        json_preserved_count = 0
        structural_indicators_count = 0
        
        for doc in docs:
            metadata = doc.metadata
            
            # Check for enhanced structure
            if metadata.get('has_enhanced_structure', False):
                analysis["enhanced_structure_count"] += 1
                
                # Analyze complexity distribution
                complexity = metadata.get('complexity_level', 'unknown')
                analysis["complexity_distribution"][complexity] = \
                    analysis["complexity_distribution"].get(complexity, 0) + 1
                
                # Analyze numbering styles
                numbering_style = metadata.get('numbering_style', '')
                if numbering_style:
                    analysis["numbering_styles"][numbering_style] = \
                        analysis["numbering_styles"].get(numbering_style, 0) + 1
                
                # Analyze paragraph statistics
                self._update_paragraph_statistics(metadata, analysis["paragraph_statistics"])
                
                # Check flattening effectiveness
                if metadata.get('article_structure_json', ''):
                    json_preserved_count += 1
                
                if any(metadata.get(field, 0) > 0 for field in ['paragraph_count', 'has_sub_paragraphs']):
                    structural_indicators_count += 1
                    
            else:
                analysis["simple_documents"] += 1
        
        # Calculate flattening effectiveness rates
        if analysis["enhanced_structure_count"] > 0:
            analysis["flattening_effectiveness"]["json_preservation_rate"] = \
                (json_preserved_count / analysis["enhanced_structure_count"]) * 100
            analysis["flattening_effectiveness"]["structural_indicators_rate"] = \
                (structural_indicators_count / analysis["enhanced_structure_count"]) * 100
        
        # Fix infinite values in paragraph statistics
        if analysis["paragraph_statistics"]["min_paragraphs"] == float('inf'):
            analysis["paragraph_statistics"]["min_paragraphs"] = 0
        
        return analysis
    
    def _update_paragraph_statistics(self, metadata: Dict[str, Any], stats: Dict[str, Any]) -> None:
        """
        Update paragraph statistics with information from a single document.
        
        This tracks detailed information about the structure of processed articles
        to understand the complexity and patterns in the GDPR regulation.
        """
        para_count = metadata.get('paragraph_count', 0)
        if para_count > 0:
            stats["min_paragraphs"] = min(stats["min_paragraphs"], para_count)
            stats["max_paragraphs"] = max(stats["max_paragraphs"], para_count)
            stats["total_paragraphs"] += para_count
            
            if metadata.get('has_sub_paragraphs', False):
                stats["articles_with_sub_paragraphs"] += 1
    
    def _calculate_quality_metrics(self, docs: List[Document]) -> Dict[str, Any]:
        """
        Calculate various quality metrics for the processed documents.
        
        These metrics help assess the overall quality of the processing pipeline
        and identify any issues that might affect the citation system's performance.
        """
        metrics = {
            "content_quality": {
                "average_content_length": 0,
                "min_content_length": float('inf'),
                "max_content_length": 0,
                "empty_content_count": 0
            },
            "metadata_completeness": {
                "articles_with_numbers": 0,
                "articles_with_titles": 0,
                "chapters_identified": 0,
                "complete_metadata_rate": 0.0
            },
            "document_types": {},
            "articles_processed": set(),
            "chapters_processed": set()
        }
        
        total_content_length = 0
        complete_metadata_count = 0
        
        for doc in docs:
            # Analyze content quality
            content_length = len(doc.page_content)
            total_content_length += content_length
            
            if content_length == 0:
                metrics["content_quality"]["empty_content_count"] += 1
            
            metrics["content_quality"]["min_content_length"] = \
                min(metrics["content_quality"]["min_content_length"], content_length)
            metrics["content_quality"]["max_content_length"] = \
                max(metrics["content_quality"]["max_content_length"], content_length)
            
            # Analyze metadata completeness
            metadata = doc.metadata
            
            if metadata.get('article_number'):
                metrics["metadata_completeness"]["articles_with_numbers"] += 1
                metrics["articles_processed"].add(metadata['article_number'])
            
            if metadata.get('article_title'):
                metrics["metadata_completeness"]["articles_with_titles"] += 1
            
            if metadata.get('chapter_number'):
                metrics["metadata_completeness"]["chapters_identified"] += 1
                metrics["chapters_processed"].add(metadata['chapter_number'])
            
            # Check for complete metadata (has article number, title, and content)
            if (metadata.get('article_number') and 
                metadata.get('article_title') and 
                content_length > 0):
                complete_metadata_count += 1
            
            # Track document types
            doc_type = metadata.get('type', 'unknown')
            metrics["document_types"][doc_type] = metrics["document_types"].get(doc_type, 0) + 1
        
        # Calculate averages and rates
        if docs:
            metrics["content_quality"]["average_content_length"] = total_content_length / len(docs)
            metrics["metadata_completeness"]["complete_metadata_rate"] = \
                (complete_metadata_count / len(docs)) * 100
        
        # Fix infinite values
        if metrics["content_quality"]["min_content_length"] == float('inf'):
            metrics["content_quality"]["min_content_length"] = 0
        
        # Convert sets to counts for JSON serialization
        metrics["unique_articles_count"] = len(metrics["articles_processed"])
        metrics["unique_chapters_count"] = len(metrics["chapters_processed"])
        metrics["articles_processed"] = sorted(list(metrics["articles_processed"]))
        metrics["chapters_processed"] = sorted(list(metrics["chapters_processed"]))
        
        return metrics
    
    def _analyze_stage_performance(self, loader_stats: Dict[str, Any],
                                  flattener_stats: Dict[str, Any],
                                  converter_stats: Dict[str, Any],
                                  embedder_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the performance of each stage in the processing pipeline.
        
        This provides insights into which stages worked well and which might
        need optimization or attention.
        """
        performance = {
            "loader_performance": self._analyze_loader_performance(loader_stats),
            "flattener_performance": self._analyze_flattener_performance(flattener_stats),
            "converter_performance": self._analyze_converter_performance(converter_stats),
            "embedder_performance": self._analyze_embedder_performance(embedder_stats)
        }
        
        return performance
    
    def _analyze_loader_performance(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze document loader performance metrics."""
        if not stats:
            return {"status": "no_statistics_available"}
        
        return {
            "status": "completed",
            "files_processed": 1,  # Single JSON file processed
            "validation_errors": len(stats.get('validation_errors', [])),
            "chunks_validated": stats.get('total_chunks', 0),
            "enhancement_rate": stats.get('enhancement_rate_percent', 0)
        }
    
    def _analyze_flattener_performance(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze metadata flattener performance metrics."""
        if not stats:
            return {"status": "no_statistics_available"}
        
        return {
            "status": "completed",
            "structures_processed": stats.get('total_processed', 0),
            "enhanced_structures_found": stats.get('enhanced_structures_found', 0),
            "enhancement_rate": stats.get('enhancement_rate_percent', 0),
            "flattening_errors": stats.get('flattening_errors', 0),
            "complexity_distribution": stats.get('complexity_distribution', {}),
            "numbering_styles_found": stats.get('numbering_styles_found', [])
        }
    
    def _analyze_converter_performance(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze document converter performance metrics."""
        if not stats:
            return {"status": "no_statistics_available"}
        
        success_rate = 0
        if stats.get('total_chunks', 0) > 0:
            success_rate = (stats.get('successful_conversions', 0) / stats['total_chunks']) * 100
        
        return {
            "status": "completed",
            "total_chunks": stats.get('total_chunks', 0),
            "successful_conversions": stats.get('successful_conversions', 0),
            "success_rate": success_rate,
            "conversion_errors": stats.get('errors', 0),
            "enhanced_structure_count": stats.get('enhanced_structure_count', 0),
            "chunk_types": stats.get('chunk_types', {}),
            "complexity_levels": stats.get('complexity_levels', {})
        }
    
    def _analyze_embedder_performance(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze vector store embedder performance metrics."""
        if not stats:
            return {"status": "no_statistics_available"}
        
        return {
            "status": "completed",
            "total_documents": stats.get('total_documents', 0),
            "successful_batches": stats.get('successful_batches', 0),
            "failed_batches": stats.get('failed_batches', 0),
            "metadata_compatibility_tested": stats.get('metadata_compatibility_tested', False),
            "metadata_errors": stats.get('metadata_errors', 0),
            "retry_attempts": stats.get('retry_attempts', 0)
        }
    
    def _perform_structural_analysis(self, docs: List[Document]) -> Dict[str, Any]:
        """
        Perform detailed structural analysis of the processed GDPR documents.
        
        This analysis helps understand the patterns and organization of the
        GDPR regulation as represented in the processed documents.
        """
        analysis = {
            "article_coverage": {},
            "chapter_coverage": {},
            "structural_patterns": {},
            "citation_readiness": {}
        }
        
        # Analyze article and chapter coverage
        articles_by_chapter = {}
        article_complexity = {}
        
        for doc in docs:
            metadata = doc.metadata
            
            article_num = metadata.get('article_number', '')
            chapter_num = metadata.get('chapter_number', '')
            
            if article_num:
                # Track articles by chapter
                if chapter_num:
                    if chapter_num not in articles_by_chapter:
                        articles_by_chapter[chapter_num] = set()
                    articles_by_chapter[chapter_num].add(article_num)
                
                # Track article complexity
                if metadata.get('has_enhanced_structure', False):
                    complexity = metadata.get('complexity_level', 'unknown')
                    article_complexity[article_num] = complexity
        
        # Convert sets to lists for JSON serialization
        analysis["article_coverage"] = {
            chapter: sorted(list(articles)) 
            for chapter, articles in articles_by_chapter.items()
        }
        
        analysis["chapter_coverage"] = {
            "total_chapters": len(articles_by_chapter),
            "chapters_with_articles": list(articles_by_chapter.keys())
        }
        
        analysis["structural_patterns"] = {
            "articles_by_complexity": self._group_articles_by_complexity(article_complexity),
            "complexity_distribution": self._calculate_complexity_distribution(article_complexity)
        }
        
        # Assess citation readiness
        analysis["citation_readiness"] = self._assess_citation_readiness(docs)
        
        return analysis
    
    def _group_articles_by_complexity(self, article_complexity: Dict[str, str]) -> Dict[str, List[str]]:
        """Group articles by their structural complexity level."""
        complexity_groups = {}
        
        for article, complexity in article_complexity.items():
            if complexity not in complexity_groups:
                complexity_groups[complexity] = []
            complexity_groups[complexity].append(article)
        
        # Sort articles within each complexity group
        for complexity in complexity_groups:
            complexity_groups[complexity].sort()
        
        return complexity_groups
    
    def _calculate_complexity_distribution(self, article_complexity: Dict[str, str]) -> Dict[str, float]:
        """Calculate the distribution of complexity levels across articles."""
        if not article_complexity:
            return {}
        
        complexity_counts = {}
        for complexity in article_complexity.values():
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
        
        total_articles = len(article_complexity)
        complexity_distribution = {
            complexity: (count / total_articles) * 100
            for complexity, count in complexity_counts.items()
        }
        
        return complexity_distribution
    
    def _assess_citation_readiness(self, docs: List[Document]) -> Dict[str, Any]:
        """
        Assess how ready the processed documents are for precise citation creation.
        
        This analysis evaluates whether the documents have the metadata structure
        needed for the sophisticated citation system to work effectively.
        """
        readiness = {
            "documents_ready_for_precise_citations": 0,
            "documents_with_basic_citations": 0,
            "documents_needing_improvement": 0,
            "readiness_rate": 0.0,
            "missing_capabilities": []
        }
        
        for doc in docs:
            metadata = doc.metadata
            
            # Check citation readiness criteria
            has_article = bool(metadata.get('article_number'))
            has_enhanced_structure = metadata.get('has_enhanced_structure', False)
            has_preserved_json = bool(metadata.get('article_structure_json'))
            has_complexity_info = bool(metadata.get('complexity_level'))
            
            if has_enhanced_structure and has_preserved_json and has_complexity_info:
                readiness["documents_ready_for_precise_citations"] += 1
            elif has_article:
                readiness["documents_with_basic_citations"] += 1
            else:
                readiness["documents_needing_improvement"] += 1
        
        # Calculate readiness rate
        if docs:
            readiness["readiness_rate"] = \
                (readiness["documents_ready_for_precise_citations"] / len(docs)) * 100
        
        return readiness
    
    def _generate_recommendations(self, docs: List[Document],
                                flattener_stats: Dict[str, Any],
                                embedder_stats: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on the processing results.
        
        This provides actionable insights for improving the processing pipeline
        or addressing any issues that were discovered.
        """
        recommendations = []
        
        # Check enhancement rate
        enhanced_count = sum(1 for doc in docs if doc.metadata.get('has_enhanced_structure', False))
        enhancement_rate = (enhanced_count / len(docs)) * 100 if docs else 0
        
        if enhancement_rate < 50:
            recommendations.append(
                f"Enhancement rate is {enhancement_rate:.1f}%. Consider reviewing the input "
                "JSON structure to ensure more documents contain enhanced metadata."
            )
        elif enhancement_rate > 90:
            recommendations.append(
                f"Excellent enhancement rate of {enhancement_rate:.1f}%. The metadata "
                "flattening approach is working very effectively."
            )
        
        # Check for metadata errors
        if embedder_stats and embedder_stats.get('metadata_errors', 0) > 0:
            error_count = embedder_stats['metadata_errors']
            recommendations.append(
                f"Found {error_count} metadata compatibility errors. Review the metadata "
                "flattening approach to ensure all complex structures are properly handled."
            )
        
        # Check flattening effectiveness
        if flattener_stats:
            flattening_errors = flattener_stats.get('flattening_errors', 0)
            if flattening_errors > 0:
                recommendations.append(
                    f"Found {flattening_errors} flattening errors. Consider adding more "
                    "robust error handling for edge cases in the metadata structure."
                )
        
        # Check citation readiness
        precise_citation_count = sum(1 for doc in docs 
                                   if doc.metadata.get('has_enhanced_structure', False) and
                                      doc.metadata.get('article_structure_json', ''))
        
        if precise_citation_count < enhanced_count:
            recommendations.append(
                "Some enhanced documents are missing preserved JSON structure. Ensure "
                "complete structure preservation for maximum citation precision."
            )
        
        if not recommendations:
            recommendations.append(
                "Processing completed successfully with no significant issues detected. "
                "The metadata flattening approach is working optimally."
            )
        
        return recommendations
    
    def _collect_sample_documents(self, docs: List[Document], max_samples: int = 5) -> List[Dict[str, Any]]:
        """
        Collect sample documents that demonstrate the enhanced processing results.
        
        These samples help understand what the processed documents look like
        and verify that the metadata flattening approach preserved the important information.
        """
        samples = []
        
        # Prioritize documents with enhanced structure for samples
        enhanced_docs = [doc for doc in docs if doc.metadata.get('has_enhanced_structure', False)]
        
        # If we don't have enough enhanced docs, include some basic ones
        sample_docs = enhanced_docs[:max_samples]
        if len(sample_docs) < max_samples:
            basic_docs = [doc for doc in docs if not doc.metadata.get('has_enhanced_structure', False)]
            sample_docs.extend(basic_docs[:max_samples - len(sample_docs)])
        
        for i, doc in enumerate(sample_docs):
            metadata = doc.metadata
            
            sample = {
                "sample_index": i + 1,
                "article_number": metadata.get('article_number', 'N/A'),
                "article_title": metadata.get('article_title', 'N/A'),
                "chapter_info": f"Chapter {metadata.get('chapter_number', 'N/A')}: {metadata.get('chapter_title', 'N/A')}",
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "enhanced_metadata": {
                    "has_enhanced_structure": metadata.get('has_enhanced_structure', False),
                    "paragraph_count": metadata.get('paragraph_count', 0),
                    "has_sub_paragraphs": metadata.get('has_sub_paragraphs', False),
                    "complexity_level": metadata.get('complexity_level', 'unknown'),
                    "numbering_style": metadata.get('numbering_style', ''),
                    "json_structure_preserved": bool(metadata.get('article_structure_json', ''))
                }
            }
            
            samples.append(sample)
        
        return samples
    
    def save_summary_to_file(self, summary: Dict[str, Any], output_path: str) -> None:
        """
        Save the comprehensive summary to a JSON file.
        
        This creates a permanent record of the processing results that can be
        referenced later for analysis, debugging, or optimization planning.
        """
        self.logger.info(f"Saving comprehensive GDPR processing summary to: {output_path}")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            # Log key summary statistics
            self._log_summary_highlights(summary)
            
        except Exception as e:
            self.logger.error(f"Error saving GDPR processing summary: {e}")
            raise
    
    def _log_summary_highlights(self, summary: Dict[str, Any]) -> None:
        """
        Log key highlights from the summary for immediate visibility.
        
        This provides a quick overview of the most important results without
        requiring users to read through the complete detailed summary.
        """
        overview = summary.get("pipeline_overview", {})
        metadata_analysis = summary.get("enhanced_metadata_analysis", {})
        
        self.logger.info("=== GDPR PROCESSING SUMMARY HIGHLIGHTS ===")
        self.logger.info(f"Total documents processed: {overview.get('total_documents_processed', 0)}")
        self.logger.info(f"Enhanced documents: {metadata_analysis.get('enhanced_structure_count', 0)}")
        self.logger.info(f"Enhancement rate: {metadata_analysis.get('enhanced_structure_count', 0) / overview.get('total_documents_processed', 1) * 100:.1f}%")
        self.logger.info(f"Overall success rate: {overview.get('overall_success_rate', 0):.1f}%")
        self.logger.info(f"Pipeline stages completed: {overview.get('pipeline_stages_completed', 0)}")
        
        recommendations = summary.get("recommendations", [])
        if recommendations:
            self.logger.info(f"Key recommendation: {recommendations[0]}")


def create_gdpr_summary_generator(logger: logging.Logger) -> GDPRProcessingSummaryGenerator:
    """
    Factory function to create a configured GDPR processing summary generator.
    
    This provides a clean interface for creating summary generator instances
    with proper dependency injection.
    """
    return GDPRProcessingSummaryGenerator(logger)