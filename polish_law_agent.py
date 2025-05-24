import os
import re
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate

class PolishLawAgent:
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Load both vector stores
        self.gdpr_db = Chroma(
            persist_directory=os.path.join(os.path.dirname(__file__), "data/chroma_db"),
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
        )
        
        self.polish_law_db = Chroma(
            persist_directory=os.path.join(os.path.dirname(__file__), "data/polish_law_db"),
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
            collection_name="polish_data_protection_law"
        )
        
        # Prompt for analysis after seeing all the context (true RAG)
        self.rag_prompt = ChatPromptTemplate.from_template(
            """You are a specialized Polish data protection law expert.
            
            User Query: {user_query}
            
            Based on the following retrieved Polish law content, identify the most relevant provisions:
            
            Retrieved Context:
            {retrieved_context}
            
            For each relevant citation you identify, provide:
            1. The specific article/paragraph number
            2. A direct quote of the relevant text from the retrieved context
            3. A brief explanation of its relevance to the query and how it relates to GDPR
            
            Important: Only cite information that appears in the retrieved context above. 
            If the context doesn't contain sufficient information, say so clearly.
            
            Format your response as a structured list of citations in this exact format:
            
            CITATION 1:
            - Article: [Number and title]
            - Quote: "[Direct quote from retrieved context]"
            - Relevance: [Brief explanation including GDPR relationship]
            
            CITATION 2:
            - Article: [Number and title]
            - Quote: "[Direct quote from retrieved context]"
            - Relevance: [Brief explanation including GDPR relationship]
            """
        )
        
        # Prompt for the final analysis combining GDPR and Polish law
        self.analysis_prompt = ChatPromptTemplate.from_template(
            """You are a specialized Polish data protection law expert.
            
            Given the user query: {user_query}
            
            GDPR context:
            {gdpr_citations}
            
            Polish implementation context:
            {polish_law_citations}
            
            Provide a comprehensive response that:
            1. Explains the GDPR requirements
            2. Details the specific Polish implementation laws
            3. Highlights any differences or unique aspects of Polish implementation
            4. Provides practical compliance guidance
            
            Format with clear sections and specific article references.
            """
        )
    
    def _retrieve_relevant_documents(self, query: str, k: int = 5) -> tuple:
        """Retrieve relevant documents and format them for the LLM."""
        # Search for relevant documents
        docs = self.polish_law_db.similarity_search(
            query, 
            k=k,
            filter={"law": "polish_data_protection"}
        )
        
        # Create formatted context
        context_pieces = []
        for i, doc in enumerate(docs):
            doc_type = doc.metadata.get('type', 'content')
            article_num = doc.metadata.get('article_number', '')
            paragraph_num = doc.metadata.get('paragraph_number', '')
            gdpr_mapping = doc.metadata.get('gdpr_mapping', '')
            
            # Add metadata to make citations more meaningful
            reference = f"Polish Data Protection Act - Article {article_num}"
            if paragraph_num:
                reference += f", Paragraph {paragraph_num}"
            if gdpr_mapping:
                reference += f" (corresponds to GDPR Article(s) {gdpr_mapping})"
                
            context_piece = f"[Document {i+1} - {reference} - {doc_type}]\n{doc.page_content}"
            context_pieces.append(context_piece)
        
        retrieved_context = "\n\n" + "="*80 + "\n\n".join(context_pieces)
        
        return docs, retrieved_context
    
    def _parse_llm_response_to_citations(self, llm_response: str) -> List[Dict[str, Any]]:
        """Parse the LLM response back into structured citations."""
        citations = []
        
        # Split by "CITATION" to find individual citations
        citation_blocks = llm_response.split("CITATION ")[1:]  # Skip the first empty split
        
        for block in citation_blocks:
            try:
                lines = block.strip().split('\n')
                citation = {
                    "article": "",
                    "quote": "",
                    "explanation": ""
                }
                
                # Parse each line to extract structured information
                for line in lines:
                    line = line.strip()
                    if line.startswith("- Article:"):
                        citation["article"] = line.replace("- Article:", "").strip()
                    elif line.startswith("- Quote:"):
                        quote = line.replace("- Quote:", "").strip()
                        # Remove surrounding quotes if present
                        if quote.startswith('"') and quote.endswith('"'):
                            quote = quote[1:-1]
                        citation["quote"] = quote
                    elif line.startswith("- Relevance:"):
                        citation["explanation"] = line.replace("- Relevance:", "").strip()
                
                # Only add citation if we have the essential information
                if citation["article"] and citation["quote"] and citation["explanation"]:
                    citations.append(citation)
                    
            except Exception as e:
                print(f"Warning: Could not parse citation block: {e}")
                continue
        
        return citations
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the user query with both GDPR and Polish law context using true RAG."""
        print("\n📜 [STEP 2/4] POLISH LAW AGENT: Analyzing Polish law implementation...")
        user_query = state["user_query"]
        gdpr_citations = state["gdpr_citations"]
        
        try:
            # Step 1: Retrieve relevant documents (true RAG approach)
            retrieved_docs, retrieved_context = self._retrieve_relevant_documents(user_query)
            
            # Step 2: Use LLM to analyze retrieved content and extract citations
            rag_chain = self.rag_prompt | self.model
            rag_response = rag_chain.invoke({
                "user_query": user_query,
                "retrieved_context": retrieved_context
            })
            
            # Step 3: Parse the LLM response back into structured citations
            polish_law_citations = self._parse_llm_response_to_citations(rag_response.content)
            
            # Fallback: if parsing fails, create a basic citation structure
            if not polish_law_citations:
                print("   Warning: Could not parse structured citations, creating fallback")
                polish_law_citations = [{
                    "article": "Polish Data Protection Act",
                    "quote": "Multiple relevant provisions found",
                    "explanation": "Polish law implementation of GDPR requirements"
                }]
            
            # Step 4: Generate the final analysis comparing GDPR and Polish law
            analysis_chain = self.analysis_prompt | self.model
            
            # Format the citations for the prompt
            gdpr_citations_text = "\n\n".join([
                f"{cite['article']}:\n{cite['quote']}\n(Relevance: {cite['explanation']})"
                for cite in gdpr_citations
            ])
            
            polish_citations_text = "\n\n".join([
                f"{cite['article']}:\n{cite['quote']}\n(Relevance: {cite['explanation']})"
                for cite in polish_law_citations
            ])
            
            analysis_response = analysis_chain.invoke({
                "user_query": user_query,
                "gdpr_citations": gdpr_citations_text,
                "polish_law_citations": polish_citations_text
            })
            
            # Update state with Polish law citations and analysis
            state["polish_law_citations"] = polish_law_citations
            state["polish_law_analysis"] = analysis_response.content
            
            print(f"✅ Completed: Found {len(polish_law_citations)} relevant Polish law provisions")
            
        except Exception as e:
            print(f"❌ Error in Polish law RAG processing: {e}")
            # Fallback to ensure the workflow continues
            state["polish_law_citations"] = [{
                "article": "System Error",
                "quote": "Could not retrieve Polish law information",
                "explanation": f"Error occurred during RAG retrieval: {str(e)}"
            }]
        
        return state