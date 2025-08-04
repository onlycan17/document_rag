#!/usr/bin/env python3
"""Test script to diagnose RAG system behavior with '우각형파수편' query"""

import os
import sys
import logging

# Add project root to path  
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.vectorstore import VectorDatabase
from src.rag import RAGChain
from src.utils import TextProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_query_expansion():
    """Test how the query is expanded"""
    query = "우각형파수편"
    
    print(f"\n{'='*80}")
    print(f"Testing query expansion for: '{query}'")
    print(f"{'='*80}\n")
    
    # Test static expansion
    static_expanded = TextProcessor.expand_query(query, use_static_expansion=True, use_dynamic_expansion=False)
    print(f"Static expansion: '{static_expanded}'")
    
    # Test dynamic expansion
    try:
        dynamic_expanded = TextProcessor.expand_query(query, use_static_expansion=False, use_dynamic_expansion=True)
        print(f"Dynamic expansion: '{dynamic_expanded}'")
    except Exception as e:
        print(f"Dynamic expansion failed: {e}")
    
    # Test keyword extraction
    keywords = TextProcessor.extract_keywords(query)
    print(f"Extracted keywords: {keywords}")
    
    print()

def test_vector_search():
    """Test vector database search"""
    query = "우각형파수편"
    
    print(f"\n{'='*80}")
    print(f"Testing vector database search for: '{query}'")
    print(f"{'='*80}\n")
    
    try:
        vector_db = VectorDatabase()
        doc_count = vector_db.get_document_count()
        print(f"Document count in vector DB: {doc_count}")
        
        if doc_count == 0:
            print("No documents in vector database!")
            return
        
        # Search for the term
        results = vector_db.search(query, k=10)
        print(f"\nFound {len(results)} results\n")
        
        # Show top 3 results
        for i, (doc, score) in enumerate(results[:3]):
            print(f"Result {i+1}:")
            print(f"  Score: {score}")
            print(f"  File: {doc.metadata.get('file_name', 'Unknown')}")
            print(f"  Chunk: {doc.metadata.get('chunk_id', 'Unknown')}")
            print(f"  Content preview: {doc.page_content[:300]}...")
            print(f"  {'='*60}")
            
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        import traceback
        traceback.print_exc()

def test_rag_query():
    """Test full RAG query"""
    query = "우각형파수편"
    
    print(f"\n{'='*80}")
    print(f"Testing full RAG query for: '{query}'")
    print(f"{'='*80}\n")
    
    try:
        rag_chain = RAGChain()
        
        # Get the answer
        result = rag_chain.query(query)
        
        print("Status:", result.get('status'))
        print("\nAnswer:")
        print(result.get('answer', 'No answer'))
        
        print(f"\nSources ({len(result.get('sources', []))} documents):")
        for source in result.get('sources', [])[:3]:
            print(f"  - {source.get('file_name')} (relevance: {source.get('relevance_percent')}%)")
            
        print("\nSearch info:")
        search_info = result.get('search_info', {})
        print(f"  Original query: {search_info.get('original_query')}")
        print(f"  Processed query: {search_info.get('processed_query')[:100]}...")
        print(f"  Documents in DB: {search_info.get('total_documents_in_db')}")
        print(f"  Search results: {search_info.get('search_results_count')}")
        
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        import traceback
        traceback.print_exc()

def search_in_documents():
    """Search for the term in actual document content"""
    query = "우각형파수편"
    
    print(f"\n{'='*80}")
    print(f"Searching for '{query}' in document chunks")
    print(f"{'='*80}\n")
    
    try:
        vector_db = VectorDatabase()
        
        # Search through all documents
        found_count = 0
        for i, doc in enumerate(vector_db.documents_cache[:1000]):  # Check first 1000 docs
            if query in doc.page_content:
                found_count += 1
                print(f"\nFound in document {i}:")
                print(f"  File: {doc.metadata.get('file_name', 'Unknown')}")
                print(f"  Chunk: {doc.metadata.get('chunk_id', 'Unknown')}")
                
                # Show context around the term
                content = doc.page_content
                index = content.find(query)
                start = max(0, index - 100)
                end = min(len(content), index + len(query) + 100)
                context = content[start:end]
                
                print(f"  Context: ...{context}...")
                
                if found_count >= 5:  # Show max 5 occurrences
                    break
        
        if found_count == 0:
            print(f"Term '{query}' not found in any document chunks")
        else:
            print(f"\nTotal occurrences found: {found_count}")
            
    except Exception as e:
        logger.error(f"Document search failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("RAG System Diagnostic for '우각형파수편' Query")
    print("=" * 80)
    
    # Run all tests
    test_query_expansion()
    test_vector_search()
    search_in_documents()
    test_rag_query()