import requests
from bs4 import BeautifulSoup
import openai
import json
import re
from typing import List, Dict, Tuple
from urllib.parse import urljoin, urlparse
import time
from config import OPENAI_API_KEY, KNOWLEDGE_BASE_URLS, EMBEDDING_MODEL

class KnowledgeBase:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required. Please set it in your environment variables.")
        
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.embedding_model = EMBEDDING_MODEL
        self.knowledge_base = {}
        self.embeddings = {}
    
    def scrape_documentation(self, base_url: str, max_pages: int = 10) -> List[Dict]:
        scraped_content = []
        visited_urls = set()
        
        try:
            response = requests.get(base_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            doc_links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(base_url, href)
                
                if (self._is_doc_page(full_url, base_url) and 
                    full_url not in visited_urls and 
                    len(doc_links) < max_pages):
                    doc_links.append(full_url)
            
            for url in doc_links[:max_pages]:
                if url not in visited_urls:
                    content = self._scrape_page(url)
                    if content:
                        scraped_content.append(content)
                        visited_urls.add(url)
                    time.sleep(1)
            
        except Exception as e:
            print(f"Error scraping {base_url}: {e}")
        
        return scraped_content
    
    def _is_doc_page(self, url: str, base_url: str) -> bool:
        parsed_url = urlparse(url)
        parsed_base = urlparse(base_url)
        
        if parsed_url.netloc != parsed_base.netloc:
            return False
        
        skip_patterns = ['.pdf', '.zip', '.jpg', '.png', '.gif', '/api/', '/search']
        for pattern in skip_patterns:
            if pattern in url.lower():
                return False
        
        return True
    
    def _scrape_page(self, url: str) -> Dict:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for script in soup(["script", "style"]):
                script.decompose()
            
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "Untitled"
            
            content_selectors = [
                'main', 'article', '.content', '.documentation', 
                '.docs-content', '#content', '.page-content'
            ]
            
            content_text = ""
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    content_text = content_elem.get_text().strip()
                    break
            
            if not content_text:
                body = soup.find('body')
                if body:
                    content_text = body.get_text().strip()
            
            content_text = re.sub(r'\s+', ' ', content_text)
            content_text = content_text[:2000]
            
            if content_text and len(content_text) > 100:
                return {
                    'url': url,
                    'title': title_text,
                    'content': content_text,
                    'scraped_at': time.time()
                }
            
        except Exception as e:
            print(f"Error scraping page {url}: {e}")
        
        return None
    
    def create_embeddings(self, content_list: List[Dict]) -> None:
        for item in content_list:
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=item['content']
                )
                
                self.embeddings[item['url']] = {
                    'embedding': response.data[0].embedding,
                    'content': item['content'],
                    'title': item['title']
                }
                
            except Exception as e:
                print(f"Error creating embedding for {item['url']}: {e}")
    
    def search_relevant_content(self, query: str, top_k: int = 3) -> List[Dict]:
        if not self.embeddings:
            return []
        
        try:
            query_response = self.client.embeddings.create(
                model=self.embedding_model,
                input=query
            )
            query_embedding = query_response.data[0].embedding
            
            similarities = []
            for url, data in self.embeddings.items():
                similarity = self._cosine_similarity(query_embedding, data['embedding'])
                similarities.append({
                    'url': url,
                    'title': data['title'],
                    'content': data['content'],
                    'similarity': similarity
                })
            
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            print(f"Error searching content: {e}")
            return []
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        import numpy as np
        
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)
    
    def generate_answer(self, query: str, topic: str) -> Dict:
        relevant_content = self.search_relevant_content(query, top_k=3)
        
        if not relevant_content:
            return {
                'answer': "I couldn't find relevant information in the knowledge base to answer your question. Please contact our support team for assistance.",
                'sources': []
            }
        
        context = "\n\n".join([
            f"Source: {item['title']}\nContent: {item['content'][:500]}..."
            for item in relevant_content
        ])
        
        try:
            prompt = f"""
You are a helpful customer support agent for Atlan, a data catalog platform. 
Answer the user's question based on the provided context from the documentation.

User Question: {query}
Topic: {topic}

Context from Documentation:
{context}

Please provide a helpful, accurate answer based on the context. If the context doesn't contain enough information to fully answer the question, say so and suggest contacting support.

Answer:
"""
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful customer support agent for Atlan."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content.strip()
            
            sources = [item['url'] for item in relevant_content]
            
            return {
                'answer': answer,
                'sources': sources
            }
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return {
                'answer': "I encountered an error while generating an answer. Please try again or contact support.",
                'sources': []
            }
    
    def initialize_knowledge_base(self) -> None:
        print("Initializing knowledge base...")
        
        all_content = []
        
        print("Scraping Atlan documentation...")
        docs_content = self.scrape_documentation(KNOWLEDGE_BASE_URLS["docs"], max_pages=5)
        all_content.extend(docs_content)
        
        print("Scraping developer hub...")
        dev_content = self.scrape_documentation(KNOWLEDGE_BASE_URLS["developer"], max_pages=5)
        all_content.extend(dev_content)
        
        print("Creating embeddings...")
        self.create_embeddings(all_content)
        
        print(f"Knowledge base initialized with {len(self.embeddings)} documents.")

if __name__ == "__main__":
    kb = KnowledgeBase()
    kb.initialize_knowledge_base()
    
    results = kb.search_relevant_content("How to create a workspace?", top_k=2)
    print("Search Results:")
    for result in results:
        print(f"Title: {result['title']}")
        print(f"Similarity: {result['similarity']:.3f}")
        print(f"URL: {result['url']}")
        print("---")

