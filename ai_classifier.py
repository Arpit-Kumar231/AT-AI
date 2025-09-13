import openai
import json
import re
from typing import Dict, List, Tuple
from config import OPENAI_API_KEY, TOPIC_TAGS, SENTIMENT_OPTIONS, PRIORITY_LEVELS, DEFAULT_MODEL

class TicketClassifier:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required. Please set it in your environment variables.")
        
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.model = DEFAULT_MODEL
    
    def classify_ticket(self, title: str, description: str) -> Dict:
        try:
            prompt = self._create_classification_prompt(title, description)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert customer support ticket classifier for Atlan, a data catalog platform."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            classification_text = response.choices[0].message.content
            return self._parse_classification(classification_text)
            
        except Exception as e:
            print(f"Error in classification: {e}")
            return self._get_default_classification()
    
    def _create_classification_prompt(self, title: str, description: str) -> str:
        return f"""
Please classify the following customer support ticket for Atlan (a data catalog platform):

Title: {title}
Description: {description}

Classify this ticket according to the following criteria:

1. TOPIC TAGS (choose the most relevant one):
   - How-to: Questions about how to use features
   - Product: General product questions or feature requests
   - Connector: Issues with data source connectors (Snowflake, BigQuery, etc.)
   - Lineage: Data lineage visualization or tracking issues
   - API/SDK: Questions about API usage or SDK integration
   - SSO: Single Sign-On authentication issues
   - Glossary: Business glossary or metadata management
   - Best practices: Questions about recommended practices
   - Sensitive data: Data privacy, security, or compliance issues

2. SENTIMENT (choose the most appropriate):
   - Frustrated: Customer is experiencing issues and showing frustration
   - Curious: Customer is asking questions to learn more
   - Angry: Customer is clearly upset or angry
   - Neutral: Customer is matter-of-fact, no strong emotion
   - Positive: Customer is happy or expressing satisfaction

3. PRIORITY (choose based on urgency and impact):
   - P0 (High): Critical issues affecting business operations, security issues, or major bugs
   - P1 (Medium): Important issues that need attention but not immediately critical
   - P2 (Low): General questions, feature requests, or minor issues

Please respond in the following JSON format:
{{
    "topic": "chosen_topic",
    "sentiment": "chosen_sentiment", 
    "priority": "chosen_priority",
    "reasoning": "brief explanation of your classification"
}}
"""
    
    def _parse_classification(self, classification_text: str) -> Dict:
        try:
            json_match = re.search(r'\{.*\}', classification_text, re.DOTALL)
            if json_match:
                classification = json.loads(json_match.group())
                
                if (classification.get("topic") in TOPIC_TAGS and 
                    classification.get("sentiment") in SENTIMENT_OPTIONS and
                    classification.get("priority") in PRIORITY_LEVELS):
                    return classification
            
            return self._get_default_classification()
            
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"Error parsing classification: {e}")
            return self._get_default_classification()
    
    def _get_default_classification(self) -> Dict:
        return {
            "topic": "Product",
            "sentiment": "Neutral",
            "priority": "P1 (Medium)",
            "reasoning": "Default classification due to parsing error"
        }
    
    def classify_bulk_tickets(self, tickets: List[Dict]) -> List[Dict]:
        classified_tickets = []
        
        for ticket in tickets:
            classification = self.classify_ticket(
                ticket.get("title", ""),
                ticket.get("description", "")
            )
            
            classified_ticket = ticket.copy()
            classified_ticket.update(classification)
            classified_tickets.append(classified_ticket)
        
        return classified_tickets

if __name__ == "__main__":
    classifier = TicketClassifier()
    
    test_ticket = {
        "title": "How do I create a new workspace?",
        "description": "I'm new to Atlan and need help setting up my first workspace."
    }
    
    result = classifier.classify_ticket(
        test_ticket["title"], 
        test_ticket["description"]
    )
    
    print("Classification Result:")
    print(json.dumps(result, indent=2))
