import os

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DEFAULT_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"

KNOWLEDGE_BASE_URLS = {
    "docs": "https://docs.atlan.com",
    "developer": "https://developer.atlan.com"
}

TOPIC_TAGS = [
    "How-to",
    "Product", 
    "Connector",
    "Lineage",
    "API/SDK",
    "SSO",
    "Glossary",
    "Best practices",
    "Sensitive data"
]

SENTIMENT_OPTIONS = [
    "Frustrated",
    "Curious", 
    "Angry",
    "Neutral",
    "Positive"
]

PRIORITY_LEVELS = [
    "P0 (High)",
    "P1 (Medium)", 
    "P2 (Low)"
]
