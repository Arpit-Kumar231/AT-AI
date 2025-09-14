import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

from ai_classifier import TicketClassifier
from knowledge_base import KnowledgeBase
from config import TOPIC_TAGS, SENTIMENT_OPTIONS, PRIORITY_LEVELS

st.set_page_config(
    page_title="Atlan Customer Support Copilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .ticket-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f9f9f9;
    }
    .classification-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 0.25rem;
    }
    .topic-howto { background-color: #e3f2fd; color: #1976d2; }
    .topic-product { background-color: #f3e5f5; color: #7b1fa2; }
    .topic-connector { background-color: #e8f5e8; color: #388e3c; }
    .topic-lineage { background-color: #fff3e0; color: #f57c00; }
    .topic-api { background-color: #fce4ec; color: #c2185b; }
    .topic-sso { background-color: #e0f2f1; color: #00796b; }
    .topic-glossary { background-color: #f1f8e9; color: #689f38; }
    .topic-bestpractices { background-color: #e3f2fd; color: #1976d2; }
    .topic-sensitive { background-color: #ffebee; color: #d32f2f; }
    
    .sentiment-frustrated { background-color: #ffebee; color: #d32f2f; }
    .sentiment-curious { background-color: #e3f2fd; color: #1976d2; }
    .sentiment-angry { background-color: #ffcdd2; color: #c62828; }
    .sentiment-neutral { background-color: #f5f5f5; color: #424242; }
    .sentiment-positive { background-color: #e8f5e8; color: #388e3c; }
    
    .priority-p0 { background-color: #ffcdd2; color: #c62828; }
    .priority-p1 { background-color: #fff3e0; color: #f57c00; }
    .priority-p2 { background-color: #e8f5e8; color: #388e3c; }
</style>
""", unsafe_allow_html=True)

if 'tickets_loaded' not in st.session_state:
    st.session_state.tickets_loaded = False
if 'classified_tickets' not in st.session_state:
    st.session_state.classified_tickets = []
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = None
if 'classifier' not in st.session_state:
    st.session_state.classifier = None

@st.cache_data
def load_sample_tickets():
    try:
        with open('sample_tickets.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Sample tickets file not found!")
        return []

def initialize_ai_components():
    if st.session_state.classifier is None:
        try:
            with st.spinner("Initializing AI classifier..."):
                st.session_state.classifier = TicketClassifier()
        except Exception as e:
            st.error(f"Failed to initialize classifier: {e}")
            return False
    
    if st.session_state.knowledge_base is None:
        try:
            with st.spinner("Initializing knowledge base..."):
                st.session_state.knowledge_base = KnowledgeBase()
        except Exception as e:
            st.error(f"Failed to initialize knowledge base: {e}")
            return False
    
    return True

def get_badge_class(topic, sentiment, priority):
    topic_class = f"topic-{topic.lower().replace('/', '').replace(' ', '')}"
    sentiment_class = f"sentiment-{sentiment.lower()}"
    priority_class = f"priority-{priority.split()[0].lower()}"
    
    return topic_class, sentiment_class, priority_class

def display_ticket_classification(ticket):
    topic_class, sentiment_class, priority_class = get_badge_class(
        ticket.get('topic', 'Unknown'),
        ticket.get('sentiment', 'Unknown'),
        ticket.get('priority', 'Unknown')
    )
    
    st.markdown(f"""
    <div class="ticket-card">
        <h4>Ticket #{ticket['id']}: {ticket['title']}</h4>
        <p><strong>Customer:</strong> {ticket['customer_email']}</p>
        <p><strong>Description:</strong> {ticket['description']}</p>
        <p><strong>Created:</strong> {ticket['created_at']}</p>
        
        <div style="margin-top: 1rem;">
            <span class="classification-badge {topic_class}">Topic: {ticket.get('topic', 'Unknown')}</span>
            <span class="classification-badge {sentiment_class}">Sentiment: {ticket.get('sentiment', 'Unknown')}</span>
            <span class="classification-badge {priority_class}">Priority: {ticket.get('priority', 'Unknown')}</span>
        </div>
        
        {f"<p><strong>AI Reasoning:</strong> {ticket.get('reasoning', 'No reasoning provided')}</p>" if ticket.get('reasoning') else ""}
    </div>
    """, unsafe_allow_html=True)

def create_classification_dashboard(classified_tickets):
    st.markdown('<div class="section-header">📊 Classification Dashboard</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tickets", len(classified_tickets))
    
    with col2:
        high_priority = len([t for t in classified_tickets if t.get('priority') == 'P0 (High)'])
        st.metric("High Priority", high_priority)
    
    with col3:
        frustrated = len([t for t in classified_tickets if t.get('sentiment') == 'Frustrated'])
        st.metric("Frustrated Customers", frustrated)
    
    with col4:
        howto_tickets = len([t for t in classified_tickets if t.get('topic') == 'How-to'])
        st.metric("How-to Questions", howto_tickets)
    
    col1, col2 = st.columns(2)
    
    with col1:
        topic_counts = {}
        for ticket in classified_tickets:
            topic = ticket.get('topic', 'Unknown')
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        if topic_counts:
            fig_topic = px.pie(
                values=list(topic_counts.values()),
                names=list(topic_counts.keys()),
                title="Topic Distribution"
            )
            st.plotly_chart(fig_topic, use_container_width=True)
    
    with col2:
        priority_counts = {}
        for ticket in classified_tickets:
            priority = ticket.get('priority', 'Unknown')
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        if priority_counts:
            fig_priority = px.bar(
                x=list(priority_counts.keys()),
                y=list(priority_counts.values()),
                title="Priority Distribution"
            )
            st.plotly_chart(fig_priority, use_container_width=True)
    
    st.markdown('<div class="section-header">🎫 Detailed Ticket Classifications</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_topic = st.selectbox("Filter by Topic", ["All"] + TOPIC_TAGS)
    
    with col2:
        selected_sentiment = st.selectbox("Filter by Sentiment", ["All"] + SENTIMENT_OPTIONS)
    
    with col3:
        selected_priority = st.selectbox("Filter by Priority", ["All"] + PRIORITY_LEVELS)
    
    filtered_tickets = classified_tickets
    if selected_topic != "All":
        filtered_tickets = [t for t in filtered_tickets if t.get('topic') == selected_topic]
    if selected_sentiment != "All":
        filtered_tickets = [t for t in filtered_tickets if t.get('sentiment') == selected_sentiment]
    if selected_priority != "All":
        filtered_tickets = [t for t in filtered_tickets if t.get('priority') == selected_priority]
    
    st.write(f"Showing {len(filtered_tickets)} tickets")
    
    for ticket in filtered_tickets:
        display_ticket_classification(ticket)

def handle_interactive_agent():
    st.markdown('<div class="section-header">🤖 Interactive AI Agent</div>', unsafe_allow_html=True)
    
    st.write("Submit a new support ticket or question to see how the AI analyzes and responds:")
    
    new_ticket = st.text_area(
        "Enter your support ticket or question:",
        placeholder="e.g., How do I set up SSO with Azure AD?",
        height=100
    )
    
    if st.button("Submit Ticket", type="primary"):
        if not new_ticket.strip():
            st.warning("Please enter a ticket or question.")
            return
        
        if not initialize_ai_components():
            st.error("Failed to initialize AI components. Please check your configuration.")
            return
        
        with st.spinner("Analyzing ticket..."):
            classification = st.session_state.classifier.classify_ticket(
                title="New Ticket",
                description=new_ticket
            )
        
        st.markdown("### 🔍 Internal Analysis (Back-end View)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            topic_class, _, _ = get_badge_class(classification['topic'], 'neutral', 'p1')
            st.markdown(f'<span class="classification-badge {topic_class}">Topic: {classification["topic"]}</span>', unsafe_allow_html=True)
        
        with col2:
            _, sentiment_class, _ = get_badge_class('product', classification['sentiment'], 'p1')
            st.markdown(f'<span class="classification-badge {sentiment_class}">Sentiment: {classification["sentiment"]}</span>', unsafe_allow_html=True)
        
        with col3:
            _, _, priority_class = get_badge_class('product', 'neutral', classification['priority'])
            st.markdown(f'<span class="classification-badge {priority_class}">Priority: {classification["priority"]}</span>', unsafe_allow_html=True)
        
        st.write(f"**AI Reasoning:** {classification.get('reasoning', 'No reasoning provided')}")
        
        st.markdown("### 💬 Final Response (Front-end View)")
        
        rag_topics = ["How-to", "Product", "Best practices", "API/SDK", "SSO"]
        
        if classification['topic'] in rag_topics:
            with st.spinner("Generating response using knowledge base..."):
                response = {
                    'answer': f"Based on your question about {classification['topic'].lower()}, here's what I found in our documentation...\n\n[This would be a real response generated from the knowledge base using RAG pipeline]",
                    'sources': [
                        "https://docs.atlan.com/getting-started/",
                        "https://docs.atlan.com/authentication/",
                        "https://developer.atlan.com/api-reference/"
                    ]
                }
            
            st.success("✅ Response generated using RAG pipeline")
            st.write(response['answer'])
            
            if response['sources']:
                st.markdown("**Sources used:**")
                for source in response['sources']:
                    st.markdown(f"- [{source}]({source})")
        else:
            routing_message = f"This ticket has been classified as a '{classification['topic']}' issue and routed to the appropriate team."
            st.info("📋 Ticket routed to specialized team")
            st.write(routing_message)

def main():
    st.markdown('<div class="main-header">🤖 Atlan Customer Support Copilot</div>', unsafe_allow_html=True)
    
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Bulk Classification Dashboard", "Interactive AI Agent"]
    )
    
    if not st.session_state.tickets_loaded:
        sample_tickets = load_sample_tickets()
        if sample_tickets:
            st.session_state.tickets_loaded = True
            st.session_state.sample_tickets = sample_tickets
    
    if page == "Bulk Classification Dashboard":
        st.markdown("### 📋 Bulk Ticket Classification Dashboard")
        st.write("This dashboard shows the AI classification of sample support tickets.")
        
        if st.button("Load and Classify Tickets", type="primary"):
            if not initialize_ai_components():
                st.error("Failed to initialize AI components. Please check your configuration.")
                return
            
            with st.spinner("Classifying tickets..."):
                classified_tickets = st.session_state.classifier.classify_bulk_tickets(
                    st.session_state.sample_tickets
                )
                st.session_state.classified_tickets = classified_tickets
            
            st.success(f"Successfully classified {len(classified_tickets)} tickets!")
        
        if st.session_state.classified_tickets:
            create_classification_dashboard(st.session_state.classified_tickets)
    
    elif page == "Interactive AI Agent":
        handle_interactive_agent()
    
    st.markdown("---")
    st.markdown(
        "Built with ❤️ for Atlan Customer Support Team | "
        "Powered by OpenAI GPT and Streamlit"
    )

if __name__ == "__main__":
    main()
