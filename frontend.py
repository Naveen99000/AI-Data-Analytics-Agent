"""
AI Data Analytics Frontend - With PDF Download
Streamlit frontend with multi-LLM support and PDF export
"""

import streamlit as st
import requests
import pandas as pd
import json
import plotly.graph_objects as go
from typing import Dict, Any, List
import time

# ============================================================================
# CONFIGURATION
# ============================================================================
API_URL = "http://localhost:8000/api"

st.set_page_config(
    page_title="AI Data Analytics Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .insight-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        background-color: #f8f9fa;
        border-left: 5px solid #667eea;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #667eea;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def display_metric_card(label: str, value: Any):
    """Display a metric card"""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

def render_chart(chart_json: str):
    """Render a Plotly chart from JSON"""
    try:
        fig = go.Figure(json.loads(chart_json))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error rendering chart: {e}")

def download_pdf_report(dataset_id: str):
    """Download PDF report"""
    try:
        response = requests.get(f"{API_URL}/export/pdf/{dataset_id}")
        
        if response.status_code == 200:
            return response.content
        else:
            st.error("Failed to generate PDF report")
            return None
    except Exception as e:
        st.error(f"Error downloading report: {e}")
        return None

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Data Analytics Platform</h1>', unsafe_allow_html=True)
    
    # Check backend status
    try:
        status_response = requests.get(f"{API_URL.replace('/api', '')}")
        if status_response.status_code == 200:
            status_data = status_response.json()
            st.success(f"✅ Connected to backend | LLM: {status_data.get('llm_provider', 'Unknown')} ({status_data.get('model', 'Unknown')})")
        else:
            st.warning("⚠️ Backend connection issue")
    except:
        st.error("❌ Cannot connect to backend. Make sure it's running on http://localhost:8000")
        return
    
    st.markdown("**Upload your data, get instant insights, and ask questions in natural language**")
    
    # Initialize session state
    if 'dataset_id' not in st.session_state:
        st.session_state.dataset_id = None
    if 'metadata' not in st.session_state:
        st.session_state.metadata = None
    if 'insights' not in st.session_state:
        st.session_state.insights = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # ========================================================================
    # SIDEBAR - FILE UPLOAD
    # ========================================================================
    with st.sidebar:
        st.header("📁 Upload Your Data")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls', 'json', 'parquet', 'txt', 'tsv'],
            help="Supported formats: CSV, Excel, JSON, Parquet, TSV"
        )
        
        if uploaded_file:
            with st.spinner("🔄 Processing your data..."):
                try:
                    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{API_URL}/upload/file", files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.dataset_id = data['dataset_id']
                        st.session_state.metadata = data['metadata']
                        st.session_state.insights = data['insights']
                        st.session_state.sample_data = data['sample_data']
                        st.session_state.cleaning_report = data['clean_report']
                        
                        st.success(f"✅ Successfully uploaded: {data['filename']}")
                        
                        # Display quick stats
                        st.markdown("### 📊 Quick Stats")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Rows", f"{data['metadata']['rows']:,}")
                        with col2:
                            st.metric("Columns", data['metadata']['columns'])
                        
                        # Show cleaning report
                        with st.expander("🧹 Cleaning Report"):
                            for step in data['clean_report']['steps_performed']:
                                st.write(f"✓ {step}")
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                
                except Exception as e:
                    st.error(f"Error uploading file: {str(e)}")
        
        st.markdown("---")
        
        # PDF Download Button
        if st.session_state.dataset_id:
            st.markdown("### 📄 Export Report")
            
            if st.button("📥 Download PDF Report", use_container_width=True):
                with st.spinner("📄 Generating PDF report..."):
                    pdf_content = download_pdf_report(st.session_state.dataset_id)
                    
                    if pdf_content:
                        st.download_button(
                            label="💾 Save PDF",
                            data=pdf_content,
                            file_name=f"analysis_report_{time.strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                        st.success("✅ Report generated successfully!")
        
        st.markdown("---")
        
        # Display uploaded datasets
        if st.session_state.dataset_id:
            st.markdown("### 📂 Current Dataset")
            st.info(f"**Dataset ID:** {st.session_state.dataset_id[:8]}...")
            
            if st.button("🗑️ Clear Dataset"):
                st.session_state.dataset_id = None
                st.session_state.metadata = None
                st.session_state.insights = []
                st.session_state.chat_history = []
                st.rerun()
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        **Features:**
        - 📤 Upload any data format
        - 🧹 Auto data cleaning
        - 💡 AI-powered insights
        - 💬 Natural language queries
        - 📊 Custom visualizations
        - 📄 PDF report export
        
        **Powered by:**
        - FastAPI Backend
        - Streamlit Frontend
        - Multiple LLM Support
        """)
    
    # ========================================================================
    # MAIN CONTENT
    # ========================================================================
    
    if st.session_state.dataset_id:
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["💡 Auto Insights", "💬 Ask Questions", "📊 Data Explorer"])
        
        # ====================================================================
        # TAB 1: AUTO INSIGHTS
        # ====================================================================
        with tab1:
            st.header("🔍 Automated Insights")
            st.markdown("AI-generated insights from your data")
            
            # Summary metrics
            if st.session_state.metadata:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    display_metric_card("Total Rows", f"{st.session_state.metadata['rows']:,}")
                
                with col2:
                    display_metric_card("Columns", st.session_state.metadata['columns'])
                
                with col3:
                    display_metric_card("Numeric Cols", len(st.session_state.metadata['numeric_columns']))
                
                with col4:
                    display_metric_card("Date Cols", len(st.session_state.metadata['datetime_columns']))
            
            st.markdown("---")
            
            # Display insights
            if st.session_state.insights:
                st.markdown(f"### 💡 Key Insights ({len(st.session_state.insights)})")
                
                for idx, insight in enumerate(st.session_state.insights):
                    st.markdown(f"""
                    <div class="insight-card">
                        <p><strong>{idx + 1}.</strong> {insight}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No insights generated yet. Upload a dataset to get started!")
            
            # Suggested visualizations
            st.markdown("---")
            st.header("📈 Suggested Visualizations")
            
            with st.spinner("Generating chart suggestions..."):
                try:
                    response = requests.get(f"{API_URL}/chart/suggest/{st.session_state.dataset_id}")
                    
                    if response.status_code == 200:
                        charts_data = response.json()
                        suggestions = charts_data.get('suggestions', [])
                        
                        if suggestions:
                            st.markdown(f"Found {len(suggestions)} chart suggestions")
                            
                            for idx, suggestion in enumerate(suggestions):
                                with st.expander(f"📊 {suggestion.get('description', f'Chart {idx+1}')}"):
                                    st.write(f"**Type:** {suggestion.get('type')}")
                                    st.write(f"**X Column:** {suggestion.get('x')}")
                                    if suggestion.get('y'):
                                        st.write(f"**Y Column:** {suggestion.get('y')}")
                                    
                                    if st.button(f"Generate Chart {idx+1}", key=f"gen_chart_{idx}"):
                                        with st.spinner("Creating chart..."):
                                            chart_response = requests.post(
                                                f"{API_URL}/chart",
                                                json={
                                                    'dataset_id': st.session_state.dataset_id,
                                                    'chart_type': suggestion.get('type'),
                                                    'x_column': suggestion.get('x'),
                                                    'y_column': suggestion.get('y')
                                                }
                                            )
                                            
                                            if chart_response.status_code == 200:
                                                chart_result = chart_response.json()
                                                if chart_result.get('success'):
                                                    render_chart(chart_result['chart_json'])
                                            else:
                                                st.error("Could not generate chart")
                        else:
                            st.info("No chart suggestions available")
                    else:
                        st.warning("Could not load suggested charts")
                
                except Exception as e:
                    st.error(f"Error loading charts: {e}")
        
        # ====================================================================
        # TAB 2: ASK QUESTIONS
        # ====================================================================
        with tab2:
            st.header("💬 Ask Anything About Your Data")
            st.markdown("Use natural language to analyze your data")
            
            # Get AI suggestions
            if st.button("💡 Get Question Suggestions"):
                with st.spinner("Generating suggestions..."):
                    try:
                        response = requests.get(f"{API_URL}/suggest/{st.session_state.dataset_id}")
                        if response.status_code == 200:
                            suggestions = response.json().get('suggestions', [])
                            st.markdown("### Suggested Questions:")
                            for i, suggestion in enumerate(suggestions, 1):
                                st.markdown(f"{i}. {suggestion}")
                    except Exception as e:
                        st.error(f"Error getting suggestions: {e}")
            
            # Example queries
            with st.expander("💡 Example Questions"):
                st.markdown("""
                - What is the average revenue by month?
                - Show me the top 10 customers by sales
                - What's the correlation between price and quantity?
                - Are there any outliers in the revenue column?
                - Show month over month growth trend
                - Compare sales across different regions
                - What are the key trends in this data?
                - Which category has the highest sales?
                """)
            
            # Chat interface
            st.markdown("---")
            
            # Display chat history in a container
            chat_container = st.container()
            with chat_container:
                for chat in st.session_state.chat_history:
                    with st.chat_message("user"):
                        st.write(chat['query'])
                    
                    with st.chat_message("assistant"):
                        st.write(chat['response'])
                        if chat.get('chart'):
                            render_chart(chat['chart'])
            
            # Clear chat button
            if st.session_state.chat_history:
                if st.button("🗑️ Clear Chat History"):
                    st.session_state.chat_history = []
                    st.rerun()
        
        # ====================================================================
        # TAB 3: DATA EXPLORER
        # ====================================================================
        with tab3:
            st.header("📊 Data Explorer")
            
            # Show sample data
            st.subheader("📋 Sample Data")
            if st.session_state.get('sample_data'):
                df_sample = pd.DataFrame(st.session_state.sample_data)
                st.dataframe(df_sample, use_container_width=True)
            
            st.markdown("---")
            
            # Column information
            st.subheader("📑 Column Information")
            if st.session_state.metadata:
                col_info = []
                for col, dtype in st.session_state.metadata['column_types'].items():
                    missing = st.session_state.metadata['missing_values'].get(col, 0)
                    col_info.append({
                        'Column': col,
                        'Type': dtype,
                        'Missing Values': missing
                    })
                
                st.dataframe(pd.DataFrame(col_info), use_container_width=True)
            
            st.markdown("---")
            
            # Custom chart builder
            st.subheader("🎨 Custom Chart Builder")
            
            col1, col2 = st.columns(2)
            
            with col1:
                chart_type = st.selectbox(
                    "Chart Type",
                    options=['bar', 'line', 'scatter', 'pie', 'histogram', 'box', 'heatmap']
                )
            
            with col2:
                all_columns = list(st.session_state.metadata['column_types'].keys())
                x_column = st.selectbox("X Column", options=all_columns)
            
            # Y column (optional for some chart types)
            if chart_type not in ['pie', 'histogram', 'heatmap']:
                numeric_cols = st.session_state.metadata['numeric_columns']
                y_column = st.selectbox("Y Column", options=numeric_cols)
            else:
                y_column = None
            
            if st.button("🎨 Generate Chart"):
                with st.spinner("Creating chart..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/chart",
                            json={
                                'dataset_id': st.session_state.dataset_id,
                                'chart_type': chart_type,
                                'x_column': x_column,
                                'y_column': y_column
                            }
                        )
                        
                        if response.status_code == 200:
                            chart_data = response.json()
                            if chart_data.get('success'):
                                render_chart(chart_data['chart_json'])
                            else:
                                st.error(chart_data.get('error', 'Could not generate chart'))
                        else:
                            st.error("Could not generate chart")
                    
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        # ====================================================================
        # CHAT INPUT - OUTSIDE TABS (IMPORTANT!)
        # ====================================================================
        st.markdown("---")
        st.markdown("### 💬 Ask a Question")
        user_query = st.chat_input("Type your question here...")
        
        if user_query:
            # Process query
            with st.spinner("🤔 Analyzing..."):
                try:
                    response = requests.post(
                        f"{API_URL}/query",
                        json={
                            'dataset_id': st.session_state.dataset_id,
                            'question': user_query
                        }
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display result
                        answer = result['result']['answer']
                        
                        # Save to history
                        st.session_state.chat_history.append({
                            'query': user_query,
                            'response': answer
                        })
                        
                        st.rerun()
                    else:
                        st.error("Sorry, I couldn't process that question.")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    else:
        # Welcome screen
        st.markdown("## 👋 Welcome!")
        st.markdown("""
        ### Get started by uploading your data
        
        Upload your data file using the sidebar and:
        
        1. **📁 Upload Data** - CSV, Excel, JSON, Parquet, TSV
        2. **🧹 Auto-Clean** - Automatic data cleaning
        3. **💡 Get Insights** - AI-generated insights
        4. **💬 Ask Questions** - Natural language queries
        5. **📊 Visualize** - Auto-generated charts
        6. **📄 Export PDF** - Download comprehensive report
        
        ---
        
        #### 🚀 Quick Start:
        
        1. Click **"Browse files"** in the sidebar
        2. Upload your data file (any format)
        3. Explore **Auto Insights**
        4. Ask questions in **Ask Questions** tab
        5. Download PDF report from sidebar
        """)
        
        # Feature showcase
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🤖 AI-Powered
            Multiple LLM support (Groq, Gemini, OpenAI, Grok)
            """)
        
        with col2:
            st.markdown("""
            ### 📊 Smart Analytics
            Auto-insights, trends, anomalies, correlations
            """)
        
        with col3:
            st.markdown("""
            ### 📄 PDF Reports
            Professional reports with charts and statistics
            """)

# ============================================================================
# RUN APP
# ============================================================================
if __name__ == "__main__":
    main()
