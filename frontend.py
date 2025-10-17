"""
AI Data Analytics Frontend - Enhanced Version
Features: Auto-suggestions, Rich UI, Dark mode, Multiple exports, Quality score, Sample datasets
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
# CUSTOM CSS WITH ANIMATIONS & GRADIENTS
# ============================================================================
st.markdown("""
<style>
    /* Main gradient background with animation */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Main header with gradient text */
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #FFD700 0%, #FFA500 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        animation: fadeInDown 1s ease-in;
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Animated insight cards */
    .insight-card {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        background: rgba(255, 255, 255, 0.95);
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        animation: slideInLeft 0.5s ease;
    }
    
    .insight-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Metric cards with gradient */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        text-align: center;
        color: white;
        transition: transform 0.3s ease;
        animation: fadeIn 1s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: bold;
        color: #FFD700;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .metric-label {
        font-size: 1.2rem;
        color: white;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Quality score badge */
    .quality-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-size: 1.5rem;
        font-weight: bold;
        color: white;
        margin: 1rem 0;
        animation: pulse 2s infinite;
    }
    
    .quality-excellent { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .quality-good { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .quality-fair { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
    .quality-poor { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    /* Suggested questions with animation */
    .suggested-question {
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        border-left: 4px solid #667eea;
        cursor: pointer;
        transition: all 0.3s ease;
        animation: fadeInUp 0.5s ease;
    }
    
    .suggested-question:hover {
        background: rgba(255, 255, 255, 1);
        transform: translateX(10px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Quick action buttons */
    .action-button {
        padding: 0.75rem 2rem;
        margin: 0.5rem;
        border-radius: 25px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    }
    
    .action-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Data preview table styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Typing effect for AI responses */
    .typing-text {
        animation: typing 3.5s steps(40, end);
        overflow: hidden;
        white-space: nowrap;
        border-right: 3px solid #667eea;
    }
    
    @keyframes typing {
        from { width: 0 }
        to { width: 100% }
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 0 20px 20px 0;
    }
    
    /* Content container */
    .content-container {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }
    
    /* Upload area */
    [data-testid="stFileUploader"] {
        border: 3px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.9);
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #764ba2;
        background: rgba(255, 255, 255, 1);
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def display_metric_card(label: str, value: Any):
    """Display animated metric card"""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

def display_quality_badge(score: float, grade: str):
    """Display quality score badge"""
    if score >= 90:
        badge_class = "quality-excellent"
    elif score >= 70:
        badge_class = "quality-good"
    elif score >= 50:
        badge_class = "quality-fair"
    else:
        badge_class = "quality-poor"
    
    st.markdown(f"""
    <div class="quality-badge {badge_class}">
        🏆 Quality Score: {score}/100 ({grade})
    </div>
    """, unsafe_allow_html=True)

def render_chart(chart_json: str):
    """Render a Plotly chart from JSON"""
    try:
        fig = go.Figure(json.loads(chart_json))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error rendering chart: {e}")

def download_file(dataset_id: str, format_type: str):
    """Download file in specified format"""
    try:
        response = requests.get(f"{API_URL}/export/{format_type}/{dataset_id}")
        if response.status_code == 200:
            return response.content
        return None
    except Exception as e:
        st.error(f"Error downloading: {e}")
        return None

def typing_effect(text: str, speed: float = 0.03):
    """Simulate typing effect"""
    placeholder = st.empty()
    displayed_text = ""
    for char in text:
        displayed_text += char
        placeholder.markdown(displayed_text)
        time.sleep(speed)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Animated header
    st.markdown('<h1 class="main-header">🤖 AI Data Analytics Platform</h1>', unsafe_allow_html=True)
    
    # Check backend status
    try:
        status_response = requests.get(f"{API_URL.replace('/api', '')}")
        if status_response.status_code == 200:
            status_data = status_response.json()
            st.success(f"✅ Connected | LLM: {status_data.get('llm_provider', 'Unknown')} ({status_data.get('model', 'Unknown')})")
        else:
            st.warning("⚠️ Backend connection issue")
    except:
        st.error("❌ Cannot connect to backend. Make sure it's running on http://localhost:8000")
        return
    
    # Initialize session state
    if 'dataset_id' not in st.session_state:
        st.session_state.dataset_id = None
    if 'metadata' not in st.session_state:
        st.session_state.metadata = None
    if 'insights' not in st.session_state:
        st.session_state.insights = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'quality_score' not in st.session_state:
        st.session_state.quality_score = None
    if 'suggested_questions' not in st.session_state:
        st.session_state.suggested_questions = []
    
    # ========================================================================
    # SIDEBAR
    # ========================================================================
    with st.sidebar:
        st.header("📁 Data Management")
        
        # Tab for Upload vs Sample Data
        upload_tab, sample_tab = st.tabs(["📤 Upload", "🎲 Samples"])
        
        with upload_tab:
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
                            st.session_state.quality_score = data['quality_score']
                            st.session_state.suggested_questions = data['suggested_questions']
                            
                            st.success(f"✅ {data['filename']}")
                            
                            # Quick stats
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Rows", f"{data['metadata']['rows']:,}")
                            with col2:
                                st.metric("Columns", data['metadata']['columns'])
                            
                            # Cleaning report
                            with st.expander("🧹 Cleaning Report"):
                                for step in data['clean_report']['steps_performed']:
                                    st.write(f"✓ {step}")
                        else:
                            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        with sample_tab:
            st.markdown("### 🎲 Load Sample Dataset")
            
            try:
                samples_response = requests.get(f"{API_URL}/samples")
                if samples_response.status_code == 200:
                    samples = samples_response.json()['samples']
                    
                    for sample in samples:
                        with st.expander(f"📊 {sample['name']}"):
                            st.write(sample['description'])
                            st.write(f"**Rows:** {sample['rows']}")
                            st.write(f"**Columns:** {sample['columns']}")
                            
                            if st.button(f"Load {sample['name']}", key=f"load_{sample['id']}"):
                                with st.spinner("Loading sample..."):
                                    load_response = requests.post(f"{API_URL}/samples/{sample['id']}")
                                    if load_response.status_code == 200:
                                        data = load_response.json()
                                        st.session_state.dataset_id = data['dataset_id']
                                        st.session_state.metadata = data['metadata']
                                        st.session_state.insights = data['insights']
                                        st.session_state.sample_data = data['sample_data']
                                        st.session_state.quality_score = data['quality_score']
                                        st.session_state.suggested_questions = data['suggested_questions']
                                        st.success("✅ Sample loaded!")
                                        st.rerun()
            except:
                st.error("Could not load samples")
        
        st.markdown("---")
        
        # Export section
        if st.session_state.dataset_id:
            st.markdown("### 💾 Export Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📄 PDF", use_container_width=True):
                    with st.spinner("Generating..."):
                        pdf_content = download_file(st.session_state.dataset_id, "pdf")
                        if pdf_content:
                            st.download_button(
                                "💾 Save PDF",
                                pdf_content,
                                f"report_{time.strftime('%Y%m%d')}.pdf",
                                "application/pdf",
                                use_container_width=True
                            )
                
                if st.button("📊 Excel", use_container_width=True):
                    with st.spinner("Generating..."):
                        excel_content = download_file(st.session_state.dataset_id, "excel")
                        if excel_content:
                            st.download_button(
                                "💾 Save Excel",
                                excel_content,
                                f"data_{time.strftime('%Y%m%d')}.xlsx",
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
            
            with col2:
                if st.button("📝 CSV", use_container_width=True):
                    with st.spinner("Generating..."):
                        csv_content = download_file(st.session_state.dataset_id, "csv")
                        if csv_content:
                            st.download_button(
                                "💾 Save CSV",
                                csv_content,
                                f"data_{time.strftime('%Y%m%d')}.csv",
                                "text/csv",
                                use_container_width=True
                            )
                
                if st.button("🔧 JSON", use_container_width=True):
                    with st.spinner("Generating..."):
                        json_content = download_file(st.session_state.dataset_id, "json")
                        if json_content:
                            st.download_button(
                                "💾 Save JSON",
                                json_content,
                                f"data_{time.strftime('%Y%m%d')}.json",
                                "application/json",
                                use_container_width=True
                            )
            
            st.markdown("---")
            
            if st.button("🗑️ Clear Dataset", use_container_width=True):
                st.session_state.dataset_id = None
                st.session_state.metadata = None
                st.session_state.insights = []
                st.session_state.chat_history = []
                st.session_state.quality_score = None
                st.session_state.suggested_questions = []
                st.rerun()
# Continuing from Part 1...

    # ========================================================================
    # MAIN CONTENT
    # ========================================================================
    
    if st.session_state.dataset_id:
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "💡 Auto Insights", 
            "💬 Ask Questions", 
            "📊 Data Explorer",
            "🔍 Column Analysis"
        ])
        
        # ====================================================================
        # TAB 1: AUTO INSIGHTS (Enhanced)
        # ====================================================================
        with tab1:
            st.markdown('<div class="content-container">', unsafe_allow_html=True)
            
            st.header("🔍 Automated Insights")
            
            # Quality Score Badge
            if st.session_state.quality_score:
                display_quality_badge(
                    st.session_state.quality_score['overall'],
                    st.session_state.quality_score['grade']
                )
                
                # Detailed quality metrics
                st.markdown("### 📊 Quality Breakdown")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Completeness", 
                        f"{st.session_state.quality_score['completeness']:.1f}%",
                        help="Percentage of non-missing values"
                    )
                
                with col2:
                    st.metric(
                        "Uniqueness", 
                        f"{st.session_state.quality_score['uniqueness']:.1f}%",
                        help="Percentage of non-duplicate rows"
                    )
                
                with col3:
                    st.metric(
                        "Consistency", 
                        f"{st.session_state.quality_score['consistency']:.1f}%",
                        help="Data type consistency"
                    )
                
                with col4:
                    st.metric(
                        "Validity", 
                        f"{st.session_state.quality_score['validity']:.1f}%",
                        help="Outlier detection score"
                    )
            
            st.markdown("---")
            
            # Summary metrics with gradient cards
            if st.session_state.metadata:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    display_metric_card("Total Rows", f"{st.session_state.metadata['rows']:,}")
                
                with col2:
                    display_metric_card("Columns", st.session_state.metadata['columns'])
                
                with col3:
                    display_metric_card("Numeric", len(st.session_state.metadata['numeric_columns']))
                
                with col4:
                    display_metric_card("Categorical", len(st.session_state.metadata['categorical_columns']))
            
            st.markdown("---")
            
            # Animated insights
            if st.session_state.insights:
                st.markdown("### 💡 Key Insights")
                
                for idx, insight in enumerate(st.session_state.insights):
                    st.markdown(f"""
                    <div class="insight-card" style="animation-delay: {idx * 0.1}s;">
                        <p><strong>{idx + 1}.</strong> {insight}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Suggested charts
            st.header("📈 Suggested Visualizations")
            
            with st.spinner("Loading chart suggestions..."):
                try:
                    response = requests.get(f"{API_URL}/chart/suggest/{st.session_state.dataset_id}")
                    
                    if response.status_code == 200:
                        charts_data = response.json()
                        suggestions = charts_data.get('suggestions', [])
                        
                        if suggestions:
                            for idx, suggestion in enumerate(suggestions):
                                with st.expander(f"📊 {suggestion.get('description', f'Chart {idx+1}')}"):
                                    col1, col2 = st.columns([3, 1])
                                    
                                    with col1:
                                        st.write(f"**Type:** {suggestion.get('type').title()}")
                                        st.write(f"**X:** {suggestion.get('x')}")
                                        if suggestion.get('y'):
                                            st.write(f"**Y:** {suggestion.get('y')}")
                                    
                                    with col2:
                                        if st.button("Generate", key=f"gen_chart_{idx}"):
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
                            st.info("No chart suggestions available")
                
                except Exception as e:
                    st.error(f"Error loading charts: {e}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # ====================================================================
        # TAB 2: ASK QUESTIONS (Enhanced with Auto-Suggestions)
        # ====================================================================
        with tab2:
            st.markdown('<div class="content-container">', unsafe_allow_html=True)
            
            st.header("💬 Ask Anything About Your Data")
            
            # Display auto-suggested questions prominently
            if st.session_state.suggested_questions:
                st.markdown("### ✨ Suggested Questions (Click to Ask)")
                st.markdown("*AI-generated questions based on your data*")
                
                for idx, question in enumerate(st.session_state.suggested_questions):
                    col1, col2 = st.columns([6, 1])
                    
                    with col1:
                        st.markdown(f"""
                        <div class="suggested-question">
                            <strong>💡 {idx + 1}.</strong> {question}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        if st.button("Ask", key=f"ask_suggestion_{idx}"):
                            # Auto-submit the question
                            with st.spinner("🤔 Analyzing..."):
                                try:
                                    response = requests.post(
                                        f"{API_URL}/query",
                                        json={
                                            'dataset_id': st.session_state.dataset_id,
                                            'question': question
                                        }
                                    )
                                    
                                    if response.status_code == 200:
                                        result = response.json()
                                        answer = result['result']['answer']
                                        
                                        st.session_state.chat_history.append({
                                            'query': question,
                                            'response': answer
                                        })
                                        
                                        st.rerun()
                                
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
                
                st.markdown("---")
            
            # Manual refresh suggestions
            if st.button("🔄 Refresh Suggestions"):
                with st.spinner("Generating new suggestions..."):
                    try:
                        response = requests.get(f"{API_URL}/suggest/{st.session_state.dataset_id}")
                        if response.status_code == 200:
                            st.session_state.suggested_questions = response.json()['suggestions']
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
            
            # Example queries
            with st.expander("💡 More Example Questions"):
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
            
            st.markdown("---")
            
            # Chat history display
            st.markdown("### 💬 Conversation History")
            
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
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # ====================================================================
        # TAB 3: DATA EXPLORER
        # ====================================================================
        with tab3:
            st.markdown('<div class="content-container">', unsafe_allow_html=True)
            
            st.header("📊 Data Explorer")
            
            # Interactive filters
            st.markdown("### 🔍 Filter Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.session_state.metadata['categorical_columns']:
                    filter_col = st.selectbox(
                        "Filter by column",
                        ['None'] + st.session_state.metadata['categorical_columns']
                    )
            
            with col2:
                if filter_col and filter_col != 'None':
                    # Get unique values for filtering
                    st.info("Filter options will appear here")
            
            st.markdown("---")
            
            # Show sample data
            st.subheader("📋 Sample Data (First 10 Rows)")
            if st.session_state.get('sample_data'):
                df_sample = pd.DataFrame(st.session_state.sample_data)
                st.dataframe(df_sample, use_container_width=True, height=400)
            
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
                        'Missing Values': missing,
                        'Missing %': f"{(missing / st.session_state.metadata['rows'] * 100):.1f}%"
                    })
                
                st.dataframe(pd.DataFrame(col_info), use_container_width=True)
            
            st.markdown("---")
            
            # Custom chart builder
            st.subheader("🎨 Custom Chart Builder")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                chart_type = st.selectbox(
                    "Chart Type",
                    options=['bar', 'line', 'scatter', 'pie', 'histogram', 'box', 'heatmap']
                )
            
            with col2:
                all_columns = list(st.session_state.metadata['column_types'].keys())
                x_column = st.selectbox("X Column", options=all_columns)
            
            with col3:
                if chart_type not in ['pie', 'histogram', 'heatmap']:
                    numeric_cols = st.session_state.metadata['numeric_columns']
                    y_column = st.selectbox("Y Column", options=numeric_cols if numeric_cols else all_columns)
                else:
                    y_column = None
            
            if st.button("🎨 Generate Custom Chart", use_container_width=True):
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
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # ====================================================================
        # TAB 4: COLUMN ANALYSIS (NEW)
        # ====================================================================
        with tab4:
            st.markdown('<div class="content-container">', unsafe_allow_html=True)
            
            st.header("🔍 Deep Column Analysis")
            st.markdown("*Analyze individual columns in detail*")
            
            # Column selector
            selected_column = st.selectbox(
                "Select a column to analyze",
                options=list(st.session_state.metadata['column_types'].keys())
            )
            
            if st.button("🔍 Analyze Column", use_container_width=True):
                with st.spinner(f"Analyzing {selected_column}..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/analyze/column",
                            json={
                                'dataset_id': st.session_state.dataset_id,
                                'column_name': selected_column
                            }
                        )
                        
                        if response.status_code == 200:
                            analysis = response.json()['analysis']
                            
                            st.success(f"✅ Analysis complete for: **{selected_column}**")
                            
                            # Basic info
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Data Type", analysis['data_type'])
                            
                            with col2:
                                st.metric("Total Values", f"{analysis['total_values']:,}")
                            
                            with col3:
                                st.metric("Missing Values", analysis['missing_values'])
                            
                            st.metric("Unique Values", f"{analysis['unique_values']:,}")
                            
                            st.markdown("---")
                            
                            # Statistical analysis for numeric columns
                            if 'mean' in analysis:
                                st.subheader("📊 Statistical Summary")
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Mean", f"{analysis['mean']:.2f}")
                                    st.metric("Min", f"{analysis['min']:.2f}")
                                
                                with col2:
                                    st.metric("Median", f"{analysis['median']:.2f}")
                                    st.metric("Max", f"{analysis['max']:.2f}")
                                
                                with col3:
                                    st.metric("Std Dev", f"{analysis['std']:.2f}")
                                
                                # Quartiles
                                st.markdown("#### Quartiles")
                                quartile_col1, quartile_col2, quartile_col3 = st.columns(3)
                                
                                with quartile_col1:
                                    st.metric("Q1 (25%)", f"{analysis['quartiles']['Q1']:.2f}")
                                
                                with quartile_col2:
                                    st.metric("Q2 (50%)", f"{analysis['quartiles']['Q2']:.2f}")
                                
                                with quartile_col3:
                                    st.metric("Q3 (75%)", f"{analysis['quartiles']['Q3']:.2f}")
                            
                            # Top values for categorical columns
                            elif 'top_values' in analysis:
                                st.subheader("🏆 Top Values")
                                
                                top_values_df = pd.DataFrame([
                                    {'Value': k, 'Count': v}
                                    for k, v in analysis['top_values'].items()
                                ])
                                
                                st.dataframe(top_values_df, use_container_width=True)
                        
                        else:
                            st.error("Could not analyze column")
                    
                    except Exception as e:
                        st.error(f"Error: {e}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # ====================================================================
        # CHAT INPUT (Outside tabs - Important for Streamlit)
        # ====================================================================
        st.markdown("---")
        st.markdown("### 💬 Ask a Question")
        user_query = st.chat_input("Type your question here...")
        
        if user_query:
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
                        answer = result['result']['answer']
                        
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
        # ====================================================================
        # WELCOME SCREEN (Enhanced)
        # ====================================================================
        st.markdown('<div class="content-container">', unsafe_allow_html=True)
        
        st.markdown("## 👋 Welcome to AI Data Analytics Platform!")
        
        st.markdown("""
        ### 🚀 Get Started in 3 Steps
        
        1. **📁 Upload Your Data** - Support for CSV, Excel, JSON, and more
        2. **💡 Get AI Insights** - Automatic analysis and suggestions
        3. **💬 Ask Questions** - Natural language queries powered by AI
        
        ---
        
        ### ✨ New Features
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            #### 🎯 Smart Features
            - ✨ Auto-suggested questions
            - 🎪 Data quality scoring
            - 🔍 Column-specific analysis
            - 🎲 Sample datasets
            """)
        
        with col2:
            st.markdown("""
            #### 🎨 Rich UI
            - 🌈 Animated gradients
            - 🎬 Smooth transitions
            - 💫 Interactive cards
            - 📊 Beautiful charts
            """)
        
        with col3:
            st.markdown("""
            #### 💾 Export Options
            - 📄 PDF Reports
            - 📊 Excel Files
            - 📝 CSV Export
            - 🔧 JSON Format
            """)
        
        st.markdown("---")
        
        st.info("👈 **Start by uploading a file or selecting a sample dataset from the sidebar!**")
        
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# RUN APP
# ============================================================================
if __name__ == "__main__":
    main()

