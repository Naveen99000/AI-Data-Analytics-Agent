"""
AI Data Analytics Platform - Complete Fixed Frontend
Chart Builder & Insights Fixed + AI Analysis First
"""

import streamlit as st
import requests
import pandas as pd
import base64
from datetime import datetime

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="AI Data Analytics Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONFIGURATION
# ============================================================================
API_URL = "http://localhost:8000/api"

# ============================================================================
# SESSION STATE
# ============================================================================
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'
if 'dataset_id' not in st.session_state:
    st.session_state.dataset_id = None
if 'metadata' not in st.session_state:
    st.session_state.metadata = None
if 'quality_score' not in st.session_state:
    st.session_state.quality_score = None
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'suggestions' not in st.session_state:
    st.session_state.suggestions = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'sample_data' not in st.session_state:
    st.session_state.sample_data = None
if 'last_uploaded' not in st.session_state:
    st.session_state.last_uploaded = None
if 'auto_charts' not in st.session_state:
    st.session_state.auto_charts = []

# ============================================================================
# THEME CSS
# ============================================================================
def get_theme_css(theme='dark'):
    if theme == 'dark':
        return """
        <style>
            .stApp { background: linear-gradient(135deg, #1A1A1A 0%, #2D1B2E 100%); }
            .main-header { color: #CE1141; text-shadow: 2px 2px 4px rgba(0,0,0,0.5); }
            h1, h2, h3 { color: #CE1141; }
            .stMetric { background: rgba(206, 17, 65, 0.1); border-radius: 10px; padding: 15px; }
            [data-testid="stMetricValue"] { color: #CE1141; font-size: 2rem; font-weight: 700; }
            .stButton button {
                background: linear-gradient(135deg, #CE1141 0%, #BF4B96 100%);
                color: white; border: none; border-radius: 8px; padding: 10px 20px;
                font-weight: 600; transition: all 0.3s;
            }
            .stButton button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(206, 17, 65, 0.4);
            }
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #2D1B2E 0%, #1A1A1A 100%);
            }
            .stTabs [data-baseweb="tab"] {
                background: rgba(206, 17, 65, 0.1);
                border-radius: 8px 8px 0 0;
                font-weight: 600;
            }
            .stTabs [data-baseweb="tab"][aria-selected="true"] {
                background: #CE1141;
                color: white;
            }
        </style>
        """
    else:
        return """
        <style>
            .stApp { background: linear-gradient(135deg, #F8F9FA 0%, #FFFFFF 100%); }
            h1, h2, h3 { color: #2C3E50; }
            .stMetric { background: #F8F9FA; border-radius: 10px; padding: 15px; }
            [data-testid="stMetricValue"] { color: #CE1141; font-size: 2rem; font-weight: 700; }
            .stButton button {
                background: linear-gradient(135deg, #CE1141 0%, #BF4B96 100%);
                color: white; border: none; border-radius: 8px;
                padding: 10px 20px; font-weight: 600;
            }
            [data-testid="stSidebar"] { background: #F8F9FA; }
        </style>
        """

st.markdown(get_theme_css(st.session_state.theme), unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def check_backend():
    try:
        response = requests.get(API_URL.replace('/api', ''))
        return response.status_code == 200
    except:
        return False

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.markdown("""
    <h1 class="main-header" style="text-align: center;">
        🎯 AI Data Analytics Platform
    </h1>
    <p style="text-align: center; color: #BF4B96; font-size: 1.2rem;">
        💎 Smart Edition | 🤖 AI with Auto Charts | 🚀 Question-Based Visualization
    </p>
    """, unsafe_allow_html=True)
    
    if not check_backend():
        st.error("❌ Backend not running. Start: `python backend.py`")
        return
    
    st.success("✅ Connected")
    
    # ========================================================================
    # SIDEBAR
    # ========================================================================
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # Theme Toggle
        st.markdown("### 🎨 Theme")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            if st.button("🌙 Dark", use_container_width=True):
                st.session_state.theme = 'dark'
                st.rerun()
        with col_t2:
            if st.button("☀️ Light", use_container_width=True):
                st.session_state.theme = 'light'
                st.rerun()
        
        st.caption(f"**{st.session_state.theme.title()} Mode**")
        st.divider()
        
        # Upload
        st.markdown("### 📁 Data Source")
        
        data_source = st.radio(
            "Choose:",
            ["📤 Upload", "🎲 Sample"],
            label_visibility="collapsed"
        )
        
        if data_source == "📤 Upload":
            uploaded_file = st.file_uploader(
                "Upload file",
                type=['csv', 'xlsx', 'json'],
                label_visibility="collapsed"
            )
            
            if uploaded_file and st.session_state.last_uploaded != uploaded_file.name:
                with st.spinner("🔄 Processing..."):
                    try:
                        files = {'file': (uploaded_file.name, uploaded_file.getvalue())}
                        response = requests.post(f"{API_URL}/upload", files=files)
                        
                        if response.status_code == 200:
                            data = response.json()
                            
                            st.session_state.dataset_id = data['dataset_id']
                            st.session_state.metadata = data['metadata']
                            st.session_state.quality_score = data['quality_score']
                            st.session_state.summary = data['summary']
                            st.session_state.suggestions = data['suggestions']
                            st.session_state.sample_data = pd.DataFrame(data['sample'])
                            st.session_state.auto_charts = data['auto_charts']
                            st.session_state.last_uploaded = uploaded_file.name
                            
                            st.success(f"✅ {data['filename']}")
                            st.toast("🎉 Data loaded!", icon="📊")
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        else:  # Sample
            try:
                response = requests.get(f"{API_URL}/samples")
                if response.status_code == 200:
                    samples = response.json()['samples']
                    for sample in samples:
                        if st.button(f"📊 {sample['name']}", key=sample['id'], use_container_width=True):
                            with st.spinner("Loading..."):
                                load_response = requests.post(f"{API_URL}/samples/{sample['id']}")
                                if load_response.status_code == 200:
                                    data = load_response.json()
                                    
                                    st.session_state.dataset_id = data['dataset_id']
                                    st.session_state.metadata = data['metadata']
                                    st.session_state.quality_score = data['quality_score']
                                    st.session_state.summary = data['summary']
                                    st.session_state.suggestions = data['suggestions']
                                    st.session_state.sample_data = pd.DataFrame(data['sample'])
                                    st.session_state.auto_charts = data['auto_charts']
                                    st.session_state.last_uploaded = None
                                    
                                    st.success("✅ Sample loaded!")
                                    st.rerun()
            except:
                st.error("Could not load samples")
        
        # Export
        if st.session_state.dataset_id:
            st.divider()
            st.markdown("### 📥 Export")
            
            if st.button("📄 PDF Report", use_container_width=True):
                try:
                    response = requests.get(f"{API_URL}/export/pdf/{st.session_state.dataset_id}")
                    if response.status_code == 200:
                        st.download_button(
                            "💾 Download PDF",
                            response.content,
                            f"report_{datetime.now().strftime('%Y%m%d')}.pdf",
                            "application/pdf",
                            use_container_width=True
                        )
                except Exception as e:
                    st.error(f"Error: {e}")
            
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                if st.button("📊 Excel", use_container_width=True):
                    try:
                        response = requests.get(f"{API_URL}/export/excel/{st.session_state.dataset_id}")
                        if response.status_code == 200:
                            st.download_button(
                                "💾 Excel",
                                response.content,
                                f"data_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                use_container_width=True
                            )
                    except:
                        pass
            
            with col_e2:
                if st.button("📝 CSV", use_container_width=True):
                    try:
                        response = requests.get(f"{API_URL}/export/csv/{st.session_state.dataset_id}")
                        if response.status_code == 200:
                            st.download_button(
                                "💾 CSV",
                                response.content,
                                f"data_{datetime.now().strftime('%Y%m%d')}.csv",
                                use_container_width=True
                            )
                    except:
                        pass
            
            st.divider()
            
            if st.button("🗑️ Clear Data", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        st.divider()
        
        with st.expander("ℹ️ About"):
            st.markdown(f"""
            **Version:** 3.0 Smart Edition  
            **Theme:** {st.session_state.theme.title()}  
            
            **✨ Smart Features:**
            - 🤖 AI auto-generates charts based on questions
            - 📊 Top/Most → Bar chart
            - 📈 Trend/Over time → Line chart
            - 📉 Distribution → Histogram
            - 🎨 Custom chart builder
            """)
    
    # ========================================================================
    # MAIN CONTENT
    # ========================================================================
    
    if st.session_state.dataset_id:
        # Quick Stats
        st.markdown("### 📊 Dataset Overview")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("📊 Rows", f"{st.session_state.metadata['rows']:,}")
        with col2:
            st.metric("📋 Columns", st.session_state.metadata['columns'])
        with col3:
            st.metric("🔢 Numeric", len(st.session_state.metadata['numeric_columns']))
        with col4:
            st.metric("📝 Text", len(st.session_state.metadata['categorical_columns']))
        with col5:
            st.metric("🏆 Quality", f"{st.session_state.quality_score['overall']}%")
        
        st.divider()
        
        # Summary
        if st.session_state.summary:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 20px; border-radius: 12px; color: white; margin: 20px 0;">
                <h3 style="color: white;">📊 Executive Summary</h3>
                <p style="font-size: 1rem; line-height: 1.6;">{st.session_state.summary}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Tabs - AI ANALYSIS FIRST!
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔍 AI Analysis",
            "📊 Auto Charts",
            "🎨 Chart Builder",
            "📈 Data View",
            "🎯 Insights"
        ])
        
        # ====================================================================
        # TAB 1: AI ANALYSIS (FIRST TAB!)
        # ====================================================================
        with tab1:
            st.markdown("### 💬 Ask Questions - Get Answers + Charts")
            st.caption("AI automatically generates suitable visualizations based on your questions")
            
            # Suggested Questions
            if st.session_state.suggestions:
                with st.expander("💡 Suggested Questions", expanded=True):
                    cols = st.columns(2)
                    for idx, suggestion in enumerate(st.session_state.suggestions):
                        with cols[idx % 2]:
                            if st.button(f"💡 {suggestion}", key=f"sug_{idx}", use_container_width=True):
                                with st.spinner("🤖 Analyzing and generating chart..."):
                                    try:
                                        response = requests.post(
                                            f"{API_URL}/query",
                                            json={
                                                'dataset_id': st.session_state.dataset_id,
                                                'question': suggestion
                                            }
                                        )
                                        
                                        if response.status_code == 200:
                                            result = response.json()
                                            
                                            st.session_state.chat_history.append({
                                                'question': suggestion,
                                                'answer': result['answer'],
                                                'chart_data': result.get('chart_data')
                                            })
                                            
                                            st.toast("✅ Analysis complete with chart!", icon="📊")
                                            st.rerun()
                                    except Exception as e:
                                        st.error(f"Error: {e}")
            
            st.divider()
            
            # Chat History with Charts
            if st.session_state.chat_history:
                st.markdown("### 💭 Analysis History")
                st.caption("Each answer includes a relevant visualization")
                
                for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
                    with st.expander(f"❓ {chat['question']}", expanded=(i==0)):
                        st.markdown("**Answer:**")
                        st.write(chat['answer'])
                        
                        if chat.get('chart_data'):
                            st.markdown("**📊 Visualization:**")
                            chart_bytes = base64.b64decode(chat['chart_data'])
                            st.image(chart_bytes)
                        else:
                            st.info("No chart generated for this question")
                
                if st.button("🗑️ Clear History", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()
            
            else:
                st.info("💡 Ask a question to see AI-generated answers with charts!")
        
        # ====================================================================
        # TAB 2: AUTO CHARTS
        # ====================================================================
        with tab2:
            st.markdown("### 📊 Automatically Generated Visualizations")
            st.caption("Charts generated on data upload")
            
            if st.session_state.auto_charts:
                cols = st.columns(2)
                for idx, chart in enumerate(st.session_state.auto_charts):
                    with cols[idx % 2]:
                        st.markdown(f"**{chart['title']}**")
                        chart_bytes = base64.b64decode(chart['data'])
                        st.image(chart_bytes, use_column_width=True)
                        st.caption(f"Type: {chart['type'].title()}")
            else:
                st.info("No auto charts available")
        
        # ====================================================================
        # TAB 3: CUSTOM CHART BUILDER (FIXED!)
        # ====================================================================
        with tab3:
            st.markdown("### 🎨 Build Custom Charts")
            st.caption("Create your own visualizations with custom column selections")
            
            col_c1, col_c2, col_c3 = st.columns(3)
            
            with col_c1:
                chart_type = st.selectbox(
                    "Chart Type",
                    options=['bar', 'line', 'scatter', 'histogram'],
                    help="Select the type of chart to generate"
                )
            
            with col_c2:
                all_columns = list(st.session_state.metadata['column_types'].keys())
                x_column = st.selectbox(
                    "X Column", 
                    options=all_columns,
                    help="Select column for X-axis"
                )
            
            with col_c3:
                # Show Y column for all chart types except histogram
                if chart_type != 'histogram':
                    numeric_cols = st.session_state.metadata['numeric_columns']
                    y_options = numeric_cols if numeric_cols else all_columns
                    
                    if y_options:
                        y_column = st.selectbox(
                            "Y Column", 
                            options=y_options,
                            help="Select column for Y-axis (numeric recommended)"
                        )
                    else:
                        st.warning("No numeric columns available")
                        y_column = None
                else:
                    y_column = None
                    st.info("Histogram only needs X column")
            
            # Chart description
            st.markdown("---")
            if chart_type == 'bar':
                st.caption("📊 Bar chart will group by X column and aggregate Y column")
            elif chart_type == 'line':
                st.caption("📈 Line chart shows trend of Y over X")
            elif chart_type == 'scatter':
                st.caption("📉 Scatter plot shows relationship between X and Y")
            elif chart_type == 'histogram':
                st.caption("📊 Histogram shows distribution of X column")
            
            if st.button("🎨 Generate Chart", use_container_width=True, type="primary"):
                # Validate inputs
                if chart_type != 'histogram' and not y_column:
                    st.error("⚠️ Please select a Y column for this chart type")
                else:
                    with st.spinner("Creating chart..."):
                        try:
                            response = requests.post(
                                f"{API_URL}/chart/custom",
                                json={
                                    'dataset_id': st.session_state.dataset_id,
                                    'chart_type': chart_type,
                                    'x_column': x_column,
                                    'y_column': y_column
                                }
                            )
                            
                            if response.status_code == 200:
                                result = response.json()
                                if result['success']:
                                    st.success("✅ Chart created successfully!")
                                    chart_bytes = base64.b64decode(result['chart_data'])
                                    st.image(chart_bytes, use_column_width=True)
                                else:
                                    st.error(f"❌ Chart generation failed: {result.get('error', 'Unknown error')}")
                            else:
                                st.error(f"❌ API error: {response.status_code}")
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
                            st.exception(e)
        
        # ====================================================================
        # TAB 4: DATA VIEW
        # ====================================================================
        with tab4:
            st.markdown("### 🗂️ Dataset Preview")
            
            if st.session_state.sample_data is not None:
                num_rows = st.slider("Rows to display", 5, min(50, len(st.session_state.sample_data)), 10)
                st.dataframe(st.session_state.sample_data.head(num_rows), use_container_width=True, height=400)
            
            st.divider()
            
            st.markdown("### 📋 Column Information")
            col_info = []
            for col, dtype in st.session_state.metadata['column_types'].items():
                missing = st.session_state.metadata['missing_values'].get(col, 0)
                col_info.append({
                    'Column': col,
                    'Type': dtype,
                    'Missing': missing,
                    'Missing %': f"{(missing / st.session_state.metadata['rows'] * 100):.1f}%"
                })
            
            st.dataframe(pd.DataFrame(col_info), use_container_width=True, hide_index=True)
        
        # ====================================================================
        # TAB 5: COLUMN INSIGHTS (FIXED!)
        # ====================================================================
        with tab5:
            st.markdown("### 🎯 Deep Column Analysis")
            st.caption("Get detailed statistics and insights for any column")
            
            col_select1, col_select2 = st.columns([3, 1])
            
            with col_select1:
                selected_column = st.selectbox(
                    "Select column to analyze:",
                    options=list(st.session_state.metadata['column_types'].keys()),
                    help="Choose any column from your dataset"
                )
            
            with col_select2:
                analyze_btn = st.button("🔍 Analyze", use_container_width=True, type="primary")
            
            if analyze_btn:
                with st.spinner(f"Analyzing {selected_column}..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/column-insights",
                            json={
                                'dataset_id': st.session_state.dataset_id,
                                'column_name': selected_column
                            }
                        )
                        
                        if response.status_code == 200:
                            insights = response.json()['insights']
                            
                            st.success(f"✅ Analysis complete for: **{selected_column}**")
                            
                            st.divider()
                            
                            # Basic Info
                            st.markdown("#### 📋 Basic Information")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Data Type", insights['type'])
                            with col2:
                                st.metric("Total Values", f"{insights['total']:,}")
                            with col3:
                                missing_pct = (insights['missing'] / insights['total'] * 100) if insights['total'] > 0 else 0
                                st.metric("Missing", f"{insights['missing']} ({missing_pct:.1f}%)")
                            
                            st.metric("Unique Values", f"{insights['unique']:,}")
                            
                            st.divider()
                            
                            # Statistical Analysis for Numeric Columns
                            if 'mean' in insights:
                                st.markdown("#### 📊 Statistical Summary")
                                
                                col_s1, col_s2, col_s3 = st.columns(3)
                                
                                with col_s1:
                                    st.metric("Mean", f"{insights['mean']:.2f}")
                                    st.metric("Minimum", f"{insights['min']:.2f}")
                                
                                with col_s2:
                                    st.metric("Median", f"{insights['median']:.2f}")
                                    st.metric("Maximum", f"{insights['max']:.2f}")
                                
                                with col_s3:
                                    st.metric("Std Deviation", f"{insights['std']:.2f}")
                                    range_val = insights['max'] - insights['min']
                                    st.metric("Range", f"{range_val:.2f}")
                                
                                # Statistical interpretation
                                st.info(f"""
                                **📈 Interpretation:**
                                - **Range:** {insights['min']:.2f} to {insights['max']:.2f}
                                - **Average:** {insights['mean']:.2f}
                                - **Spread (Std Dev):** {insights['std']:.2f}
                                - **Middle Value:** {insights['median']:.2f}
                                """)
                            
                            # Top Values for Categorical Columns
                            elif 'top_values' in insights:
                                st.markdown("#### 🏆 Top Values Distribution")
                                
                                top_df = pd.DataFrame([
                                    {
                                        'Value': k, 
                                        'Count': v,
                                        'Percentage': f"{(v / insights['total'] * 100):.1f}%"
                                    }
                                    for k, v in insights['top_values'].items()
                                ])
                                
                                st.dataframe(top_df, use_container_width=True, hide_index=True)
                                
                                st.divider()
                                
                                # Visual representation
                                st.markdown("**📊 Visual Distribution:**")
                                for idx, row in top_df.iterrows():
                                    percentage = (row['Count'] / insights['total'] * 100)
                                    st.progress(percentage / 100, text=f"{row['Value']}: {row['Count']} ({percentage:.1f}%)")
                        
                        else:
                            st.error(f"❌ API returned error: {response.status_code}")
                            
                    except Exception as e:
                        st.error(f"❌ Error analyzing column: {e}")
                        st.exception(e)
            
            else:
                st.info("👆 Select a column and click 'Analyze' to see detailed insights")
        
        # ====================================================================
        # CHAT INPUT (Outside tabs - Main feature)
        # ====================================================================
        st.divider()
        st.markdown("### 💬 Ask Your Own Question")
        st.caption("Type any question - AI will provide answer + relevant chart")
        
        user_question = st.chat_input("💬 e.g., 'What are the top 5 products by sales?'")
        
        if user_question:
            with st.spinner("🤖 Analyzing and generating visualization..."):
                try:
                    response = requests.post(
                        f"{API_URL}/query",
                        json={
                            'dataset_id': st.session_state.dataset_id,
                            'question': user_question
                        }
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        st.session_state.chat_history.append({
                            'question': user_question,
                            'answer': result['answer'],
                            'chart_data': result.get('chart_data')
                        })
                        
                        st.toast("✅ Done! Check AI Analysis tab", icon="📊")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
    
    else:
        # Welcome Screen
        st.markdown("""
        <div style="text-align: center; padding: 50px;">
            <h2>👋 Welcome!</h2>
            <p style="font-size: 1.2rem; color: #BF4B96;">
                Upload data to get AI-powered insights with automatic chart generation
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 🤖 Smart AI")
            st.write("Ask questions → Get answers + charts")
        with col2:
            st.markdown("### 📊 Auto Charts")
            st.write("Instant visualizations on upload")
        with col3:
            st.markdown("### 🎨 Custom Builder")
            st.write("Create your own charts")

if __name__ == "__main__":
    main()
