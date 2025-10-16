"""
AI Data Analytics Backend - With Multi-LLM Support + PDF Export
Supports: Groq, OpenAI, Gemini, Grok/xAI
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import uvicorn
import pandas as pd
import numpy as np
import json
import os
import uuid
import re
import plotly.graph_objects as go
import plotly.express as px

# PDF Generation
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io
import tempfile

# Multi-LLM Support
import openai
from groq import Groq
import google.generativeai as genai

load_dotenv()

# ============================================================================
# LLM CONFIGURATION
# ============================================================================
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

# Initialize clients based on provider
groq_client = None
gemini_model = None
openai_client = None

if LLM_PROVIDER == "groq":
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        groq_client = Groq(api_key=groq_api_key)
        MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    else:
        print("⚠️ GROQ_API_KEY not found")

elif LLM_PROVIDER == "gemini":
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
        gemini_model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-pro"))
        MODEL_NAME = "gemini-pro"
    else:
        print("⚠️ GEMINI_API_KEY not found")

elif LLM_PROVIDER == "openai":
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        openai.api_key = openai_api_key
        MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    else:
        print("⚠️ OPENAI_API_KEY not found")

elif LLM_PROVIDER == "xai":  # Grok
    xai_api_key = os.getenv("XAI_API_KEY")
    if xai_api_key:
        openai_client = openai.OpenAI(
            api_key=xai_api_key,
            base_url="https://api.x.ai/v1"
        )
        MODEL_NAME = "grok-beta"
    else:
        print("⚠️ XAI_API_KEY not found")

# ============================================================================
# GLOBAL STORAGE
# ============================================================================
DATASETS = {}

# ============================================================================
# UNIVERSAL LLM CALLER
# ============================================================================
class UniversalLLM:
    """Universal LLM interface supporting multiple providers"""
    
    @staticmethod
    def chat_completion(messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """Universal chat completion across all providers"""
        
        try:
            if LLM_PROVIDER == "groq" and groq_client:
                response = groq_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            
            elif LLM_PROVIDER == "gemini" and gemini_model:
                # Convert messages to Gemini format
                prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
                response = gemini_model.generate_content(prompt)
                return response.text
            
            elif LLM_PROVIDER == "openai":
                response = openai.ChatCompletion.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            
            elif LLM_PROVIDER == "xai" and openai_client:  # Grok
                response = openai_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            
            else:
                return "LLM not configured. Please set up API key in .env file."
        
        except Exception as e:
            return f"Error calling LLM: {str(e)}"

# ============================================================================
# FILE PARSER
# ============================================================================
class FileParser:
    """Parse any data format into DataFrame"""
    
    SUPPORTED_FORMATS = {'.csv', '.xlsx', '.xls', '.json', '.parquet', '.txt', '.tsv'}
    
    @staticmethod
    def parse_file(file_path: str) -> pd.DataFrame:
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext == '.csv':
                return FileParser._parse_csv(file_path)
            elif file_ext in ['.xlsx', '.xls']:
                return pd.read_excel(file_path)
            elif file_ext == '.json':
                return FileParser._parse_json(file_path)
            elif file_ext == '.parquet':
                return pd.read_parquet(file_path)
            elif file_ext == '.tsv':
                return pd.read_csv(file_path, sep='\t')
            elif file_ext == '.txt':
                return FileParser._parse_csv(file_path)
            else:
                raise ValueError(f"Unsupported format: {file_ext}")
        except Exception as e:
            raise Exception(f"Error parsing file: {str(e)}")
    
    @staticmethod
    def _parse_csv(file_path: str) -> pd.DataFrame:
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        separators = [',', ';', '\t', '|']
        
        for encoding in encodings:
            for sep in separators:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, sep=sep)
                    if len(df.columns) > 1:
                        return df
                except:
                    continue
        return pd.read_csv(file_path)
    
    @staticmethod
    def _parse_json(file_path: str) -> pd.DataFrame:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list) and value:
                    return pd.DataFrame(value)
            return pd.DataFrame([data])
        else:
            raise ValueError("Unsupported JSON structure")

# ============================================================================
# DATA PROCESSOR
# ============================================================================
class DataProcessor:
    """Clean, transform, and prepare data for analysis"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_df = df.copy()
        self.metadata = {}
        self.analyze_data()
    
    def analyze_data(self):
        self.metadata = {
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'column_types': {k: str(v) for k, v in self.df.dtypes.to_dict().items()},
            'missing_values': self.df.isnull().sum().to_dict(),
            'numeric_columns': self.df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': self.df.select_dtypes(include=['object']).columns.tolist(),
            'datetime_columns': self.df.select_dtypes(include=['datetime']).columns.tolist(),
        }
    
    def autoclean(self) -> Dict[str, Any]:
        report = {"steps_performed": [], "rows_before": len(self.df)}
        
        # Remove duplicates
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            self.df = self.df.drop_duplicates()
            report['steps_performed'].append(f"Removed {duplicates} duplicate rows")
        
        # Fill missing values
        for col in self.df.columns:
            missing_pct = self.df[col].isnull().sum() / len(self.df)
            if missing_pct > 0 and missing_pct < 0.5:
                if col in self.metadata['numeric_columns']:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                else:
                    mode_val = self.df[col].mode()
                    fill_val = mode_val[0] if len(mode_val) > 0 else "Unknown"
                    self.df[col].fillna(fill_val, inplace=True)
                report['steps_performed'].append(f"Filled missing values in {col}")
        
        # Auto-detect datetime columns (FIXED)
        for col in self.df.select_dtypes(include='object').columns:
            sample = self.df[col].dropna().head(5)
            if len(sample) > 0:
                try:
                    pd.to_datetime(sample, errors='raise', dayfirst=True, format='mixed')
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce', dayfirst=True, format='mixed')
                    report['steps_performed'].append(f"Converted {col} to datetime")
                except:
                    pass
        
        # Standardize column names
        self.df.columns = [self.clean_column_name(col) for col in self.df.columns]
        report['steps_performed'].append("Standardized column names")
        
        # Remove completely empty columns
        null_cols = self.df.columns[self.df.isnull().all()].tolist()
        if null_cols:
            self.df = self.df.drop(columns=null_cols)
            report['steps_performed'].append(f"Removed {len(null_cols)} empty columns")
        
        report["rows_after"] = len(self.df)
        self.analyze_data()
        return report
    
    def clean_column_name(self, col: str) -> str:
        col = re.sub(r'[^\w\s]', '', str(col))
        col = re.sub(r'\s+', '_', col)
        return col.lower().strip('_')
    
    def get_summary_stats(self) -> Dict[str, Any]:
        stats_df = self.df.describe(include='all')
        return {
            'basic_stats': stats_df.to_dict(),
            'metadata': self.metadata,
            'sample_data': self.df.head(10).to_dict('records')
        }

# ============================================================================
# INSIGHT GENERATOR
# ============================================================================
class InsightGenerator:
    """Generate automated insights from data"""
    
    def __init__(self, df: pd.DataFrame, metadata: Dict):
        self.df = df
        self.metadata = metadata
    
    def generate_insights(self) -> List[str]:
        insights = []
        
        # Data size insight
        insights.append(f"📊 Dataset contains {self.metadata['rows']:,} rows and {self.metadata['columns']} columns")
        
        # Missing data insights
        missing = {k: v for k, v in self.metadata['missing_values'].items() if v > 0}
        if missing:
            worst_col = max(missing, key=missing.get)
            insights.append(f"⚠️ {worst_col} has the most missing values ({missing[worst_col]:,} missing)")
        
        # Numeric insights
        for col in self.metadata['numeric_columns'][:3]:
            if col in self.df.columns:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                insights.append(f"📈 {col}: mean={mean_val:.2f}, std={std_val:.2f}")
        
        # Categorical insights
        for col in self.metadata['categorical_columns'][:3]:
            if col in self.df.columns:
                unique_count = self.df[col].nunique()
                most_common = self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else "N/A"
                insights.append(f"🏷️ {col}: {unique_count} unique values, most common = '{most_common}'")
        
        return insights

# ============================================================================
# AI DATA ANALYST
# ============================================================================
class AIDataAnalyst:
    """AI-powered data analysis using LLMs"""
    
    def __init__(self, df: pd.DataFrame, metadata: Dict):
        self.df = df
        self.metadata = metadata
    
    def analyze_with_ai(self, question: str) -> Dict[str, Any]:
        """AI-powered analysis"""
        
        # Prepare data context (OPTIMIZED - Less data)
        data_context = self._prepare_context()
        
        # Build prompt
        prompt = f"""You are a data analyst. Analyze this dataset and answer the question.

Dataset Context:
{data_context}

Sample Data (first 3 rows):
{self.df.head(3).to_string()}

Question: {question}

Provide a clear, concise answer with specific insights."""
        
        messages = [
            {"role": "system", "content": "You are an expert data analyst. Be concise."},
            {"role": "user", "content": prompt}
        ]
        
        # Get AI response (REDUCED max_tokens)
        ai_response = UniversalLLM.chat_completion(messages, temperature=0.7, max_tokens=800)
        
        return {
            "question": question,
            "answer": ai_response,
            "data_context": self.metadata
        }
    
    def _prepare_context(self) -> str:
        """Prepare minimal context to save tokens"""
        ctx = []
        ctx.append(f"Rows: {self.metadata['rows']:,}")
        ctx.append(f"Columns: {self.metadata['columns']}")
        
        # Limit to first 3 columns only
        ctx.append(f"Numeric: {', '.join(self.metadata['numeric_columns'][:3])}")
        ctx.append(f"Categorical: {', '.join(self.metadata['categorical_columns'][:3])}")
        
        return "\n".join(ctx)
    
    def suggest_analysis(self) -> List[str]:
        """AI suggests analysis ideas"""
        
        # Get only first 5 columns to reduce tokens
        cols_sample = list(self.df.columns[:5])
        
        prompt = f"""Based on this dataset, suggest 5 analysis questions.

Columns: {', '.join(cols_sample)}

Suggest 5 specific questions (be brief)."""
        
        messages = [
            {"role": "system", "content": "You are a data consultant. Be concise."},
            {"role": "user", "content": prompt}
        ]
        
        response = UniversalLLM.chat_completion(messages, temperature=0.8, max_tokens=300)
        
        # Parse response into list
        suggestions = [line.strip() for line in response.split('\n') if line.strip() and not line.strip().startswith('#')]
        return suggestions[:5]

# ============================================================================
# CHART GENERATOR
# ============================================================================
class ChartGenerator:
    """Generate interactive Plotly charts"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def create_chart(self, chart_type: str, x_col: str, y_col: Optional[str] = None, 
                     color_col: Optional[str] = None) -> Dict[str, Any]:
        """Create chart based on type and columns"""
        
        try:
            if chart_type == "bar":
                fig = px.bar(self.df, x=x_col, y=y_col, color=color_col, title=f"{y_col} by {x_col}")
            
            elif chart_type == "line":
                fig = px.line(self.df, x=x_col, y=y_col, color=color_col, title=f"{y_col} over {x_col}")
            
            elif chart_type == "scatter":
                fig = px.scatter(self.df, x=x_col, y=y_col, color=color_col, title=f"{y_col} vs {x_col}")
            
            elif chart_type == "histogram":
                fig = px.histogram(self.df, x=x_col, color=color_col, title=f"Distribution of {x_col}")
            
            elif chart_type == "box":
                fig = px.box(self.df, x=x_col, y=y_col, color=color_col, title=f"{y_col} distribution by {x_col}")
            
            elif chart_type == "pie":
                value_counts = self.df[x_col].value_counts()
                fig = px.pie(values=value_counts.values, names=value_counts.index, title=f"Distribution of {x_col}")
            
            elif chart_type == "heatmap":
                numeric_df = self.df.select_dtypes(include=[np.number])
                corr_matrix = numeric_df.corr()
                fig = px.imshow(corr_matrix, text_auto=True, title="Correlation Heatmap")
            
            else:
                raise ValueError(f"Unsupported chart type: {chart_type}")
            
            return {
                "success": True,
                "chart_json": fig.to_json(),
                "chart_html": fig.to_html()
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def auto_suggest_charts(self, metadata: Dict) -> List[Dict]:
        """Suggest appropriate chart types based on data"""
        suggestions = []
        
        numeric_cols = metadata['numeric_columns']
        categorical_cols = metadata['categorical_columns']
        
        # Bar charts for categorical vs numeric
        if categorical_cols and numeric_cols:
            suggestions.append({
                "type": "bar",
                "x": categorical_cols[0],
                "y": numeric_cols[0],
                "description": f"Bar chart of {numeric_cols[0]} by {categorical_cols[0]}"
            })
        
        # Line chart for time series
        datetime_cols = metadata['datetime_columns']
        if datetime_cols and numeric_cols:
            suggestions.append({
                "type": "line",
                "x": datetime_cols[0],
                "y": numeric_cols[0],
                "description": f"Time series of {numeric_cols[0]}"
            })
        
        # Scatter plot for numeric vs numeric
        if len(numeric_cols) >= 2:
            suggestions.append({
                "type": "scatter",
                "x": numeric_cols[0],
                "y": numeric_cols[1],
                "description": f"Scatter plot of {numeric_cols[1]} vs {numeric_cols[0]}"
            })
        
        # Histogram for numeric distribution
        if numeric_cols:
            suggestions.append({
                "type": "histogram",
                "x": numeric_cols[0],
                "description": f"Distribution of {numeric_cols[0]}"
            })
        
        # Pie chart for categorical
        if categorical_cols:
            suggestions.append({
                "type": "pie",
                "x": categorical_cols[0],
                "description": f"Pie chart of {categorical_cols[0]}"
            })
        
        return suggestions

# ============================================================================
# PDF REPORT GENERATOR
# ============================================================================
class PDFReportGenerator:
    """Generate professional PDF reports"""
    
    def __init__(self, dataset_id: str):
        self.dataset_id = dataset_id
        self.dataset = DATASETS.get(dataset_id)
        if not self.dataset:
            raise ValueError("Dataset not found")
    
    def generate_report(self) -> str:
        """Generate comprehensive PDF report"""
        
        # Create temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        pdf_path = temp_file.name
        
        # Create PDF document
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2ca02c'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        # Title
        story.append(Paragraph("Data Analysis Report", title_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Dataset Overview
        story.append(Paragraph("Dataset Overview", heading_style))
        overview_data = [
            ["Metric", "Value"],
            ["Total Rows", f"{self.dataset['metadata']['rows']:,}"],
            ["Total Columns", str(self.dataset['metadata']['columns'])],
            ["Numeric Columns", str(len(self.dataset['metadata']['numeric_columns']))],
            ["Categorical Columns", str(len(self.dataset['metadata']['categorical_columns']))],
        ]
        
        overview_table = Table(overview_data, colWidths=[3*inch, 3*inch])
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(overview_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Insights
        if 'insights' in self.dataset and self.dataset['insights']:
            story.append(Paragraph("Key Insights", heading_style))
            for insight in self.dataset['insights']:
                story.append(Paragraph(f"• {insight}", styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
        
        # Sample Data
        story.append(Paragraph("Sample Data", heading_style))
        df = self.dataset['processor'].df
        sample_data = df.head(5).values.tolist()
        sample_data.insert(0, df.columns.tolist())
        
        sample_table = Table(sample_data, colWidths=[6.5*inch/len(df.columns)]*len(df.columns))
        sample_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ca02c')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(sample_table)
        
        # Build PDF
        doc.build(story)
        
        return pdf_path

# ============================================================================
# API MODELS
# ============================================================================
class QueryRequest(BaseModel):
    dataset_id: str
    question: str

class ChartRequest(BaseModel):
    dataset_id: str
    chart_type: str
    x_column: str
    y_column: Optional[str] = None
    color_column: Optional[str] = None

# ============================================================================
# FASTAPI APP
# ============================================================================
app = FastAPI(title="AI Data Analytics Agent", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
def read_root():
    return {
        "message": "AI Data Analytics Agent API",
        "version": "2.0",
        "llm_provider": LLM_PROVIDER,
        "model": MODEL_NAME,
        "status": "running"
    }

@app.post("/api/upload/file")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process data file"""
    try:
        # Validate file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in FileParser.SUPPORTED_FORMATS:
            raise HTTPException(400, f"Unsupported file format: {file_ext}")
        
        # Save file temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
        temp_file.write(await file.read())
        temp_file.close()
        
        # Parse file
        df = FileParser.parse_file(temp_file.name)
        os.unlink(temp_file.name)
        
        # Process data
        processor = DataProcessor(df)
        clean_report = processor.autoclean()
        
        # Generate insights
        insight_gen = InsightGenerator(processor.df, processor.metadata)
        insights = insight_gen.generate_insights()
        
        # Store dataset
        dataset_id = str(uuid.uuid4())
        DATASETS[dataset_id] = {
            'filename': file.filename,
            'processor': processor,
            'metadata': processor.metadata,
            'insights': insights,
            'upload_time': datetime.now().isoformat()
        }
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "filename": file.filename,
            "metadata": processor.metadata,
            "clean_report": clean_report,
            "insights": insights,
            "sample_data": processor.df.head(10).to_dict('records')
        }
    
    except Exception as e:
        raise HTTPException(500, f"Error processing file: {str(e)}")

@app.post("/api/query")
async def query_data(request: QueryRequest):
    """Ask AI questions about data"""
    try:
        dataset = DATASETS.get(request.dataset_id)
        if not dataset:
            raise HTTPException(404, "Dataset not found")
        
        analyst = AIDataAnalyst(dataset['processor'].df, dataset['metadata'])
        result = analyst.analyze_with_ai(request.question)
        
        return {
            "success": True,
            "result": result
        }
    
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")

@app.get("/api/suggest/{dataset_id}")
async def suggest_questions(dataset_id: str):
    """AI suggests analysis questions"""
    try:
        dataset = DATASETS.get(dataset_id)
        if not dataset:
            raise HTTPException(404, "Dataset not found")
        
        analyst = AIDataAnalyst(dataset['processor'].df, dataset['metadata'])
        suggestions = analyst.suggest_analysis()
        
        return {
            "success": True,
            "suggestions": suggestions
        }
    
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")

@app.post("/api/chart")
async def create_chart(request: ChartRequest):
    """Generate interactive chart"""
    try:
        dataset = DATASETS.get(request.dataset_id)
        if not dataset:
            raise HTTPException(404, "Dataset not found")
        
        chart_gen = ChartGenerator(dataset['processor'].df)
        result = chart_gen.create_chart(
            request.chart_type,
            request.x_column,
            request.y_column,
            request.color_column
        )
        
        return result
    
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")

@app.get("/api/chart/suggest/{dataset_id}")
async def suggest_charts(dataset_id: str):
    """Suggest appropriate charts"""
    try:
        dataset = DATASETS.get(dataset_id)
        if not dataset:
            raise HTTPException(404, "Dataset not found")
        
        chart_gen = ChartGenerator(dataset['processor'].df)
        suggestions = chart_gen.auto_suggest_charts(dataset['metadata'])
        
        return {
            "success": True,
            "suggestions": suggestions
        }
    
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")

@app.get("/api/export/pdf/{dataset_id}")
async def export_pdf(dataset_id: str):
    """Export analysis report as PDF"""
    try:
        pdf_gen = PDFReportGenerator(dataset_id)
        pdf_path = pdf_gen.generate_report()
        
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=f"report_{dataset_id}.pdf"
        )
    
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")

@app.get("/api/datasets")
async def list_datasets():
    """List all uploaded datasets"""
    return {
        "datasets": [
            {
                "id": k,
                "filename": v['filename'],
                "rows": v['metadata']['rows'],
                "columns": v['metadata']['columns'],
                "upload_time": v['upload_time']
            }
            for k, v in DATASETS.items()
        ]
    }

# ============================================================================
# RUN SERVER
# ============================================================================
if __name__ == "__main__":
    print(f"\n🚀 Starting AI Data Analytics Agent")
    print(f"📊 LLM Provider: {LLM_PROVIDER.upper()}")
    print(f"🤖 Model: {MODEL_NAME}")
    print(f"🌐 Server: http://localhost:8000")
    print(f"📖 Docs: http://localhost:8000/docs\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
