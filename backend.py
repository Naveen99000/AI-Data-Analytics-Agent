"""
AI Data Analytics Backend - Enhanced Version
Features: Multiple exports, data quality score, column analysis, sample datasets
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
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
import io
import plotly.graph_objects as go
import plotly.express as px

# PDF Generation
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
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

elif LLM_PROVIDER == "xai":
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
# GLOBAL STORAGE & SAMPLE DATASETS
# ============================================================================
DATASETS = {}

SAMPLE_DATASETS = {
    "sales": {
        "name": "E-commerce Sales Data",
        "description": "Sample sales dataset with orders, customers, and products",
        "data": pd.DataFrame({
            'order_id': range(1, 101),
            'customer_name': [f'Customer_{i}' for i in range(1, 101)],
            'product': np.random.choice(['Laptop', 'Phone', 'Tablet', 'Monitor'], 100),
            'quantity': np.random.randint(1, 5, 100),
            'price': np.random.uniform(100, 2000, 100).round(2),
            'date': pd.date_range('2024-01-01', periods=100, freq='D')
        })
    },
    "hr": {
        "name": "HR Employee Data",
        "description": "Sample HR dataset with employee information",
        "data": pd.DataFrame({
            'employee_id': range(1, 51),
            'name': [f'Employee_{i}' for i in range(1, 51)],
            'department': np.random.choice(['Sales', 'IT', 'HR', 'Finance'], 50),
            'salary': np.random.randint(40000, 120000, 50),
            'years_experience': np.random.randint(1, 20, 50),
            'rating': np.random.uniform(3.0, 5.0, 50).round(1)
        })
    },
    "finance": {
        "name": "Financial Transactions",
        "description": "Sample financial transaction data",
        "data": pd.DataFrame({
            'transaction_id': range(1, 201),
            'account': [f'ACC_{i:04d}' for i in range(1, 201)],
            'type': np.random.choice(['Debit', 'Credit'], 200),
            'amount': np.random.uniform(10, 5000, 200).round(2),
            'category': np.random.choice(['Food', 'Transport', 'Shopping', 'Bills'], 200),
            'date': pd.date_range('2024-01-01', periods=200, freq='H')
        })
    }
}

# ============================================================================
# UNIVERSAL LLM CALLER
# ============================================================================
class UniversalLLM:
    @staticmethod
    def chat_completion(messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 1000) -> str:
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
            
            elif LLM_PROVIDER == "xai" and openai_client:
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
# DATA PROCESSOR (Enhanced)
# ============================================================================
class DataProcessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_df = df.copy()
        self.metadata = {}
        self.quality_score = {}
        self.analyze_data()
        self.calculate_quality_score()
    
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
    
    def calculate_quality_score(self) -> Dict[str, Any]:
        """Calculate data quality score"""
        scores = []
        
        # 1. Completeness (0-100)
        total_cells = self.df.shape[0] * self.df.shape[1]
        missing_cells = self.df.isnull().sum().sum()
        completeness = ((total_cells - missing_cells) / total_cells) * 100 if total_cells > 0 else 0
        scores.append(completeness)
        
        # 2. Uniqueness (0-100) - Check for duplicates
        uniqueness = ((len(self.df) - self.df.duplicated().sum()) / len(self.df)) * 100 if len(self.df) > 0 else 0
        scores.append(uniqueness)
        
        # 3. Consistency (0-100) - Check data types consistency
        consistency = 100  # Default
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                # Check if numeric values are stored as strings
                try:
                    pd.to_numeric(self.df[col], errors='raise')
                    consistency -= 10  # Penalty for type inconsistency
                except:
                    pass
        consistency = max(0, consistency)
        scores.append(consistency)
        
        # 4. Validity (0-100) - Check for outliers in numeric columns
        validity = 100
        for col in self.metadata['numeric_columns']:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))).sum()
            if outliers > len(self.df) * 0.05:  # More than 5% outliers
                validity -= 5
        validity = max(0, validity)
        scores.append(validity)
        
        # Overall score
        overall_score = sum(scores) / len(scores)
        
        self.quality_score = {
            'overall': round(overall_score, 2),
            'completeness': round(completeness, 2),
            'uniqueness': round(uniqueness, 2),
            'consistency': round(consistency, 2),
            'validity': round(validity, 2),
            'grade': self._get_grade(overall_score)
        }
        
        return self.quality_score
    
    def _get_grade(self, score: float) -> str:
        if score >= 90:
            return 'A+'
        elif score >= 80:
            return 'A'
        elif score >= 70:
            return 'B'
        elif score >= 60:
            return 'C'
        else:
            return 'D'
    
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
        
        # Auto-detect datetime columns
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
        self.calculate_quality_score()
        return report
    
    def clean_column_name(self, col: str) -> str:
        col = re.sub(r'[^\w\s]', '', str(col))
        col = re.sub(r'\s+', '_', col)
        return col.lower().strip('_')
    
    def analyze_column(self, column_name: str) -> Dict[str, Any]:
        """Deep analysis of specific column"""
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' not found")
        
        col_data = self.df[column_name]
        analysis = {
            'column_name': column_name,
            'data_type': str(col_data.dtype),
            'total_values': len(col_data),
            'missing_values': col_data.isnull().sum(),
            'unique_values': col_data.nunique(),
        }
        
        if col_data.dtype in [np.int64, np.float64]:
            analysis.update({
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'mean': float(col_data.mean()),
                'median': float(col_data.median()),
                'std': float(col_data.std()),
                'quartiles': {
                    'Q1': float(col_data.quantile(0.25)),
                    'Q2': float(col_data.quantile(0.50)),
                    'Q3': float(col_data.quantile(0.75))
                }
            })
        else:
            top_values = col_data.value_counts().head(10).to_dict()
            analysis['top_values'] = {str(k): int(v) for k, v in top_values.items()}
        
        return analysis
# Continuing from Part 1...

# ============================================================================
# INSIGHT GENERATOR (Enhanced)
# ============================================================================
class InsightGenerator:
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
# AI DATA ANALYST (Enhanced)
# ============================================================================
class AIDataAnalyst:
    def __init__(self, df: pd.DataFrame, metadata: Dict):
        self.df = df
        self.metadata = metadata
    
    def analyze_with_ai(self, question: str) -> Dict[str, Any]:
        data_context = self._prepare_context()
        
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
        
        ai_response = UniversalLLM.chat_completion(messages, temperature=0.7, max_tokens=800)
        
        return {
            "question": question,
            "answer": ai_response,
            "data_context": self.metadata
        }
    
    def _prepare_context(self) -> str:
        ctx = []
        ctx.append(f"Rows: {self.metadata['rows']:,}")
        ctx.append(f"Columns: {self.metadata['columns']}")
        ctx.append(f"Numeric: {', '.join(self.metadata['numeric_columns'][:3])}")
        ctx.append(f"Categorical: {', '.join(self.metadata['categorical_columns'][:3])}")
        return "\n".join(ctx)
    
    def suggest_analysis(self) -> List[str]:
        cols_sample = list(self.df.columns[:5])
        
        prompt = f"""Based on this dataset, suggest 5 analysis questions.

Columns: {', '.join(cols_sample)}

Suggest 5 specific questions (be brief)."""
        
        messages = [
            {"role": "system", "content": "You are a data consultant. Be concise."},
            {"role": "user", "content": prompt}
        ]
        
        response = UniversalLLM.chat_completion(messages, temperature=0.8, max_tokens=300)
        
        suggestions = [line.strip() for line in response.split('\n') if line.strip() and not line.strip().startswith('#')]
        return suggestions[:5]

# ============================================================================
# CHART GENERATOR
# ============================================================================
class ChartGenerator:
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def create_chart(self, chart_type: str, x_col: str, y_col: Optional[str] = None, 
                     color_col: Optional[str] = None) -> Dict[str, Any]:
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
        suggestions = []
        numeric_cols = metadata['numeric_columns']
        categorical_cols = metadata['categorical_columns']
        
        if categorical_cols and numeric_cols:
            suggestions.append({
                "type": "bar",
                "x": categorical_cols[0],
                "y": numeric_cols[0],
                "description": f"Bar chart of {numeric_cols[0]} by {categorical_cols[0]}"
            })
        
        datetime_cols = metadata['datetime_columns']
        if datetime_cols and numeric_cols:
            suggestions.append({
                "type": "line",
                "x": datetime_cols[0],
                "y": numeric_cols[0],
                "description": f"Time series of {numeric_cols[0]}"
            })
        
        if len(numeric_cols) >= 2:
            suggestions.append({
                "type": "scatter",
                "x": numeric_cols[0],
                "y": numeric_cols[1],
                "description": f"Scatter plot of {numeric_cols[1]} vs {numeric_cols[0]}"
            })
        
        if numeric_cols:
            suggestions.append({
                "type": "histogram",
                "x": numeric_cols[0],
                "description": f"Distribution of {numeric_cols[0]}"
            })
        
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
    def __init__(self, dataset_id: str):
        self.dataset_id = dataset_id
        self.dataset = DATASETS.get(dataset_id)
        if not self.dataset:
            raise ValueError("Dataset not found")
    
    def generate_report(self) -> str:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        pdf_path = temp_file.name
        
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
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
            ["Quality Score", f"{self.dataset['quality_score']['overall']}/100 ({self.dataset['quality_score']['grade']})"],
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

class ColumnAnalysisRequest(BaseModel):
    dataset_id: str
    column_name: str

# ============================================================================
# FASTAPI APP
# ============================================================================
app = FastAPI(title="AI Data Analytics Agent", version="3.0")

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
        "message": "AI Data Analytics Agent API - Enhanced Version",
        "version": "3.0",
        "llm_provider": LLM_PROVIDER,
        "model": MODEL_NAME,
        "status": "running",
        "features": [
            "Multiple export formats",
            "Data quality scoring",
            "Column-specific analysis",
            "Sample datasets",
            "Auto-suggested questions"
        ]
    }

@app.post("/api/upload/file")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in FileParser.SUPPORTED_FORMATS:
            raise HTTPException(400, f"Unsupported file format: {file_ext}")
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
        temp_file.write(await file.read())
        temp_file.close()
        
        df = FileParser.parse_file(temp_file.name)
        os.unlink(temp_file.name)
        
        processor = DataProcessor(df)
        clean_report = processor.autoclean()
        
        insight_gen = InsightGenerator(processor.df, processor.metadata)
        insights = insight_gen.generate_insights()
        
        # Auto-generate suggested questions
        analyst = AIDataAnalyst(processor.df, processor.metadata)
        suggested_questions = analyst.suggest_analysis()
        
        dataset_id = str(uuid.uuid4())
        DATASETS[dataset_id] = {
            'filename': file.filename,
            'processor': processor,
            'metadata': processor.metadata,
            'insights': insights,
            'quality_score': processor.quality_score,
            'suggested_questions': suggested_questions,
            'upload_time': datetime.now().isoformat()
        }
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "filename": file.filename,
            "metadata": processor.metadata,
            "clean_report": clean_report,
            "insights": insights,
            "quality_score": processor.quality_score,
            "suggested_questions": suggested_questions,
            "sample_data": processor.df.head(10).to_dict('records')
        }
    
    except Exception as e:
        raise HTTPException(500, f"Error processing file: {str(e)}")

@app.post("/api/query")
async def query_data(request: QueryRequest):
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
    try:
        dataset = DATASETS.get(dataset_id)
        if not dataset:
            raise HTTPException(404, "Dataset not found")
        
        return {
            "success": True,
            "suggestions": dataset.get('suggested_questions', [])
        }
    
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")

@app.post("/api/chart")
async def create_chart(request: ChartRequest):
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

# NEW: Column-specific analysis
@app.post("/api/analyze/column")
async def analyze_column(request: ColumnAnalysisRequest):
    try:
        dataset = DATASETS.get(request.dataset_id)
        if not dataset:
            raise HTTPException(404, "Dataset not found")
        
        analysis = dataset['processor'].analyze_column(request.column_name)
        
        return {
            "success": True,
            "analysis": analysis
        }
    
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")

# NEW: Data quality score
@app.get("/api/quality/{dataset_id}")
async def get_quality_score(dataset_id: str):
    try:
        dataset = DATASETS.get(dataset_id)
        if not dataset:
            raise HTTPException(404, "Dataset not found")
        
        return {
            "success": True,
            "quality_score": dataset['quality_score']
        }
    
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")

# NEW: Export to Excel
@app.get("/api/export/excel/{dataset_id}")
async def export_excel(dataset_id: str):
    try:
        dataset = DATASETS.get(dataset_id)
        if not dataset:
            raise HTTPException(404, "Dataset not found")
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            dataset['processor'].df.to_excel(writer, sheet_name='Data', index=False)
        
        output.seek(0)
        
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=data_{dataset_id}.xlsx"}
        )
    
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")

# NEW: Export to CSV
@app.get("/api/export/csv/{dataset_id}")
async def export_csv(dataset_id: str):
    try:
        dataset = DATASETS.get(dataset_id)
        if not dataset:
            raise HTTPException(404, "Dataset not found")
        
        output = io.StringIO()
        dataset['processor'].df.to_csv(output, index=False)
        output.seek(0)
        
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=data_{dataset_id}.csv"}
        )
    
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")

# NEW: Export to JSON
@app.get("/api/export/json/{dataset_id}")
async def export_json(dataset_id: str):
    try:
        dataset = DATASETS.get(dataset_id)
        if not dataset:
            raise HTTPException(404, "Dataset not found")
        
        json_data = dataset['processor'].df.to_json(orient='records')
        
        return StreamingResponse(
            iter([json_data]),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=data_{dataset_id}.json"}
        )
    
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")

@app.get("/api/export/pdf/{dataset_id}")
async def export_pdf(dataset_id: str):
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

# NEW: Get sample datasets
@app.get("/api/samples")
async def get_sample_datasets():
    return {
        "samples": [
            {
                "id": key,
                "name": value["name"],
                "description": value["description"],
                "rows": len(value["data"]),
                "columns": len(value["data"].columns)
            }
            for key, value in SAMPLE_DATASETS.items()
        ]
    }

# NEW: Load sample dataset
@app.post("/api/samples/{sample_id}")
async def load_sample_dataset(sample_id: str):
    try:
        if sample_id not in SAMPLE_DATASETS:
            raise HTTPException(404, "Sample dataset not found")
        
        sample = SAMPLE_DATASETS[sample_id]
        df = sample["data"].copy()
        
        processor = DataProcessor(df)
        clean_report = processor.autoclean()
        
        insight_gen = InsightGenerator(processor.df, processor.metadata)
        insights = insight_gen.generate_insights()
        
        analyst = AIDataAnalyst(processor.df, processor.metadata)
        suggested_questions = analyst.suggest_analysis()
        
        dataset_id = str(uuid.uuid4())
        DATASETS[dataset_id] = {
            'filename': f'{sample["name"]}.csv',
            'processor': processor,
            'metadata': processor.metadata,
            'insights': insights,
            'quality_score': processor.quality_score,
            'suggested_questions': suggested_questions,
            'upload_time': datetime.now().isoformat()
        }
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "filename": sample["name"],
            "metadata": processor.metadata,
            "clean_report": clean_report,
            "insights": insights,
            "quality_score": processor.quality_score,
            "suggested_questions": suggested_questions,
            "sample_data": processor.df.head(10).to_dict('records')
        }
    
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")

@app.get("/api/datasets")
async def list_datasets():
    return {
        "datasets": [
            {
                "id": k,
                "filename": v['filename'],
                "rows": v['metadata']['rows'],
                "columns": v['metadata']['columns'],
                "quality_score": v['quality_score']['overall'],
                "upload_time": v['upload_time']
            }
            for k, v in DATASETS.items()
        ]
    }

# ============================================================================
# RUN SERVER
# ============================================================================
if __name__ == "__main__":
    print(f"\n🚀 Starting AI Data Analytics Agent - Enhanced Version")
    print(f"📊 LLM Provider: {LLM_PROVIDER.upper()}")
    print(f"🤖 Model: {MODEL_NAME}")
    print(f"🌐 Server: http://localhost:8000")
    print(f"📖 Docs: http://localhost:8000/docs")
    print(f"\n✨ New Features:")
    print(f"   • Auto-suggested questions")
    print(f"   • Data quality scoring")
    print(f"   • Column-specific analysis")
    print(f"   • Multiple export formats (PDF, Excel, CSV, JSON)")
    print(f"   • Sample datasets\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

