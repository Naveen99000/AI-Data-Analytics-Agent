"""
AI Data Analytics Platform - Final Fixed Backend
All Issues Resolved: Chart Builder + Column Insights
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional, Tuple
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
import tempfile
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# PDF Generation
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors as rb_colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# LLM
from groq import Groq
import google.generativeai as genai

load_dotenv()

# ============================================================================
# LLM CONFIGURATION
# ============================================================================
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
groq_client = None
gemini_model = None

if LLM_PROVIDER == "groq":
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        groq_client = Groq(api_key=groq_api_key)
        MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

elif LLM_PROVIDER == "gemini":
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
        gemini_model = genai.GenerativeModel("gemini-pro")
        MODEL_NAME = "gemini-pro"

# ============================================================================
# COLOR SCHEMES
# ============================================================================
COLOR_SCHEMES = {
    'squid_game_primary': ['#CE1141', '#00A86E', '#F7C325', '#FF006D', '#7E57C2'],
    'squid_game_vibrant': ['#8C205C', '#BF4B96', '#A6859D', '#0E7373', '#ADD9D1'],
    'professional': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
}

# ============================================================================
# GLOBAL STORAGE
# ============================================================================
DATASETS = {}

SAMPLE_DATASETS = {
    "sales": {
        "name": "E-commerce Sales",
        "data": pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100),
            'product': np.random.choice(['Laptop', 'Phone', 'Tablet'], 100),
            'quantity': np.random.randint(1, 10, 100),
            'price': np.random.uniform(100, 2000, 100).round(2),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
        })
    }
}

# ============================================================================
# MATPLOTLIB SETUP
# ============================================================================
def setup_plot_style():
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'figure.dpi': 100,
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 12,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.facecolor': '#F8F9FA',
        'figure.facecolor': 'white'
    })
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=COLOR_SCHEMES['squid_game_vibrant'])

setup_plot_style()

# ============================================================================
# FILE PARSER
# ============================================================================
class FileParser:
    SUPPORTED_FORMATS = {'.csv', '.xlsx', '.xls', '.json', '.parquet'}
    
    @staticmethod
    def parse_file(file_path: str) -> pd.DataFrame:
        ext = Path(file_path).suffix.lower()
        
        if ext == '.csv':
            return pd.read_csv(file_path)
        elif ext in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        elif ext == '.json':
            return pd.read_json(file_path)
        elif ext == '.parquet':
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported format: {ext}")

# ============================================================================
# DATA PROCESSOR
# ============================================================================
class DataProcessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
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
            'datetime_columns': self.df.select_dtypes(include=['datetime']).columns.tolist()
        }
    
    def calculate_quality_score(self):
        total_cells = self.df.shape[0] * self.df.shape[1]
        missing_cells = self.df.isnull().sum().sum()
        completeness = ((total_cells - missing_cells) / total_cells) * 100 if total_cells > 0 else 0
        
        uniqueness = ((len(self.df) - self.df.duplicated().sum()) / len(self.df)) * 100 if len(self.df) > 0 else 0
        
        overall = (completeness + uniqueness) / 2
        
        self.quality_score = {
            'overall': round(overall, 1),
            'completeness': round(completeness, 1),
            'uniqueness': round(uniqueness, 1),
            'grade': self._get_grade(overall)
        }
    
    def _get_grade(self, score: float) -> str:
        if score >= 90: return 'A'
        elif score >= 80: return 'B'
        elif score >= 70: return 'C'
        else: return 'D'
    
    def autoclean(self) -> Dict[str, Any]:
        report = {"steps": [], "rows_before": len(self.df)}
        
        dupes = self.df.duplicated().sum()
        if dupes > 0:
            self.df = self.df.drop_duplicates()
            report['steps'].append(f"Removed {dupes} duplicates")
        
        for col in self.metadata['numeric_columns']:
            if self.df[col].isnull().sum() > 0:
                self.df[col].fillna(self.df[col].median(), inplace=True)
                report['steps'].append(f"Filled {col} missing values")
        
        report["rows_after"] = len(self.df)
        self.analyze_data()
        self.calculate_quality_score()
        return report

# ============================================================================
# AI ANALYST WITH SMART CHART GENERATION
# ============================================================================
class AIAnalyst:
    def __init__(self, df: pd.DataFrame, metadata: Dict):
        self.df = df
        self.metadata = metadata
    
    def analyze_question(self, question: str) -> Tuple[str, Optional[str]]:
        """Analyze data and generate suitable chart based on question"""
        
        context = f"""
Dataset: {self.metadata['rows']} rows, {self.metadata['columns']} columns
Columns: {', '.join(self.df.columns[:5])}
Numeric: {', '.join(self.metadata['numeric_columns'][:3])}
Categorical: {', '.join(self.metadata['categorical_columns'][:3])}

Question: {question}

Provide a clear, concise answer with specific insights and data points.
"""
        
        messages = [
            {"role": "system", "content": "You are a data analyst. Be concise and data-driven."},
            {"role": "user", "content": context}
        ]
        
        try:
            if groq_client:
                response = groq_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=600
                )
                answer = response.choices[0].message.content
            elif gemini_model:
                prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
                response = gemini_model.generate_content(prompt)
                answer = response.text
            else:
                answer = "LLM not configured"
            
            chart_path = self._generate_smart_chart(question)
            
            return answer, chart_path
        
        except Exception as e:
            return f"Error: {str(e)}", None
    
    def _generate_smart_chart(self, question: str) -> Optional[str]:
        """Generate smart chart based on question context"""
        question_lower = question.lower()
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if any(word in question_lower for word in ['top', 'most', 'highest', 'best', 'largest', 'greatest']):
                if self.metadata['numeric_columns'] and self.metadata['categorical_columns']:
                    cat_col = self.metadata['categorical_columns'][0]
                    num_col = self.metadata['numeric_columns'][0]
                    
                    grouped = self.df.groupby(cat_col)[num_col].sum().sort_values(ascending=False).head(10)
                    
                    ax.bar(range(len(grouped)), grouped.values, color=COLOR_SCHEMES['squid_game_vibrant'])
                    ax.set_xticks(range(len(grouped)))
                    ax.set_xticklabels(grouped.index, rotation=45, ha='right')
                    ax.set_title(f'Top 10 {cat_col} by {num_col}', fontsize=14, fontweight='bold')
                    ax.set_ylabel(num_col)
                    ax.grid(axis='y', alpha=0.3)
            
            elif any(word in question_lower for word in ['average', 'mean']):
                if self.metadata['numeric_columns']:
                    num_col = self.metadata['numeric_columns'][0]
                    if self.metadata['categorical_columns']:
                        cat_col = self.metadata['categorical_columns'][0]
                        grouped = self.df.groupby(cat_col)[num_col].mean().head(10)
                        
                        ax.bar(range(len(grouped)), grouped.values, color=COLOR_SCHEMES['squid_game_vibrant'][1])
                        ax.set_xticks(range(len(grouped)))
                        ax.set_xticklabels(grouped.index, rotation=45, ha='right')
                        ax.set_title(f'Average {num_col} by {cat_col}', fontsize=14, fontweight='bold')
                        ax.set_ylabel(f'Average {num_col}')
                        ax.grid(axis='y', alpha=0.3)
            
            elif any(word in question_lower for word in ['trend', 'over time', 'change', 'growth', 'timeline']):
                if self.metadata['datetime_columns'] and self.metadata['numeric_columns']:
                    date_col = self.metadata['datetime_columns'][0]
                    num_col = self.metadata['numeric_columns'][0]
                    
                    df_sorted = self.df.sort_values(date_col)
                    ax.plot(df_sorted[date_col], df_sorted[num_col], 
                           color=COLOR_SCHEMES['squid_game_vibrant'][2], linewidth=2, marker='o')
                    ax.set_title(f'{num_col} Trend Over Time', fontsize=14, fontweight='bold')
                    ax.set_xlabel(date_col)
                    ax.set_ylabel(num_col)
                    ax.grid(alpha=0.3)
                    plt.xticks(rotation=45)
            
            elif any(word in question_lower for word in ['distribution', 'spread', 'range']):
                if self.metadata['numeric_columns']:
                    num_col = self.metadata['numeric_columns'][0]
                    data = self.df[num_col].dropna()
                    
                    ax.hist(data, bins=30, color=COLOR_SCHEMES['squid_game_vibrant'][3], alpha=0.7, edgecolor='black')
                    ax.set_title(f'Distribution of {num_col}', fontsize=14, fontweight='bold')
                    ax.set_xlabel(num_col)
                    ax.set_ylabel('Frequency')
                    ax.grid(alpha=0.3)
            
            elif any(word in question_lower for word in ['count', 'how many', 'number of']):
                if self.metadata['categorical_columns']:
                    cat_col = self.metadata['categorical_columns'][0]
                    data = self.df[cat_col].value_counts().head(10)
                    
                    ax.bar(range(len(data)), data.values, color=COLOR_SCHEMES['squid_game_vibrant'][4])
                    ax.set_xticks(range(len(data)))
                    ax.set_xticklabels(data.index, rotation=45, ha='right')
                    ax.set_title(f'Count by {cat_col}', fontsize=14, fontweight='bold')
                    ax.set_ylabel('Count')
                    ax.grid(axis='y', alpha=0.3)
            
            elif any(word in question_lower for word in ['compare', 'comparison', 'vs', 'versus']):
                if len(self.metadata['numeric_columns']) >= 2:
                    num_col1 = self.metadata['numeric_columns'][0]
                    num_col2 = self.metadata['numeric_columns'][1]
                    
                    if self.metadata['categorical_columns']:
                        cat_col = self.metadata['categorical_columns'][0]
                        grouped = self.df.groupby(cat_col)[[num_col1, num_col2]].sum().head(5)
                        
                        x = range(len(grouped))
                        width = 0.35
                        
                        ax.bar([i - width/2 for i in x], grouped[num_col1], width, 
                              label=num_col1, color=COLOR_SCHEMES['squid_game_vibrant'][0])
                        ax.bar([i + width/2 for i in x], grouped[num_col2], width, 
                              label=num_col2, color=COLOR_SCHEMES['squid_game_vibrant'][1])
                        
                        ax.set_xticks(x)
                        ax.set_xticklabels(grouped.index, rotation=45, ha='right')
                        ax.set_title(f'Comparison: {num_col1} vs {num_col2}', fontsize=14, fontweight='bold')
                        ax.legend()
                        ax.grid(axis='y', alpha=0.3)
            
            else:
                if self.metadata['numeric_columns']:
                    num_col = self.metadata['numeric_columns'][0]
                    if self.metadata['categorical_columns']:
                        cat_col = self.metadata['categorical_columns'][0]
                        grouped = self.df.groupby(cat_col)[num_col].sum().sort_values(ascending=False).head(10)
                        
                        ax.bar(range(len(grouped)), grouped.values, color=COLOR_SCHEMES['squid_game_vibrant'])
                        ax.set_xticks(range(len(grouped)))
                        ax.set_xticklabels(grouped.index, rotation=45, ha='right')
                        ax.set_title(f'Analysis Result', fontsize=14, fontweight='bold')
                        ax.set_ylabel(num_col)
                        ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            plt.savefig(temp_file.name, dpi=100, bbox_inches='tight')
            plt.close()
            
            return temp_file.name
        
        except Exception as e:
            print(f"Chart generation error: {e}")
            return None
    
    def quick_summary(self) -> str:
        prompt = f"""Generate a 3-sentence summary for this dataset:
- {self.metadata['rows']} rows, {self.metadata['columns']} columns
- Columns: {', '.join(self.df.columns[:5])}
Be concise."""
        
        messages = [
            {"role": "system", "content": "You are a data analyst."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            if groq_client:
                response = groq_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=200
                )
                return response.choices[0].message.content
            elif gemini_model:
                prompt_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
                response = gemini_model.generate_content(prompt_text)
                return response.text
            else:
                return "LLM not configured"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def suggest_questions(self) -> List[str]:
        prompt = f"""Suggest 5 analysis questions for this dataset:
Columns: {', '.join(self.df.columns[:5])}

List 5 questions (one per line)."""
        
        messages = [
            {"role": "system", "content": "You are a data consultant."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            if groq_client:
                response = groq_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.8,
                    max_tokens=300
                )
                text = response.choices[0].message.content
            elif gemini_model:
                prompt_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
                response = gemini_model.generate_content(prompt_text)
                text = response.text
            else:
                return []
            
            questions = [line.strip() for line in text.split('\n') if line.strip() and len(line.strip()) > 10]
            return questions[:5]
        
        except Exception as e:
            return []
    
    def get_column_insights(self, column: str) -> Dict[str, Any]:
        """Get column insights with proper type conversion - FIXED"""
        if column not in self.df.columns:
            return {"error": "Column not found"}
        
        col_data = self.df[column]
        insights = {
            'name': column,
            'type': str(col_data.dtype),
            'total': int(len(col_data)),  # Convert to Python int
            'missing': int(col_data.isnull().sum()),  # Convert to Python int
            'unique': int(col_data.nunique())  # Convert to Python int
        }
        
        if col_data.dtype in [np.int64, np.float64]:
            insights.update({
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'mean': float(col_data.mean()),
                'median': float(col_data.median()),
                'std': float(col_data.std())
            })
        else:
            top = col_data.value_counts().head(5)
            insights['top_values'] = {str(k): int(v) for k, v in top.items()}
        
        return insights

# ============================================================================
# AUTO CHART GENERATOR
# ============================================================================
class AutoChartGenerator:
    def __init__(self, df: pd.DataFrame, metadata: Dict):
        self.df = df
        self.metadata = metadata
    
    def generate_auto_charts(self) -> List[Dict[str, str]]:
        charts = []
        
        if self.metadata['numeric_columns']:
            chart_path = self._create_distribution_chart()
            if chart_path:
                charts.append({
                    'type': 'distribution',
                    'title': 'Numeric Distribution',
                    'path': chart_path
                })
        
        if self.metadata['categorical_columns']:
            chart_path = self._create_categorical_chart()
            if chart_path:
                charts.append({
                    'type': 'categorical',
                    'title': 'Category Distribution',
                    'path': chart_path
                })
        
        if len(self.metadata['numeric_columns']) >= 2:
            chart_path = self._create_correlation_chart()
            if chart_path:
                charts.append({
                    'type': 'correlation',
                    'title': 'Correlation Matrix',
                    'path': chart_path
                })
        
        if self.metadata['datetime_columns'] and self.metadata['numeric_columns']:
            chart_path = self._create_timeseries_chart()
            if chart_path:
                charts.append({
                    'type': 'timeseries',
                    'title': 'Time Series',
                    'path': chart_path
                })
        
        return charts
    
    def _create_distribution_chart(self) -> Optional[str]:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            col = self.metadata['numeric_columns'][0]
            data = self.df[col].dropna()
            
            ax.hist(data, bins=30, color=COLOR_SCHEMES['squid_game_vibrant'][0], alpha=0.7, edgecolor='black')
            ax.set_title(f'Distribution of {col}', fontsize=14, fontweight='bold')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            ax.grid(alpha=0.3)
            
            plt.tight_layout()
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            plt.savefig(temp_file.name, dpi=100, bbox_inches='tight')
            plt.close()
            
            return temp_file.name
        except:
            return None
    
    def _create_categorical_chart(self) -> Optional[str]:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            col = self.metadata['categorical_columns'][0]
            data = self.df[col].value_counts().head(10)
            
            ax.bar(range(len(data)), data.values, color=COLOR_SCHEMES['squid_game_vibrant'])
            ax.set_xticks(range(len(data)))
            ax.set_xticklabels(data.index, rotation=45, ha='right')
            ax.set_title(f'Top 10 {col}', fontsize=14, fontweight='bold')
            ax.set_ylabel('Count')
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            plt.savefig(temp_file.name, dpi=100, bbox_inches='tight')
            plt.close()
            
            return temp_file.name
        except:
            return None
    
    def _create_correlation_chart(self) -> Optional[str]:
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            numeric_df = self.df[self.metadata['numeric_columns'][:5]]
            corr = numeric_df.corr()
            
            im = ax.imshow(corr, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
            
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha='right')
            ax.set_yticklabels(corr.columns)
            
            for i in range(len(corr.columns)):
                for j in range(len(corr.columns)):
                    text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=10)
            
            ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
            plt.colorbar(im, ax=ax)
            
            plt.tight_layout()
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            plt.savefig(temp_file.name, dpi=100, bbox_inches='tight')
            plt.close()
            
            return temp_file.name
        except:
            return None
    
    def _create_timeseries_chart(self) -> Optional[str]:
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            date_col = self.metadata['datetime_columns'][0]
            numeric_col = self.metadata['numeric_columns'][0]
            
            df_sorted = self.df.sort_values(date_col)
            
            ax.plot(df_sorted[date_col], df_sorted[numeric_col], 
                   color=COLOR_SCHEMES['squid_game_vibrant'][2], linewidth=2)
            ax.set_title(f'{numeric_col} over Time', fontsize=14, fontweight='bold')
            ax.set_xlabel(date_col)
            ax.set_ylabel(numeric_col)
            ax.grid(alpha=0.3)
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            plt.savefig(temp_file.name, dpi=100, bbox_inches='tight')
            plt.close()
            
            return temp_file.name
        except:
            return None

# ============================================================================
# PDF GENERATOR
# ============================================================================
class PDFGenerator:
    def __init__(self, dataset_id: str):
        self.dataset = DATASETS.get(dataset_id)
        if not self.dataset:
            raise ValueError("Dataset not found")
    
    def generate(self) -> str:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        doc = SimpleDocTemplate(temp_file.name, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=22,
            textColor=rb_colors.HexColor('#CE1141'),
            alignment=TA_CENTER,
            spaceAfter=20
        )
        
        story.append(Paragraph("Data Analysis Report", title_style))
        story.append(Spacer(1, 0.3*inch))
        
        overview_data = [
            ["Metric", "Value"],
            ["Rows", f"{self.dataset['metadata']['rows']:,}"],
            ["Columns", str(self.dataset['metadata']['columns'])],
            ["Quality", f"{self.dataset['quality_score']['overall']}% ({self.dataset['quality_score']['grade']})"]
        ]
        
        table = Table(overview_data, colWidths=[3*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), rb_colors.HexColor('#CE1141')),
            ('TEXTCOLOR', (0, 0), (-1, 0), rb_colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, rb_colors.black)
        ]))
        
        story.append(table)
        doc.build(story)
        
        return temp_file.name

# ============================================================================
# API MODELS
# ============================================================================
class QueryRequest(BaseModel):
    dataset_id: str
    question: str

class ColumnRequest(BaseModel):
    dataset_id: str
    column_name: str

# ============================================================================
# FASTAPI APP
# ============================================================================
app = FastAPI(title="AI Data Analytics", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
def read_root():
    return {
        "message": "AI Data Analytics - Final Edition",
        "version": "1.0",
        "status": "running",
        "llm_provider": LLM_PROVIDER,
        "model": MODEL_NAME
    }

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix)
        temp_file.write(await file.read())
        temp_file.close()
        
        df = FileParser.parse_file(temp_file.name)
        os.unlink(temp_file.name)
        
        processor = DataProcessor(df)
        clean_report = processor.autoclean()
        
        analyst = AIAnalyst(processor.df, processor.metadata)
        summary = analyst.quick_summary()
        suggestions = analyst.suggest_questions()
        
        chart_generator = AutoChartGenerator(processor.df, processor.metadata)
        auto_charts = chart_generator.generate_auto_charts()
        
        charts_data = []
        for chart in auto_charts:
            with open(chart['path'], 'rb') as f:
                chart_data = base64.b64encode(f.read()).decode()
                charts_data.append({
                    'type': chart['type'],
                    'title': chart['title'],
                    'data': chart_data
                })
            os.unlink(chart['path'])
        
        dataset_id = str(uuid.uuid4())
        DATASETS[dataset_id] = {
            'filename': file.filename,
            'processor': processor,
            'analyst': analyst,
            'metadata': processor.metadata,
            'quality_score': processor.quality_score,
            'summary': summary,
            'suggestions': suggestions,
            'auto_charts': charts_data,
            'timestamp': datetime.now().isoformat()
        }
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "filename": file.filename,
            "metadata": processor.metadata,
            "quality_score": processor.quality_score,
            "summary": summary,
            "suggestions": suggestions,
            "clean_report": clean_report,
            "sample": processor.df.head(5).to_dict('records'),
            "auto_charts": charts_data
        }
    
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/query")
async def query_data(request: QueryRequest):
    try:
        dataset = DATASETS.get(request.dataset_id)
        if not dataset:
            raise HTTPException(404, "Dataset not found")
        
        answer, chart_path = dataset['analyst'].analyze_question(request.question)
        
        result = {
            "success": True,
            "answer": answer,
            "has_chart": chart_path is not None
        }
        
        if chart_path:
            with open(chart_path, 'rb') as f:
                chart_data = base64.b64encode(f.read()).decode()
                result['chart_data'] = chart_data
            os.unlink(chart_path)
        
        return result
    
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/suggestions/{dataset_id}")
async def get_suggestions(dataset_id: str):
    dataset = DATASETS.get(dataset_id)
    if not dataset:
        raise HTTPException(404, "Dataset not found")
    
    return {
        "success": True,
        "suggestions": dataset['suggestions']
    }

@app.post("/api/column-insights")
async def column_insights(request: ColumnRequest):
    try:
        dataset = DATASETS.get(request.dataset_id)
        if not dataset:
            raise HTTPException(404, "Dataset not found")
        
        insights = dataset['analyst'].get_column_insights(request.column_name)
        
        return {
            "success": True,
            "insights": insights
        }
    
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/chart/custom")
async def create_custom_chart(request: dict):
    """Create custom chart - FINAL FIXED VERSION"""
    try:
        dataset = DATASETS.get(request['dataset_id'])
        if not dataset:
            return {"success": False, "error": "Dataset not found"}
        
        df = dataset['processor'].df
        chart_type = request['chart_type']
        x_col = request.get('x_column')
        y_col = request.get('y_column')
        
        if x_col not in df.columns:
            return {"success": False, "error": f"Column '{x_col}' not found"}
        
        if y_col and y_col not in df.columns:
            return {"success": False, "error": f"Column '{y_col}' not found"}
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        try:
            if chart_type == 'bar':
                if y_col:
                    grouped = df.groupby(x_col)[y_col].sum().sort_values(ascending=False).head(10)
                    ax.bar(range(len(grouped)), grouped.values, color=COLOR_SCHEMES['squid_game_vibrant'])
                    ax.set_xticks(range(len(grouped)))
                    ax.set_xticklabels(grouped.index, rotation=45, ha='right')
                    ax.set_title(f'{y_col} by {x_col}', fontsize=14, fontweight='bold')
                    ax.set_ylabel(y_col)
                else:
                    data = df[x_col].value_counts().head(10)
                    ax.bar(range(len(data)), data.values, color=COLOR_SCHEMES['squid_game_vibrant'])
                    ax.set_xticks(range(len(data)))
                    ax.set_xticklabels(data.index, rotation=45, ha='right')
                    ax.set_title(f'Count by {x_col}', fontsize=14, fontweight='bold')
                    ax.set_ylabel('Count')
            
            elif chart_type == 'line' and y_col:
                df_sorted = df.sort_values(x_col).head(100)
                ax.plot(df_sorted[x_col], df_sorted[y_col], 
                       color=COLOR_SCHEMES['squid_game_vibrant'][0], linewidth=2, marker='o')
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f'{y_col} vs {x_col}', fontsize=14, fontweight='bold')
                plt.xticks(rotation=45, ha='right')
            
            elif chart_type == 'scatter' and y_col:
                ax.scatter(df[x_col], df[y_col], 
                          color=COLOR_SCHEMES['squid_game_vibrant'][1], alpha=0.6, s=50)
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f'{y_col} vs {x_col}', fontsize=14, fontweight='bold')
            
            elif chart_type == 'histogram':
                data = df[x_col].dropna()
                ax.hist(data, bins=30, color=COLOR_SCHEMES['squid_game_vibrant'][2], alpha=0.7, edgecolor='black')
                ax.set_xlabel(x_col)
                ax.set_ylabel('Frequency')
                ax.set_title(f'Distribution of {x_col}', fontsize=14, fontweight='bold')
            
            else:
                plt.close()
                return {"success": False, "error": f"Invalid chart type or missing Y column"}
            
            ax.grid(alpha=0.3)
            plt.tight_layout()
            
            # Use BytesIO instead of temp file - FIXED
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            chart_data = base64.b64encode(buf.read()).decode()
            buf.close()
            plt.close()
            
            return {"success": True, "chart_data": chart_data}
        
        except Exception as plot_error:
            plt.close()
            return {"success": False, "error": f"Chart generation failed: {str(plot_error)}"}
    
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}

@app.get("/api/export/pdf/{dataset_id}")
async def export_pdf(dataset_id: str):
    try:
        generator = PDFGenerator(dataset_id)
        pdf_path = generator.generate()
        
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=f"report_{dataset_id[:8]}.pdf"
        )
    
    except Exception as e:
        raise HTTPException(500, str(e))

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
            headers={"Content-Disposition": f"attachment; filename=data_{dataset_id[:8]}.csv"}
        )
    
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/export/excel/{dataset_id}")
async def export_excel(dataset_id: str):
    try:
        dataset = DATASETS.get(dataset_id)
        if not dataset:
            raise HTTPException(404, "Dataset not found")
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            dataset['processor'].df.to_excel(writer, index=False)
        output.seek(0)
        
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=data_{dataset_id[:8]}.xlsx"}
        )
    
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/samples")
async def get_samples():
    return {
        "samples": [
            {
                "id": key,
                "name": value["name"],
                "rows": len(value["data"])
            }
            for key, value in SAMPLE_DATASETS.items()
        ]
    }

@app.post("/api/samples/{sample_id}")
async def load_sample(sample_id: str):
    if sample_id not in SAMPLE_DATASETS:
        raise HTTPException(404, "Sample not found")
    
    sample = SAMPLE_DATASETS[sample_id]
    df = sample["data"].copy()
    
    processor = DataProcessor(df)
    clean_report = processor.autoclean()
    
    analyst = AIAnalyst(processor.df, processor.metadata)
    summary = analyst.quick_summary()
    suggestions = analyst.suggest_questions()
    
    chart_generator = AutoChartGenerator(processor.df, processor.metadata)
    auto_charts = chart_generator.generate_auto_charts()
    
    charts_data = []
    for chart in auto_charts:
        with open(chart['path'], 'rb') as f:
            chart_data = base64.b64encode(f.read()).decode()
            charts_data.append({
                'type': chart['type'],
                'title': chart['title'],
                'data': chart_data
            })
        os.unlink(chart['path'])
    
    dataset_id = str(uuid.uuid4())
    DATASETS[dataset_id] = {
        'filename': f'{sample["name"]}.csv',
        'processor': processor,
        'analyst': analyst,
        'metadata': processor.metadata,
        'quality_score': processor.quality_score,
        'summary': summary,
        'suggestions': suggestions,
        'auto_charts': charts_data,
        'timestamp': datetime.now().isoformat()
    }
    
    return {
        "success": True,
        "dataset_id": dataset_id,
        "filename": sample["name"],
        "metadata": processor.metadata,
        "quality_score": processor.quality_score,
        "summary": summary,
        "suggestions": suggestions,
        "clean_report": clean_report,
        "sample": processor.df.head(5).to_dict('records'),
        "auto_charts": charts_data
    }

# ============================================================================
# RUN
# ============================================================================
if __name__ == "__main__":
    print("\n🎯 AI Data Analytics - Final Edition")
    print(f"📊 LLM: {LLM_PROVIDER.upper()} ({MODEL_NAME})")
    print("🌐 Server: http://localhost:8000")
    print("📖 Docs: http://localhost:8000/docs")
    print("✨ All features working: Chart Builder + Column Insights\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
