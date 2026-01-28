#!/usr/bin/env python3
"""
Data Preparation Automation System
Analyze CSV and generate suggestions using the full framework (P01-P35 + CHK-001-CHK-035).
Usage: python analyze_csv.py your_file.csv
"""

import pandas as pd
import numpy as np
import json
from typing import List, Dict
from datetime import datetime

# ============================================================================
# CLASS: DIAGNOSE PROBLEMS
# ============================================================================

class DataPreparationDiagnostics:
    """Diagnose dataset issues using the P01-P35 framework."""
    
    def __init__(self, df: pd.DataFrame, verbose=True):
        self.df = df
        self.problems_found = []
        self.verbose = verbose
    
    def diagnose(self) -> List[Dict]:
        """Run a full diagnostic pass."""
        print("🔍 Executando diagnóstico completo...\n")
        
        self.check_missing_values()      # P01
        self.check_duplicates()          # P02
        self.check_outliers()            # P03
        self.check_data_types()          # P04
        self.check_categories()          # P05
        self.check_class_balance()       # P09
        self.check_multicollinearity()   # P13
        self.check_constant_columns()    # P26
        self.check_dimensions()          # P31
        
        return self.problems_found
    
    def check_missing_values(self):
        """P01: Detect missing values."""
        missing_rate = (self.df.isnull().sum() / len(self.df) * 100)
        for col, rate in missing_rate[missing_rate > 0].items():
            severity = "CRÍTICO" if rate > 50 else "ALTO" if rate > 10 else "MÉDIO"
            self.problems_found.append({
                'problem_id': 'P01',
                'problem_name': 'Valores em Falta',
                'column': col,
                'severity': severity,
                'rate': rate,
                'message': f'Coluna "{col}" tem {rate:.1f}% de valores em falta',
                'checklist_ref': 'CHK-001'
            })
    
    def check_duplicates(self):
        """P02: Detect duplicates."""
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            pct = (duplicates / len(self.df)) * 100
            self.problems_found.append({
                'problem_id': 'P02',
                'problem_name': 'Dados Duplicados',
                'severity': 'ALTO',
                'count': duplicates,
                'percentage': pct,
                'message': f'Dataset tem {duplicates} linhas duplicadas ({pct:.1f}%)',
                'checklist_ref': 'CHK-002'
            })
    
    def check_outliers(self):
        """P03: Detect outliers with IQR."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0:  # Coluna constante
                continue
            
            outliers = ((self.df[col] < Q1 - 1.5*IQR) | (self.df[col] > Q3 + 1.5*IQR)).sum()
            if outliers > 0:
                pct = (outliers / len(self.df)) * 100
                severity = 'MÉDIO' if pct < 5 else 'ALTO' if pct < 10 else 'CRÍTICO'
                self.problems_found.append({
                    'problem_id': 'P03',
                    'problem_name': 'Outliers',
                    'column': col,
                    'severity': severity,
                    'count': outliers,
                    'percentage': pct,
                    'message': f'Coluna "{col}" tem {outliers} outliers ({pct:.1f}%)',
                    'checklist_ref': 'CHK-003'
                })
    
    def check_data_types(self):
        """P04: Validate data types."""
        issues = []
        for col in self.df.columns:
            # Números armazenados como string?
            if self.df[col].dtype == 'object':
                try:
                    converted = pd.to_numeric(self.df[col], errors='coerce')
                    valid_ratio = converted.notna().sum() / len(self.df)
                    if valid_ratio > 0.8 and valid_ratio < 1.0:
                        issues.append({
                            'column': col,
                            'issue': 'Parece ser numérico mas armazenado como texto',
                            'valid_ratio': valid_ratio
                        })
                except:
                    pass
        
        if issues:
            self.problems_found.append({
                'problem_id': 'P04',
                'problem_name': 'Tipos de Dados Errados',
                'severity': 'ALTO',
                'issues': issues,
                'message': f'Detectados {len(issues)} problemas de tipo de dado',
                'checklist_ref': 'CHK-004'
            })
    
    def check_categories(self):
        """P05: Validate categories."""
        categorical = self.df.select_dtypes(include=['object']).columns
        for col in categorical:
            value_counts = self.df[col].value_counts()
            rare = (value_counts / len(self.df) < 0.01).sum()
            
            if rare > 5:
                self.problems_found.append({
                    'problem_id': 'P25',
                    'problem_name': 'Categorias Raras',
                    'column': col,
                    'severity': 'MÉDIO',
                    'rare_categories': rare,
                    'message': f'Coluna "{col}" tem {rare} categorias raras (<1%)',
                    'checklist_ref': 'CHK-027'
                })
    
    def check_class_balance(self):
        """P09: Check class imbalance."""
        # Detectar possível target column
        last_col = self.df.columns[-1]
        target_candidates = [col for col in self.df.columns 
                            if 'target' in col.lower() or col == 'y' or col == 'label']
        
        target_col = target_candidates[0] if target_candidates else last_col
        
        if self.df[target_col].dtype in ['object', 'int', 'bool']:
            dist = self.df[target_col].value_counts()
            if len(dist) <= 10:  # Classificação
                imbalance_ratio = dist.max() / dist.min()
                if imbalance_ratio > 3:
                    self.problems_found.append({
                        'problem_id': 'P09',
                        'problem_name': 'Classes Desbalanceadas',
                        'column': target_col,
                        'severity': 'ALTO',
                        'ratio': imbalance_ratio,
                        'distribution': dist.to_dict(),
                        'message': f'Classes desbalanceadas em "{target_col}" (razão {imbalance_ratio:.1f}:1)',
                        'checklist_ref': 'CHK-009'
                    })
    
    def check_multicollinearity(self):
        """P13: Detect multicollinearity."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 2:
            corr = self.df[numeric_cols].corr().abs()
            high_corr = []
            
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    if corr.iloc[i, j] > 0.95:
                        high_corr.append({
                            'col1': corr.columns[i],
                            'col2': corr.columns[j],
                            'correlation': float(corr.iloc[i, j])
                        })
            
            if high_corr:
                self.problems_found.append({
                    'problem_id': 'P13',
                    'problem_name': 'Multicolinearidade',
                    'severity': 'MÉDIO',
                    'pairs': high_corr,
                    'message': f'Detectadas {len(high_corr)} pares altamente correlacionados (>0.95)',
                    'checklist_ref': 'CHK-013'
                })
    
    def check_constant_columns(self):
        """P26: Detect constant columns."""
        for col in self.df.columns:
            if self.df[col].nunique() == 1:
                self.problems_found.append({
                    'problem_id': 'P26',
                    'problem_name': 'Coluna Constante',
                    'column': col,
                    'severity': 'MÉDIO',
                    'value': self.df[col].iloc[0],
                    'message': f'Coluna "{col}" tem apenas 1 valor único',
                    'checklist_ref': 'CHK-028'
                })
    
    def check_dimensions(self):
        """P31: Check dimensionality."""
        n_features = self.df.shape[1]
        n_samples = self.df.shape[0]
        ratio = n_features / n_samples
        
        if ratio > 0.5:
            self.problems_found.append({
                'problem_id': 'P31',
                'problem_name': 'Dimensionalidade Alta',
                'severity': 'MÉDIO',
                'ratio': ratio,
                'features': n_features,
                'samples': n_samples,
                'message': f'Muitas features relativo a amostras (razão {ratio:.2f})',
                'checklist_ref': 'CHK-014'
            })


# ============================================================================
# CLASS: QUICK ANALYZER (STATS-ONLY)
# ============================================================================

class QuickAnalyzer:
    """Lightweight analyzer using precomputed CSV stats."""

    @staticmethod
    def analyze_stats(stats: Dict) -> List[Dict]:
        """
        Analyze precomputed stats and return a list of detected problems.
        Expected keys in stats:
        - rows, columns, missing_percentage, duplicates
        - outliers (dict), high_correlations (list)
        - target_distribution (dict), target_column (str)
        """
        problems: List[Dict] = []

        rows = stats.get("rows", 0) or 0
        missing = stats.get("missing_percentage", {}) or {}
        duplicates = stats.get("duplicates", 0) or 0
        outliers = stats.get("outliers", {}) or {}
        high_corr = stats.get("high_correlations", []) or []
        target_dist = stats.get("target_distribution", {}) or {}
        target_col = stats.get("target_column", "")

        for col, rate in missing.items():
            if rate <= 0:
                continue
            severity = "CRÍTICO" if rate > 50 else "ALTO" if rate > 10 else "MÉDIO"
            problems.append({
                "problem_id": "P01",
                "problem_name": "Valores em Falta",
                "column": col,
                "severity": severity,
                "rate": rate,
                "message": f'Coluna "{col}" tem {rate:.1f}% de valores em falta',
                "checklist_ref": "CHK-001"
            })

        if duplicates and rows:
            pct = (duplicates / rows) * 100
            problems.append({
                "problem_id": "P02",
                "problem_name": "Dados Duplicados",
                "severity": "ALTO",
                "count": duplicates,
                "percentage": pct,
                "message": f'Dataset tem {duplicates} linhas duplicadas ({pct:.1f}%)',
                "checklist_ref": "CHK-002"
            })

        for col, outlier_info in outliers.items():
            count = outlier_info.get("count", 0)
            if count <= 0 or not rows:
                continue
            pct = (count / rows) * 100
            severity = "MÉDIO" if pct < 5 else "ALTO" if pct < 10 else "CRÍTICO"
            problems.append({
                "problem_id": "P03",
                "problem_name": "Outliers",
                "column": col,
                "severity": severity,
                "count": count,
                "percentage": pct,
                "message": f'Coluna "{col}" tem {count} outliers ({pct:.1f}%)',
                "checklist_ref": "CHK-003"
            })

        if high_corr:
            problems.append({
                "problem_id": "P13",
                "problem_name": "Multicolinearidade",
                "severity": "MÉDIO",
                "pairs": high_corr,
                "message": f"Detectadas {len(high_corr)} pares altamente correlacionados (>0.95)",
                "checklist_ref": "CHK-013"
            })

        if target_dist and len(target_dist) >= 2:
            values = list(target_dist.values())
            min_val = min(values)
            max_val = max(values)
            if min_val > 0:
                ratio = max_val / min_val
                if ratio > 3:
                    problems.append({
                        "problem_id": "P09",
                        "problem_name": "Classes Desbalanceadas",
                        "column": target_col or "target",
                        "severity": "ALTO",
                        "ratio": ratio,
                        "distribution": target_dist,
                        "message": f'Classes desbalanceadas em "{target_col}" (razão {ratio:.1f}:1)',
                        "checklist_ref": "CHK-009"
                    })

        return problems


# ============================================================================
# CLASS: SOLUTION
# ============================================================================

class SolutionGenerator:
    """Generate Python code snippets for each problem."""
    
    @staticmethod
    def get_code(problem: Dict) -> str:
        """Return a Python snippet tailored to the given problem."""
        
        problem_id = problem.get('problem_id')
        
        if problem_id == 'P01':  # Missing Values
            col = problem.get('column', 'column_name')
            return f"""
# P01: Missing values in "{col}"
missing_rate = (df['{col}'].isnull().sum() / len(df)) * 100
print(f"Missing rate: {{missing_rate:.1f}}%")

if missing_rate < 10:
    # Option 1: Drop rows (few missing)
    df = df.dropna(subset=['{col}'])
elif missing_rate < 50:
    # Option 2: Impute (some missing)
    if df['{col}'].dtype in ['float64', 'int64']:
        df['{col}'] = df['{col}'].fillna(df['{col}'].median())
    else:
        df['{col}'] = df['{col}'].fillna(df['{col}'].mode()[0])
else:
    # Option 3: Drop column (too many missing)
    df = df.drop(columns=['{col}'])
    print(f"Column {{col}} removed due to excessive missingness")
            """
        
        elif problem_id == 'P02':  # Duplicates
            return """
# P02: Duplicate rows
print(f"Duplicates before: {df.duplicated().sum()}")
df = df.drop_duplicates()
print(f"Duplicates after: {df.duplicated().sum()}")
            """
        
        elif problem_id == 'P03':  # Outliers
            col = problem.get('column', 'column_name')
            return f"""
# P03: Outliers in "{col}"
Q1 = df['{col}'].quantile(0.25)
Q3 = df['{col}'].quantile(0.75)
IQR = Q3 - Q1

# Option 1: Remove outliers
df_clean = df[~((df['{col}'] < Q1 - 1.5*IQR) | (df['{col}'] > Q3 + 1.5*IQR))]
print(f"Removed {{len(df) - len(df_clean)}} outlier rows")

# Option 2: Capping (limit values)
df['{col}'] = df['{col}'].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
            """
        
        elif problem_id == 'P04':  # Wrong Types
            return """
# P04: Wrong data types
# Convert strings to numeric where possible
for col in df.select_dtypes(include=['object']).columns:
    try:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        print(f"Column '{col}' converted to numeric")
    except:
        pass

# Convert to datetime where applicable
for col in df.columns:
    if 'date' in col.lower() or 'time' in col.lower():
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            print(f"Column '{col}' converted to datetime")
        except:
            pass

print(df.dtypes)
            """
        
        elif problem_id == 'P09':  # Class Imbalance
            return """
# P09: Class imbalance
from sklearn.utils import resample

print(df['target'].value_counts())

# Option 1: Upsampling (minority class)
df_majority = df[df['target'] == df['target'].value_counts().index[0]]
df_minority = df[df['target'] == df['target'].value_counts().index[1]]

df_minority_upsampled = resample(df_minority,
                                  replace=True,
                                  n_samples=len(df_majority),
                                  random_state=42)
df_balanced = pd.concat([df_majority, df_minority_upsampled])

# Option 2: Use class weights in the model
# from sklearn.ensemble import RandomForestClassifier
# rf = RandomForestClassifier(class_weight='balanced')
            """
        
        elif problem_id == 'P13':  # Multicollinearity
            return """
# P13: Multicollinearity
import matplotlib.pyplot as plt

corr_matrix = df.corr().abs()
high_corr_pairs = []

for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > 0.95:
            high_corr_pairs.append({
                'col1': corr_matrix.columns[i],
                'col2': corr_matrix.columns[j],
                'corr': corr_matrix.iloc[i, j]
            })
            print(f"High correlation: {corr_matrix.columns[i]} <-> {corr_matrix.columns[j]} ({corr_matrix.iloc[i, j]:.3f})")

# Drop one of the highly correlated columns
for pair in high_corr_pairs:
    if pair['col2'] in df.columns:
        df = df.drop(columns=[pair['col2']])
            """
        
        elif problem_id == 'P25':  # Rare Categories
            col = problem.get('column', 'column_name')
            return f"""
# P25: Rare categories in "{col}"
print(df['{col}'].value_counts())

min_freq = len(df) * 0.01
rare_categories = df['{col}'].value_counts()[df['{col}'].value_counts() < min_freq].index

df['{col}'] = df['{col}'].apply(lambda x: 'Other' if x in rare_categories else x)
print(f"Grouped {{len(rare_categories)}} rare categories into 'Other'")
print(df['{col}'].value_counts())
            """
        
        elif problem_id == 'P26':  # Constant Columns
            col = problem.get('column', 'column_name')
            return f"""
# P26: Constant column "{col}"
df = df.drop(columns=['{col}'])
print(f"Column '{col}' removed (zero variance)")
            """
        
        elif problem_id == 'P31':  # High Dimensionality
            return """
# P31: High dimensionality
from sklearn.feature_selection import SelectKBest, f_classif

# Select top 10 features
X = df.drop('target', axis=1)
y = df['target']

selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()].tolist()

df = df[selected_features + ['target']]
print(f"Reduced features from {X.shape[1]} to {len(selected_features)}")
            """
        
        else:
            return "# Solução específica não disponível"


# ============================================================================
# CLASS: REPORT
# ============================================================================

class ExecutiveReport:
    """Generate an executive report."""
    
    def __init__(self, df: pd.DataFrame, problems_found: List[Dict]):
        self.df = df
        self.problems_found = problems_found
        self.report = {}
    
    def generate(self) -> Dict:
        """Build the full report payload."""
        self.report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': self.get_dataset_info(),
            'summary': self.get_summary(),
            'problems_by_severity': self.get_problems_by_severity(),
            'quick_wins': self.get_quick_wins(),
            'recommended_actions': self.get_recommended_actions(),
            'code_snippets': self.get_code_snippets()
        }
        return self.report
    
    def get_dataset_info(self) -> Dict:
        """Basic dataset information."""
        return {
            'shape': f"{self.df.shape[0]} linhas × {self.df.shape[1]} colunas",
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.astype(str).to_dict(),
            'memory_mb': round(self.df.memory_usage(deep=True).sum() / 1024**2, 2),
            'completeness': round(100 * (1 - self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1])), 1)
        }
    
    def get_summary(self) -> Dict:
        """Executive summary."""
        total = len(self.problems_found)
        critical = sum(1 for p in self.problems_found if p['severity'] == 'CRÍTICO')
        high = sum(1 for p in self.problems_found if p['severity'] == 'ALTO')
        medium = sum(1 for p in self.problems_found if p['severity'] == 'MÉDIO')
        
        status = '🔴 CRÍTICO' if critical > 0 else '🟡 ATENÇÃO' if high > 0 else '🟢 BOM'
        
        return {
            'total_problems': total,
            'critical': critical,
            'high': high,
            'medium': medium,
            'status': status,
            'recommendation': 'Resolver problemas críticos antes de treinar modelo'
        }
    
    def get_problems_by_severity(self) -> Dict:
        """Problems grouped by severity."""
        return {
            'CRÍTICO': [p for p in self.problems_found if p['severity'] == 'CRÍTICO'],
            'ALTO': [p for p in self.problems_found if p['severity'] == 'ALTO'],
            'MÉDIO': [p for p in self.problems_found if p['severity'] == 'MÉDIO']
        }
    
    def get_quick_wins(self) -> List[Dict]:
        """Quick wins that are easy to address."""
        quick = [
            p for p in self.problems_found 
            if p['problem_id'] in ['P26', 'P02', 'P25']  # Constantes, duplicados, raras
        ]
        return quick[:5]
    
    def get_recommended_actions(self) -> List[str]:
        """Recommended actions in priority order."""
        actions = []
        
        critical = sum(1 for p in self.problems_found if p['severity'] == 'CRÍTICO')
        if critical > 0:
            actions.append(f"1. ⚠️  URGENTE: Resolver {critical} problema(s) CRÍTICO")
        
        for problem in sorted(self.problems_found, 
                             key=lambda x: {'CRÍTICO': 0, 'ALTO': 1, 'MÉDIO': 2}.get(x['severity'])):
            actions.append(f"   - {problem['problem_id']}: {problem['message']}")
        
        actions.extend([
            "2. Executar código sugerido para cada problema",
            "3. Validar contra checklist CHK-001-CHK-035",
            "4. Re-executar análise para confirmar resoluções"
        ])
        
        return actions
    
    def get_code_snippets(self) -> List[Dict]:
        """Top code suggestions."""
        snippets = []
        
        for problem in sorted(self.problems_found,
                             key=lambda x: {'CRÍTICO': 0, 'ALTO': 1, 'MÉDIO': 2}.get(x['severity'])):
            code = SolutionGenerator.get_code(problem)
            snippets.append({
                'problem_id': problem['problem_id'],
                'problem_name': problem['problem_name'],
                'severity': problem['severity'],
                'message': problem['message'],
                'code': code
            })
        
        return snippets


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def analyze_csv(filepath: str, output_json: str = None):
    """Analyze CSV and generate a report."""
    
    print("="*70)
    print("📊 DATA PREPARATION ANALYSIS")
    print("="*70)
    
    # Carregar CSV
    try:
        df = pd.read_csv(filepath)
        print(f"✅ CSV carregado: {df.shape[0]} linhas × {df.shape[1]} colunas\n")
    except Exception as e:
        print(f"❌ Erro ao carregar CSV: {e}")
        return
    
    # Diagnosticar
    diagnostics = DataPreparationDiagnostics(df)
    problems = diagnostics.diagnose()
    
    print(f"✅ {len(problems)} problema(s) detectado(s)\n")
    
    # Gerar relatório
    report = ExecutiveReport(df, problems)
    full_report = report.generate()
    
    # Imprimir resumo
    summary = full_report['summary']
    print(f"Status: {summary['status']}")
    print(f"Críticos: {summary['critical']} | Altos: {summary['high']} | Médios: {summary['medium']}\n")
    
    # Problemas por severidade
    print("🚨 PROBLEMAS DETECTADOS:")
    print("-"*70)
    
    for severity in ['CRÍTICO', 'ALTO', 'MÉDIO']:
        problems_severity = full_report['problems_by_severity'][severity]
        if problems_severity:
            print(f"\n{severity}:")
            for p in problems_severity:
                print(f"  • {p['problem_id']}: {p['message']}")
    
    # Quick wins
    print(f"\n💡 QUICK WINS (Fáceis de Resolver):")
    print("-"*70)
    for p in full_report['quick_wins']:
        print(f"  • {p['problem_id']}: {p['message']}")
    
    # Próximos passos
    print(f"\n📋 PRÓXIMOS PASSOS:")
    print("-"*70)
    for action in full_report['recommended_actions']:
        print(f"  {action}")
    
    # Código sugerido (primeiros 3)
    print(f"\n🔧 CÓDIGO SUGERIDO (Top 3):")
    print("-"*70)
    for snippet in full_report['code_snippets'][:3]:
        print(f"\n### {snippet['problem_id']}: {snippet['problem_name']}")
        print(f"Severidade: {snippet['severity']}")
        print(f"Mensagem: {snippet['message']}")
        print("```python")
        print(snippet['code'])
        print("```")
    
    # Salvar JSON
    if output_json:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(full_report, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Relatório salvo em: {output_json}")
    
    return full_report


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python analyze_csv.py seu_arquivo.csv [output.json]")
        print("\nExemplo:")
        print("  python analyze_csv.py dados.csv relatorio.json")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    json_file = sys.argv[2] if len(sys.argv) > 2 else "relatorio.json"
    
    report = analyze_csv(csv_file, json_file)
    
    print("\n" + "="*70)
    print("✅ Análise completa!")
    print("="*70)
