# Academic Lab Architecture

## System Overview

```
┌─────────────────────────────────────────┐
│          Academic Lab v2                │
├─────────────────────────────────────────┤
│  Frontend (React/Next.js)               │
│  - Upload datasets                      │
│  - View analysis results                │
│  - See recommendations                  │
├─────────────────────────────────────────┤
│  API (FastAPI/Python)                   │
│  - Data analysis engine                 │
│  - Problem detection                    │
│  - Decision trees                       │
│  - Validation logic                     │
├─────────────────────────────────────────┤
│  Database (PostgreSQL)                  │
│  - Session data                         │
│  - Analysis results                     │
│  - User feedback                        │
└─────────────────────────────────────────┘
```

## Core Components

### 1. Data Analysis Engine (`backend/analyze_csv.py`)

**Responsibility:** Analyze uploaded CSV files

**Key Functions:**
- `QuickAnalyzer.analyze()` - Main analysis function
- `_missing_action_hint()` - P01 decision tree
- `_compute_stats()` - Statistics calculation

**Features:**
- MCAR/MAR/MNAR detection
- Missing value analysis
- Outlier detection
- Type checking
- Edge case handling

### 2. Missing Type Classification (`backend/main.py`)

**Responsibility:** Detect missing value pattern type

**Key Functions:**
- `classify_missing_type()` - MCAR/MAR/MNAR detection
- Heuristic-based approach
- Fast and practical

**Algorithm:**
1. Check correlation with other columns
2. If correlated with categorical → MAR
3. If > 50% missing → MNAR
4. Otherwise → MCAR

### 3. Validation System (`backend/state.py`)

**Responsibility:** Track and report validation status

**Key Features:**
- CHK-001: Missing values (13 criteria)
- In-memory validation report
- Progress tracking
- Acceptance criteria checking

### 4. API Endpoints (`backend/main.py`)

**Available:**
```
POST /analyze
  - Input: CSV file
  - Output: Problems + recommendations

GET /session/{session_id}
  - Output: Session state + checklist_report

GET /health
  - Output: Server status
```

## Data Flow

```
1. Student uploads CSV
   ↓
2. Backend receives file
   ↓
3. QuickAnalyzer.analyze()
   ├─ Compute statistics
   ├─ Classify missing types (MCAR/MAR/MNAR)
   ├─ Detect problems (P01-P35)
   ├─ Apply decision trees
   └─ Build validation report
   ↓
4. API returns analysis + recommendations
   ↓
5. Frontend displays results
   ↓
6. Student sees actionable guidance
```

## Testing Strategy

### Unit Tests (18 total)

**Phase 1 (9 tests):**
- MCAR/MAR/MNAR classification
- P01 decision tree
- CHK-001 validation

**Phase 2 (9 tests):**
- Edge cases (empty, single, large datasets)
- Real-world datasets (Finance, Healthcare, IoT)
- Performance benchmarks

### CI/CD

- GitHub Actions runs tests on every push
- Must pass before merge to main
- Main → staging → production workflow

## Performance

### Benchmarks
- Missing classification: < 1 second (100k rows)
- Missing classification: < 5 seconds (1M rows)
- Memory usage: < 500MB (1M rows)
- API response: < 2 seconds

### Optimization
- Efficient correlation calculation
- Vectorized operations (NumPy/Pandas)
- Minimal data copying
- Caching (if applicable)

## Outlier Detection Logic

### Priority
1. **Context wins over thresholds**: if a column has `min_expected`/`max_expected`, we use that range.
2. **Sensitivity applies only without context**: z-score threshold (default **3.0**).
3. **Dismiss is session-local**: marking P03 as false alarm affects only the current session.

### Context Validation Warnings
When context is provided:
- If the context range does **not overlap** the data range, we keep it but return a warning.
- If the context range **covers all data points**, we return an info warning (no outliers within range).

## Deployment

### Environments

```
Development (Local)
  - http://localhost:8000
  - Debug mode enabled

Staging
  - https://staging-api.academiclab.com
  - Pre-production testing
  - Student testing

Production
  - https://api.academiclab.com
  - Live student data
  - Full performance required
```

### Deployment Tools
- Docker for containerization
- systemd for service management
- GitHub Actions for CI/CD
- PostgreSQL for data storage

## Security

### Data Protection
- No sensitive data logging
- Input validation on all endpoints
- SQL injection prevention
- XSS protection

### Access Control
- API rate limiting (if needed)
- Session management
- CORS configuration
- Environment secrets (.env)

## Future Improvements

1. **More Problems:** Implement P04-P35 detection
2. **Machine Learning:** Predict best data preparation approach
3. **Feedback Loop:** Learn from student actions
4. **Collaboration:** Multi-student workspace
5. **Integration:** Connect to GitHub, Kaggle, etc.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Reporting bugs
- Suggesting features
- Submitting code

---

**Questions?** Open an issue or contact us at support@academiclab.com
