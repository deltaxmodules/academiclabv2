# Academic Lab v2

**Advanced Data Preparation Framework for Data Scientists**

## What is Academic Lab?

Academic Lab is an intelligent educational platform that helps students master data preparation - one of the most important skills in data science.

## Key Features ✨

### 🔍 Smart Missing Values Detection
- Automatic classification: MCAR, MAR, or MNAR
- Context-aware recommendations
- Works with any dataset

### 🎯 Intelligent Decision Trees
- Personalized guidance per column
- Considers dataset size and data type
- Explains reasoning

### ✅ Comprehensive Validation
- 88 validation criteria
- Real-time checklist feedback
- Production-ready data

### 📊 Real-World Data Support
- Tested with Finance, Healthcare, IoT datasets
- Handles 1M+ rows
- Performance optimized

## Quick Start

### For Students

1. **Access:** https://academiclab.com
2. **Upload:** Your CSV dataset
3. **Analyze:** Get personalized recommendations
4. **Learn:** Understand data quality issues
5. **Improve:** Follow guidance to prepare data

### For Developers

```bash
git clone https://github.com/deltaxmodules/academiclabv2.git
cd academiclabv2

# Setup
python -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt

# Run tests
pytest -q
# Expected: 18 passed ✅

# Start server
uvicorn backend.main:app --reload
# Visit: http://localhost:8000
```

## Documentation

- [Staging Runbook](STAGING_RUNBOOK.md) - How to deploy
- [Branch Protection](BRANCH_PROTECTION.md) - GitHub setup
- [Contributing](CONTRIBUTING.md) - How to contribute
- [Architecture](ARCHITECTURE.md) - System design

## Testing

```bash
# All tests
pytest -q

# Specific test file
pytest backend/tests/test_edge_cases.py -v

# With coverage
pytest --cov=backend
```

## Current Status

- ✅ Phase 1: MCAR/MAR/MNAR + Validation
- ✅ Phase 2: Edge cases + Real data testing
- ✅ Tests: 18/18 passing
- 🚀 Staging: Ready for student testing
- 📅 Production: Coming Feb 1, 2026

## Support

- 📧 Email: support@academiclab.com
- 💬 Discord: https://discord.gg/academiclab
- 🐛 Issues: https://github.com/deltaxmodules/academiclabv2/issues
- 📚 Docs: https://docs.academiclab.com

## License

See [LICENSE](LICENSE)

## Contributors

- Backend Team: Phase 1 & 2 Implementation
- Jorge: Technical Leadership & Architecture

---

**Academic Lab: Making Data Scientists, One Dataset at a Time** 📊
