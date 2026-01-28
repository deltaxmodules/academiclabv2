# Deployment Guide

## Pre-deployment Checklist
- [ ] All tests passing (18/18)
- [ ] Code reviewed
- [ ] Performance benchmarks OK
- [ ] Edge cases covered

## Staging Deployment

### Prerequisites
- Python 3.12+
- Environment variables configured
- Database (if required)
- Redis (if required)

### Steps
1. Clone repository
   ```
   git clone <URL>
   git checkout staging
   ```

2. Create virtual environment
   ```
   python3.12 -m venv .venv312
   source .venv312/bin/activate
   ```

3. Install dependencies
   ```
   pip install -r backend/requirements.txt
   ```

4. Run tests
   ```
   backend/.venv312/bin/pytest -q
   ```

5. Configure environment
   - Copy `backend/.env.example` → `backend/.env`
   - Set required values

6. Start backend
   ```
   python backend/main.py
   ```

7. Start frontend
   ```
   cd frontend
   npm install
   npm run dev
   ```

### Health Check
- `GET /health` should return `ok`

## Rollback
- Revert to previous commit and redeploy
