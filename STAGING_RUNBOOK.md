# Staging Deployment Runbook

## Quick Start (5 minutes)

### 1. Clone & Setup
```bash
git clone https://github.com/deltaxmodules/academiclabv2.git
cd academiclabv2
git checkout staging

python -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with staging values:
#   DATABASE_URL=postgresql://user:pass@staging-db:5432/academiclab
#   API_URL=https://staging-api.academiclab.com
#   ENVIRONMENT=staging
#   DEBUG=false
```

### 3. Run Tests
```bash
pytest -q
# Expected: 18 passed ✅
```

### 4. Start Server
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
# Access: http://localhost:8000
```

## Deployment Options

### Option A: Docker (Recommended)
```bash
docker build -t academiclab:staging .
docker run -p 8000:8000 --env-file .env academiclab:staging
```

### Option B: systemd Service
```bash
sudo cp academiclab.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable academiclab-staging
sudo systemctl start academiclab-staging
sudo systemctl status academiclab-staging
```

### Option C: Cloud Platform
```bash
# Heroku
git push heroku staging:main

# Railway / Render
# Use GitHub integration → deploy from staging branch
```

## Verify Deployment

```bash
# Health check
curl https://staging-api.academiclab.com/health
# Expected: {"status": "ok"}

# Test API
curl https://staging-api.academiclab.com/session/test123
# Expected: Session data with checklist_report

# Monitor logs
docker logs -f academiclab
# OR
journalctl -u academiclab-staging -f
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Port 8000 already in use | `lsof -i :8000` then kill process |
| Database connection error | Check DATABASE_URL in .env |
| Tests failing | Run `pytest -v` for details |
| Performance slow | Check system resources (CPU, memory) |

## Rollback

```bash
# If something breaks
git checkout main  # revert to main
# Or redeploy previous working commit
git checkout [previous-commit-hash]
```

## Support
- Logs: `/var/log/academiclab/`
- Health: `https://staging-api.academiclab.com/health`
- Issues: https://github.com/deltaxmodules/academiclabv2/issues
