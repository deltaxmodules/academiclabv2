# Production Deployment Strategy

## Branch Model

```
Feature Development
  ↓
feature/feature-name → main
  ↓
Code Review + Tests
  ↓
Merge to main
  ↓
main → staging
  ↓
Student Testing (3-5 days)
  ↓
staging → production
  ↓
Production Deployment
```

## Branches

### feature/* (Development)
- Branch from: main
- Purpose: Develop new features/fixes
- Protection: None
- Delete after: Merge to main
- Example: `feature/p02-duplicates-detection`

### main (Integration)
- Source of truth
- Always stable
- All tests passing
- Ready for staging

**Rules:**
```
- Require PR review (1 approval)
- Require tests to pass
- Require up-to-date branch
- No force push
- No direct commits
```

### staging (Pre-production)
- Merged from: main
- Purpose: Student testing
- Duration: 3-5 days
- Testing: Real students, real data

**Rules:**
```
- Require PR review (1 approval)
- Require tests to pass
- Allow hotfixes from feature branches
- Can force push for critical hotfixes
- No direct commits
```

### production (Live)
- Merged from: staging
- Purpose: Live student data
- Duration: Until next release
- Monitoring: Performance + errors

**Rules:**
```
- Require PR review (2 approvals)
- Require tests to pass
- Require code owner approval
- No force push
- No direct commits
- Auto-deploy on merge
```

## Deployment Workflow

### 1. Feature Development
```bash
git checkout main
git pull origin main
git checkout -b feature/my-feature

# Make changes
git add ...
git commit -m "feat: description"
git push -u origin feature/my-feature

# Create PR on GitHub (from feature → main)
```

### 2. Code Review
- 1 approval required
- Tests must pass (GitHub Actions)
- Automated checks pass
- Then merge

### 3. To Staging
```bash
# After merge to main
git checkout staging
git pull origin staging
git merge main
git push origin staging

# Or via GitHub: Create PR main → staging, merge
```

### 4. Student Testing
- Announce to students
- Monitor for issues
- Collect feedback
- Fix critical bugs (hotfix branches)

### 5. To Production
```bash
# After 3-5 days in staging
git checkout production
git pull origin production
git merge staging
git push origin production

# Auto-deploy (GitHub Actions)
# Monitor uptime/performance
```

## Hotfix Workflow

Critical bug found in production?

```bash
git checkout -b hotfix/bug-description main
# OR
git checkout -b hotfix/bug-description staging

# Fix the bug
git add ...
git commit -m "fix: critical issue"
git push -u origin hotfix/bug-description

# Create PR
# After merge: redeploy
```

## Release Checklist

Before moving to production:

```
✅ All tests passing (18+ tests)
✅ No critical bugs reported
✅ Performance acceptable (< 2s response time)
✅ Student feedback collected
✅ Security review done
✅ Documentation updated
✅ .env variables ready
✅ Database backup ready
✅ Rollback plan prepared
✅ Monitoring configured
```

## Monitoring

### Performance
- Response time (target: < 2 seconds)
- CPU usage (target: < 70%)
- Memory usage (target: < 500MB)
- Database query time (target: < 1s)

### Errors
- Error rate (target: < 0.1%)
- Exception tracking (Sentry)
- Log aggregation (ELK/Datadog)
- Alert on critical errors

### User Metrics
- Active users
- Datasets analyzed
- Feature usage
- Student satisfaction

## Rollback Procedure

If critical issue in production:

```bash
# Option 1: Quick rollback
git checkout production
git revert [bad-commit]
git push origin production

# Option 2: Revert to previous stable
git checkout production
git reset --hard [previous-stable-commit]
git push -f origin production

# Option 3: Deploy from staging
git checkout production
git merge staging --force
git push -f origin production

# Monitor logs after rollback
journalctl -u academiclab-production -f
```

## Communication

### Before Deployment
```
Subject: [STAGING] Academic Lab Update Available

Hi students,

We've deployed Phase 2 improvements to staging!

New features:
- MCAR/MAR/MNAR detection
- Better decision trees
- Performance improvements

Test it: https://staging.academiclab.com
If you find issues, please report them.
```

### Before Production
```
Subject: [PRODUCTION] Academic Lab v2 Live! 🚀

Hi students,

Academic Lab v2 is now live!

Improvements:
- Automatic missing value classification
- Intelligent recommendations
- Better data validation

Access: https://academiclab.com

Questions? support@academiclab.com
```

## Metrics & Goals

| Metric | Current | Target | Timeline |
|--------|---------|--------|----------|
| Test Coverage | 18 tests | 30+ tests | Q2 2026 |
| Response Time | < 2s | < 1s | Q1 2026 |
| Uptime | 99% | 99.9% | Q2 2026 |
| Student Satisfaction | TBD | > 4/5 stars | Ongoing |

---

## Questions?

- Process: See CONTRIBUTING.md
- Architecture: See ARCHITECTURE.md
- Deployment: See STAGING_RUNBOOK.md, DEPLOYMENT.md
- Contact: support@academiclab.com
