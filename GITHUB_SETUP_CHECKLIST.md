# GitHub Setup Checklist

## Repository
- [ ] Repository created
- [ ] Access permissions set

## Branches
- [ ] `main`
- [ ] `staging`
- [ ] `feature/phase-2-edge-cases`
- [ ] `production` (optional)

## Branch Protection
- [ ] main: PR required + tests required
- [ ] staging: merge from main only
- [ ] production: merge from staging only

## Secrets
- [ ] OPENAI_API_KEY
- [ ] DATABASE_URL
- [ ] REDIS_URL (if used)

## CI/CD
- [ ] Run tests on PR
- [ ] Deploy to staging on merge to main
- [ ] Deploy to production on merge to staging
