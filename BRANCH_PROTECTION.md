# Branch Protection Setup Checklist

## Why Branch Protection?
- Ensures code quality
- Prevents accidental pushes to main/staging
- Requires reviews before merge
- Maintains production stability

## Setup Instructions

### For `main` branch (Production-ready code)

Go to: GitHub → Settings → Branches → Branch protection rules → Add rule

```
Branch name pattern: main

☑ Require a pull request before merging
  - Required number of approvals: 1
  - Require review from code owners: No (set if you have CODEOWNERS file)
  - Dismiss stale pull request approvals: Yes
  - Require approval of reviews before merging: No

☑ Require status checks to pass before merging
  - Require branches to be up to date before merging: Yes
  - Required status checks:
    - tests (GitHub Actions)

☑ Restrict who can push to matching branches
  - Include administrators: Yes

☑ Allow force pushes: No
☑ Allow deletions: No
```

### For `staging` branch (Pre-production testing)

```
Branch name pattern: staging

☑ Require a pull request before merging
  - Required number of approvals: 1

☑ Require status checks to pass
  - tests (GitHub Actions)

☑ Restrict pushes to matching branches
  - Allow only: Maintainers/Admins

☑ Allow force pushes: No
☑ Allow deletions: No
```

### For `feature/*` branches (Development features)

```
Branch name pattern: feature/*

☑ Require a pull request before merging
  - Required number of approvals: 1

☑ Require status checks: No (optional during dev)

☑ Allow force pushes: Yes (while developing)
☑ Allow deletions: Yes
```

## Verify It's Working

After setup:
1. Try to push directly to `main` - should be blocked ❌
2. Try to merge PR without approval - should be blocked ❌
3. Merge with approval + tests passing - should work ✅

## If You Need Help
See GitHub docs: https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/managing-a-branch-protection-rule
