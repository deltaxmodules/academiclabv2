# Contributing to Academic Lab

Thank you for wanting to contribute! 🙏

## Code of Conduct

- Be respectful and constructive
- Help make this a welcoming space
- Report issues professionally

## How to Contribute

### 1. Report a Bug
- Go to [Issues](https://github.com/deltaxmodules/academiclabv2/issues)
- Click "New Issue"
- Describe: What happened, Expected, Actual, Steps to reproduce
- Include: Python version, OS, Error message

### 2. Suggest a Feature
- Open Issue with title "Feature: ..."
- Describe: What problem it solves, How it works, Benefits
- Include: Example use case

### 3. Submit Code

#### Setup Development Environment
```bash
git clone https://github.com/deltaxmodules/academiclabv2.git
cd academiclabv2
git checkout -b feature/your-feature-name

python -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt
```

#### Make Changes
- Keep commits small and focused
- Write clear commit messages
- Add/update tests for changes
- Follow existing code style

#### Run Tests Before Pushing
```bash
pytest -q
# Must pass before push!
```

#### Push and Create PR
```bash
git push -u origin feature/your-feature-name
```

Then on GitHub:
- Go to repo
- Click "Compare & pull request"
- Describe changes clearly
- Wait for review

### 4. Code Review Process

1. **Submit PR** → GitHub Actions runs tests
2. **Tests Pass** → Code review starts
3. **Feedback** → Address comments
4. **Approval** → Merge to main
5. **Deploy** → Auto-deploy to staging

## Coding Standards

### Python
- Use PEP 8 style guide
- Type hints for functions
- Docstrings for classes/methods
- Max line length: 100 characters

### Tests
- Write tests for new code
- Maintain > 80% coverage
- Use descriptive test names
- Run `pytest -q` before push

### Commits
```
Format: "Type: Description"

Types:
  - feat: New feature
  - fix: Bug fix
  - test: Add/update tests
  - docs: Documentation
  - style: Code style (no logic change)
  - refactor: Code refactor
  - perf: Performance improvement
  - ci: CI/CD changes

Examples:
  feat: Add MCAR/MAR/MNAR detection
  fix: Handle empty datasets correctly
  test: Add edge case tests
  docs: Update README with examples
```

## Questions?

- Docs: https://docs.academiclab.com
- Discord: https://discord.gg/academiclab
- Email: support@academiclab.com

---

**Happy contributing!** 🚀
