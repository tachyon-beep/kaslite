# Git Ignore Strategy and Best Practices

This document explains the comprehensive `.gitignore` strategy implemented for the Kaslite project and provides guidelines for maintaining clean version control.

## Overview

The `.gitignore` file has been structured to address common issues in machine learning and AI projects, organized into logical sections for easy maintenance and understanding.

## Structure and Rationale

### 1. Python Core Files
```
__pycache__/
*.py[cod]
*.so
build/
dist/
*.egg-info/
```
**Why**: Python compilation artifacts and distribution files should never be committed as they're platform-specific and automatically generated.

### 2. Virtual Environments & Package Managers
```
env/
venv/
.venv/
poetry.lock
Pipfile.lock
.conda/
```
**Why**: Environment directories can be huge and are machine-specific. Lock files may contain absolute paths or platform-specific dependencies.

### 3. Development Tools & IDEs
```
.vscode/
.idea/
.ruff_cache/
.mypy_cache/
```
**Why**: IDE configurations are personal preferences and tool caches can be regenerated.

### 4. Machine Learning Specific Files
```
data/
models/
checkpoints/
mlruns/
wandb/
*.pt
*.pkl
```
**Why**: ML projects generate large data files, model artifacts, and experiment logs that are typically too large for git and can be regenerated.

### 5. Security & Secrets
```
secrets/
*.key
*.pem
credentials.json
.env
```
**Why**: Prevents accidental commit of sensitive information like API keys, certificates, and credentials.

## Best Practices

### 1. Use Environment Templates
- Commit `.env.example` with example configurations
- Never commit actual `.env` files with real credentials
- Document all required environment variables

### 2. Preserve Important Directories
- Use `.gitkeep` files to preserve empty directories that are needed by the application
- Exclude the contents but keep the structure

### 3. Be Selective with Data
- Generally exclude raw datasets (they can be large and change frequently)
- Consider committing small sample datasets for testing
- Use tools like DVC for data version control

### 4. Model Artifacts Strategy
- Exclude model checkpoints and weights from git
- Use model registries (MLflow, Weights & Biases) for model versioning
- Commit model configurations and hyperparameters

### 5. Documentation Exceptions
- Include essential documentation files
- Exclude generated documentation (build artifacts)
- Use `!filename` patterns to include specific files that would otherwise be ignored

## Common Pitfalls and Solutions

### Problem: Large Files Accidentally Committed
**Solution**: Use `git-lfs` for large files that must be tracked, or exclude them entirely.

### Problem: Credentials Leaked
**Solution**: 
1. Immediately revoke compromised credentials
2. Use tools like `git-secrets` to prevent future leaks
3. Follow the principle of least privilege

### Problem: Platform-Specific Files
**Solution**: Include comprehensive OS-specific patterns (macOS .DS_Store, Windows Thumbs.db, Linux .directory).

### Problem: Development vs Production Configs
**Solution**: Use environment-specific configuration files and templates.

## Maintenance Guidelines

### Regular Reviews
- Review `.gitignore` during major dependency updates
- Add new patterns when introducing new tools or frameworks
- Remove obsolete patterns when tools are deprecated

### Team Collaboration
- Document any project-specific ignore patterns
- Use comments to explain non-obvious patterns
- Keep the file organized and well-structured

### Tool Integration
- Configure development tools to respect `.gitignore`
- Use pre-commit hooks to enforce ignore patterns
- Integrate with CI/CD pipelines for validation

## Extension for New Components

When adding new tools or frameworks to the project:

1. **Research the tool's artifact patterns**
2. **Add appropriate ignore patterns**
3. **Document the reasoning**
4. **Test with a representative workflow**

### Example: Adding a New ML Framework
```bash
# Framework artifacts
new_framework_cache/
*.new_extension
new_framework_logs/

# Framework-specific models
*.new_model_format
```

## Monitoring and Validation

### Regular Checks
```bash
# Check for accidentally tracked large files
git ls-files | xargs ls -lah | sort -k5 -h | tail -10

# Check for potential credential patterns
git log --all --full-history --source -- "**/*.env" "**/*secret*" "**/*key*"

# Validate ignore patterns work
git status --ignored
```

### Automated Validation
Consider adding pre-commit hooks to:
- Prevent large file commits
- Scan for credential patterns
- Validate ignore file syntax

## Recovery Procedures

### If Large Files Were Committed
```bash
# Use BFG repo cleaner or git filter-branch
git filter-branch --tree-filter 'rm -rf path/to/large/files' HEAD
```

### If Credentials Were Committed
1. **Immediately revoke the credentials**
2. **Remove from history using BFG or filter-branch**
3. **Force push to update all remotes**
4. **Notify team members to re-clone**

## Integration with Project Tools

### DVC Integration
The `.gitignore` is configured to work seamlessly with DVC (Data Version Control):
- Excludes DVC cache but tracks `.dvc` files
- Preserves data directory structure while ignoring contents

### Docker Integration
- Excludes Docker build context files
- Preserves docker-compose files for reproducibility
- Ignores container-specific artifacts

### CI/CD Integration
- Excludes build artifacts while preserving configuration
- Compatible with containerized workflows
- Supports both local and cloud-based CI systems

This comprehensive approach ensures that the repository remains clean, secure, and performant while supporting the full ML development lifecycle.
