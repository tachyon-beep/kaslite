# Repository Management & Git Strategy

This document outlines the comprehensive repository management strategy implemented for Kaslite, focusing on maintaining a clean, secure, and efficient development environment.

## 🎯 Overview

The Kaslite project implements a sophisticated `.gitignore` strategy and repository validation system designed specifically for machine learning and AI development workflows. This approach addresses common issues such as:

- **Large file management** (datasets, models, checkpoints)
- **Security** (preventing credential leaks)
- **Development artifacts** (cache files, temporary files)
- **Cross-platform compatibility** (OS-specific files)
- **Collaborative development** (IDE-specific configurations)

## 📁 Enhanced .gitignore Structure

The `.gitignore` file is organized into logical sections:

### Core Categories
- **Python & Dependencies**: Bytecode, distributions, virtual environments
- **Development Tools**: IDE configs, linting caches, type checking artifacts
- **Operating Systems**: macOS, Windows, Linux specific files
- **Machine Learning**: Datasets, models, experiment tracking, MLOps tools
- **Security**: Environment files, credentials, certificates
- **Performance**: Profiling outputs, monitoring data

### Key Features
- **Comprehensive ML coverage**: Supports MLflow, Weights & Biases, TensorBoard, DVC
- **Security-first approach**: Prevents accidental credential commits
- **Development-friendly**: Excludes artifacts while preserving essential configs
- **Extensible design**: Easy to add new tools and frameworks

## 🛠️ Validation Tools

### 1. Repository Validator (`scripts/validate_gitignore.py`)

A comprehensive validation tool that checks for:

```bash
# Run full validation
python scripts/validate_gitignore.py validate

# Check ignore status
python scripts/validate_gitignore.py status

# Find large files
python scripts/validate_gitignore.py large
```

**Features:**
- **Large file detection** (configurable threshold)
- **Sensitive content scanning** (API keys, passwords, etc.)
- **Development artifact identification**
- **Repository statistics and health metrics**

### 2. Pre-commit Hook (`scripts/pre_commit_hook.py`)

Automated validation that runs before each commit:

```bash
# Install the pre-commit hook
./scripts/setup_git_hooks.sh

# Test manually
python scripts/pre_commit_hook.py
```

**Prevents commits containing:**
- Files larger than 50MB
- Potential credentials or secrets
- Development artifacts that should be ignored

## 📋 Configuration Management

### Environment Variables
- **Template**: `.env.example` provides a comprehensive template
- **Security**: Actual `.env` files are ignored by git
- **Documentation**: All variables are documented with examples

### Local Overrides
```bash
# Copy template and customize
cp .env.example .env
# Edit with your local settings
nano .env
```

## 🔄 Workflows

### Daily Development
1. **Clone repository**: Clean checkout with no artifacts
2. **Setup environment**: Copy `.env.example` to `.env` and configure
3. **Install hooks**: Run `./scripts/setup_git_hooks.sh`
4. **Develop safely**: Hooks prevent accidental commits of sensitive data

### Periodic Maintenance
```bash
# Weekly repository health check
python scripts/validate_gitignore.py validate

# Review ignored files
python scripts/validate_gitignore.py status

# Check for large files
python scripts/validate_gitignore.py large
```

### Team Onboarding
1. **Documentation**: Read this guide and `docs/git_ignore_strategy.md`
2. **Setup**: Install git hooks and configure environment
3. **Validation**: Run repository validator to ensure clean state

## 🚀 Best Practices

### For Developers
- **Always use environment templates** for local configuration
- **Never commit credentials** - use environment variables or secret managers
- **Review large files** before adding to repository
- **Run validation tools** regularly to maintain repository health

### For Data Scientists
- **Use DVC or similar tools** for dataset version control
- **Store model artifacts** in MLflow registry or similar systems
- **Keep datasets external** to the main repository
- **Document data sources** and preprocessing steps

### For DevOps/MLOps
- **Leverage CI/CD integration** with validation tools
- **Monitor repository size** and performance metrics
- **Implement backup strategies** for critical configurations
- **Maintain security scanning** in deployment pipelines

## 🔧 Troubleshooting

### Common Issues

#### Large Repository Size
```bash
# Find large files
python scripts/validate_gitignore.py large

# Clean up if needed (DESTRUCTIVE - backup first!)
git filter-branch --tree-filter 'rm -rf path/to/large/files' HEAD
```

#### Credential Leak
1. **Immediately revoke** the exposed credentials
2. **Remove from git history** using BFG Repo-Cleaner or git filter-branch
3. **Update team** and force re-clone if necessary
4. **Review and improve** prevention measures

#### Development Artifacts
```bash
# Clean common artifacts
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete
find . -name ".pytest_cache" -type d -exec rm -rf {} +
```

### Recovery Procedures

#### If Hooks Were Bypassed
```bash
# Re-validate entire repository
python scripts/validate_gitignore.py validate

# Reinstall hooks
./scripts/setup_git_hooks.sh
```

#### If .gitignore Becomes Corrupted
```bash
# Backup current version
cp .gitignore .gitignore.backup

# Restore from this documentation or git history
git checkout HEAD -- .gitignore
```

## 📊 Metrics and Monitoring

### Repository Health Indicators
- **File count**: Monitor tracked vs ignored file ratios
- **Repository size**: Keep under reasonable limits
- **Security scans**: Regular credential pattern detection
- **Performance**: Clone and operation speed

### Automated Checks
- **Pre-commit validation**: Every commit
- **CI/CD integration**: Every pull request
- **Scheduled scans**: Weekly repository health checks
- **Security audits**: Quarterly comprehensive reviews

## 🔮 Future Enhancements

### Planned Improvements
- **Integration with git-lfs** for large file management
- **Advanced security scanning** with specialized tools
- **Custom hook configurations** per development workflow
- **Automated cleanup** and optimization scripts

### Extension Points
- **New ML frameworks**: Easy addition of new ignore patterns
- **Team-specific rules**: Customizable validation criteria
- **Integration APIs**: Webhook support for external tools
- **Reporting dashboards**: Visual repository health metrics

## 📚 Related Documentation

- [`docs/git_ignore_strategy.md`](git_ignore_strategy.md) - Detailed strategy explanation
- [`.env.example`](../.env.example) - Environment configuration template
- [`scripts/validate_gitignore.py`](../scripts/validate_gitignore.py) - Validation tool
- [`scripts/pre_commit_hook.py`](../scripts/pre_commit_hook.py) - Pre-commit hook
- [`scripts/setup_git_hooks.sh`](../scripts/setup_git_hooks.sh) - Hook installation

This comprehensive approach ensures that the Kaslite repository remains clean, secure, and performant throughout its development lifecycle while supporting the complex requirements of modern ML/AI projects.
