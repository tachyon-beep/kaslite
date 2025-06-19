#!/bin/bash

# Git Hooks Installation Script
# This script sets up git hooks for repository validation

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOOKS_DIR="$REPO_ROOT/.git/hooks"

echo "🔧 Setting up git hooks for Kaslite..."

# Ensure hooks directory exists
mkdir -p "$HOOKS_DIR"

# Install pre-commit hook
PRE_COMMIT_HOOK="$HOOKS_DIR/pre-commit"

cat > "$PRE_COMMIT_HOOK" << 'EOF'
#!/bin/bash
# Auto-generated pre-commit hook for Kaslite
# Runs validation checks before allowing commits

exec python3 "$(git rev-parse --show-toplevel)/scripts/pre_commit_hook.py"
EOF

chmod +x "$PRE_COMMIT_HOOK"

echo "✅ Pre-commit hook installed successfully!"
echo ""
echo "🎯 The hook will now:"
echo "   • Check for large files (>50MB)"
echo "   • Scan for potential credentials/secrets"
echo "   • Detect development artifacts"
echo ""
echo "💡 To skip hooks (not recommended): git commit --no-verify"
echo "💡 To uninstall hooks: rm .git/hooks/pre-commit"
echo ""
echo "🧪 Test the installation:"
echo "   python3 scripts/pre_commit_hook.py"
