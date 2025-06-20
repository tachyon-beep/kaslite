#!/usr/bin/env python3
"""
Pre-commit hook for repository validation.

This script runs basic checks before allowing commits to prevent
common issues with large files, credentials, and development artifacts.
"""

import subprocess
import sys
from pathlib import Path
from typing import Callable


def run_command(command: list[str]) -> tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, cwd=Path(__file__).parent.parent)
        return result.returncode, result.stdout, result.stderr
    except subprocess.SubprocessError as e:
        return 1, "", str(e)


def check_file_sizes(max_size_mb: float = 50.0) -> bool:
    """Check for large files being committed."""
    # Get staged files
    code, output, _ = run_command(["git", "diff", "--cached", "--name-only"])
    if code != 0:
        return True  # No staged files or git error

    staged_files = [f.strip() for f in output.split("\n") if f.strip()]
    large_files = []

    for file_path in staged_files:
        full_path = Path(file_path)
        if full_path.exists() and full_path.is_file():
            size_mb = full_path.stat().st_size / (1024 * 1024)
            if size_mb > max_size_mb:
                large_files.append((file_path, size_mb))

    if large_files:
        print("❌ Large files detected in commit:")
        for file_path, size_mb in large_files:
            print(f"  • {file_path} ({size_mb:.2f} MB)")
        print(f"\n💡 Files larger than {max_size_mb}MB should not be committed to git.")
        print("   Consider using git-lfs or excluding these files.")
        return False

    return True


def _check_line_for_sensitive_patterns(line: str, patterns: list[str]) -> bool:
    """Check if a line contains sensitive patterns."""
    if not line.startswith("+") or line.startswith("+++"):
        return False

    line_lower = line.lower()
    for pattern in patterns:
        if pattern in line_lower and "=" in line_lower:
            return True
    return False


def check_sensitive_content() -> bool:
    """Check for potentially sensitive content."""
    # Get staged file content
    code, output, _ = run_command(["git", "diff", "--cached"])
    if code != 0:
        return True

    sensitive_patterns = ["password", "secret", "api_key", "private_key", "token", "credential"]

    lines = output.split("\n")
    issues = []

    for i, line in enumerate(lines):
        if _check_line_for_sensitive_patterns(line, sensitive_patterns):
            issues.append((i + 1, line.strip()))

    if issues:
        print("❌ Potentially sensitive content detected:")
        for line_num, content in issues[:5]:  # Show first 5
            print(f"  • Line {line_num}: {content[:80]}...")
        print("\n💡 Review these changes for sensitive information.")
        print("   Use environment variables or config files for secrets.")
        return False

    return True


def check_development_artifacts() -> bool:
    """Check for development artifacts being committed."""
    code, output, _ = run_command(["git", "diff", "--cached", "--name-only"])
    if code != 0:
        return True

    staged_files = [f.strip() for f in output.split("\n") if f.strip()]
    artifacts = []

    artifact_patterns = ["__pycache__", ".pyc", ".pyo", ".coverage", ".pytest_cache"]

    for file_path in staged_files:
        for pattern in artifact_patterns:
            if pattern in file_path:
                artifacts.append(file_path)
                break

    if artifacts:
        print("❌ Development artifacts detected in commit:")
        for artifact in artifacts:
            print(f"  • {artifact}")
        print("\n💡 These files should be in .gitignore")
        return False

    return True


def main() -> int:
    """Main pre-commit validation."""
    print("🔍 Running pre-commit validation...")

    checks: list[tuple[str, Callable[[], bool]]] = [
        ("File sizes", check_file_sizes),
        ("Sensitive content", check_sensitive_content),
        ("Development artifacts", check_development_artifacts),
    ]

    all_passed = True

    for _, check_func in checks:
        if not check_func():
            all_passed = False

    if all_passed:
        print("✅ All pre-commit checks passed!")
        return 0
    else:
        print("\n❌ Pre-commit validation failed!")
        print("💡 Fix the issues above and try committing again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
