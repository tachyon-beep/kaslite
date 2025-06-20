#!/usr/bin/env python3
"""
Git Ignore Validation and Maintenance Script

This script helps validate the .gitignore file and identify potential issues
with ignored/tracked files in the repository.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set


def run_git_command(command: List[str]) -> str:
    """Run a git command and return the output."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, cwd=Path(__file__).parent.parent)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Git command failed: {' '.join(command)}")
        print(f"Error: {e.stderr}")
        return ""


def get_tracked_files() -> Set[str]:
    """Get all files currently tracked by git."""
    output = run_git_command(["git", "ls-files"])
    return set(output.split("\n")) if output else set()


def get_ignored_files() -> Set[str]:
    """Get all files that are ignored by git."""
    output = run_git_command(["git", "status", "--ignored", "--porcelain"])
    ignored_files = set()
    for line in output.split("\n"):
        if line.startswith("!!"):
            ignored_files.add(line[3:])
    return ignored_files


def get_large_files(threshold_mb: float = 10.0) -> List[Dict[str, str]]:
    """Find large files in the repository."""
    large_files = []
    tracked_files = get_tracked_files()

    for file_path in tracked_files:
        full_path = Path(file_path)
        if full_path.exists() and full_path.is_file():
            size_mb = full_path.stat().st_size / (1024 * 1024)
            if size_mb > threshold_mb:
                large_files.append({"path": file_path, "size_mb": f"{size_mb:.2f}", "size_bytes": str(full_path.stat().st_size)})

    return sorted(large_files, key=lambda x: float(x["size_mb"]), reverse=True)


def check_sensitive_patterns() -> List[str]:
    """Check for potentially sensitive files that might be tracked."""
    sensitive_patterns = ["*.env", "*.key", "*.pem", "*secret*", "*password*", "*credential*", "*.p12", "*.jks"]

    sensitive_files = []
    tracked_files = get_tracked_files()

    for file_path in tracked_files:
        file_name = os.path.basename(file_path).lower()
        for pattern in sensitive_patterns:
            # Simple pattern matching (could be improved with fnmatch)
            if pattern.replace("*", "") in file_name:
                sensitive_files.append(file_path)
                break

    return sensitive_files


def check_common_artifacts() -> List[str]:
    """Check for common development artifacts that should be ignored."""
    artifact_patterns = ["__pycache__", ".pytest_cache", ".mypy_cache", ".coverage", "*.pyc", "*.pyo"]

    artifacts = []
    tracked_files = get_tracked_files()

    for file_path in tracked_files:
        # Skip legitimate test files and logger modules
        if file_path.startswith("tests/") or file_path.endswith("logger.py") or file_path.endswith("test_logger.md"):
            continue

        for pattern in artifact_patterns:
            pattern_clean = pattern.replace("*", "").replace(".", "")
            if pattern_clean in file_path and pattern_clean != "":
                artifacts.append(file_path)
                break

    return artifacts


def _check_large_files() -> List[Dict[str, str]]:
    """Check and report large files."""
    print("📊 Checking for large files (>10MB)...")
    large_files = get_large_files()
    if large_files:
        print("⚠️  Large files found:")
        for file_info in large_files[:10]:  # Show top 10
            print(f"  • {file_info['path']} ({file_info['size_mb']} MB)")
        if len(large_files) > 10:
            print(f"  ... and {len(large_files) - 10} more")
        print()
    else:
        print("✅ No large files found\n")
    return large_files


def _check_sensitive_files() -> List[str]:
    """Check and report sensitive files."""
    print("🔐 Checking for potentially sensitive files...")
    sensitive_files = check_sensitive_patterns()
    if sensitive_files:
        print("⚠️  Potentially sensitive files found:")
        for file_path in sensitive_files:
            print(f"  • {file_path}")
        print("❗ Review these files and ensure they don't contain secrets\n")
    else:
        print("✅ No obviously sensitive files found\n")
    return sensitive_files


def _check_artifacts() -> List[str]:
    """Check and report development artifacts."""
    print("🧹 Checking for development artifacts...")
    artifacts = check_common_artifacts()
    if artifacts:
        print("⚠️  Development artifacts found:")
        for artifact in artifacts[:10]:  # Show top 10
            print(f"  • {artifact}")
        if len(artifacts) > 10:
            print(f"  ... and {len(artifacts) - 10} more")
        print("💡 Consider adding these patterns to .gitignore\n")
    else:
        print("✅ No obvious development artifacts found\n")
    return artifacts


def validate_gitignore() -> bool:
    """Run comprehensive validation of the .gitignore configuration."""
    print("🔍 Validating .gitignore configuration...\n")

    large_files = _check_large_files()
    sensitive_files = _check_sensitive_files()
    artifacts = _check_artifacts()

    # Summary
    total_tracked = len(get_tracked_files())
    total_ignored = len(get_ignored_files())

    print("📋 Summary:")
    print(f"  • Tracked files: {total_tracked}")
    print(f"  • Ignored files: {total_ignored}")

    has_issues = bool(large_files or sensitive_files or artifacts)
    if has_issues:
        print("\n⚠️  Issues found - review the output above")
    else:
        print("\n✅ Repository looks clean!")

    return not has_issues


def show_ignored_status() -> None:
    """Show the current ignore status."""
    print("📁 Current ignore status:\n")

    # Show ignored files
    output = run_git_command(["git", "status", "--ignored", "--porcelain"])
    if output:
        ignored_count = 0
        for line in output.split("\n"):
            if line.startswith("!!"):
                if ignored_count == 0:
                    print("🚫 Ignored files/directories:")
                print(f"  • {line[3:]}")
                ignored_count += 1
                if ignored_count > 20:  # Limit output
                    remaining = len([l for l in output.split("\n") if l.startswith("!!")]) - 20
                    print(f"  ... and {remaining} more")
                    break
    else:
        print("✅ No ignored files found")


def _handle_validate_command() -> int:
    """Handle the validate command."""
    success = validate_gitignore()
    return 0 if success else 1


def _handle_status_command() -> None:
    """Handle the status command."""
    show_ignored_status()


def _handle_large_command() -> None:
    """Handle the large files command."""
    large_files = get_large_files()
    if large_files:
        print("📊 Large files in repository:")
        for file_info in large_files:
            print(f"  • {file_info['path']} ({file_info['size_mb']} MB)")
    else:
        print("✅ No large files found")


def _show_usage() -> None:
    """Show usage information."""
    print("Git Ignore Validation Tool")
    print("\nUsage:")
    print("  python scripts/validate_gitignore.py validate  # Full validation")
    print("  python scripts/validate_gitignore.py status   # Show ignore status")
    print("  python scripts/validate_gitignore.py large    # Find large files")


def main() -> None:
    """Main function."""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "validate":
            sys.exit(_handle_validate_command())
        elif command == "status":
            _handle_status_command()
        elif command == "large":
            _handle_large_command()
        else:
            print(f"Unknown command: {command}")
            sys.exit(1)
    else:
        _show_usage()


if __name__ == "__main__":
    main()
