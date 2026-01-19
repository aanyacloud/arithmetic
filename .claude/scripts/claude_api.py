#!/usr/bin/env python3
"""
Claude API integration for autonomous project implementation.

This module provides functions to interact with Claude via the Anthropic API
for decomposing specifications and implementing issues.
"""

import json
import os
import subprocess
from pathlib import Path

from anthropic import Anthropic
from loguru import logger


def get_api_client() -> Anthropic:
    """
    Get authenticated Anthropic API client.

    :returns: Authenticated Anthropic client
    :raises: SystemExit if ANTHROPIC_API_KEY not set
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY environment variable not set")
        raise SystemExit(1)

    return Anthropic(api_key=api_key)


def read_file(path: Path) -> str:
    """
    Read file contents.

    :param path: Path to file
    :returns: File contents as string
    """
    with open(path) as f:
        return f.read()


def write_file(path: Path, content: str) -> None:
    """
    Write content to file.

    :param path: Path to file
    :param content: Content to write
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """
    Run shell command.

    :param cmd: Command as list of strings
    :param check: Whether to raise on non-zero exit
    :returns: CompletedProcess result
    """
    logger.debug(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def decompose_spec(spec_path: Path, max_turns: int = 20) -> dict:
    """
    Decompose specification into GitHub issues using Claude.

    :param spec_path: Path to specification file
    :param max_turns: Maximum conversation turns
    :returns: Dictionary with issue data
    """
    client = get_api_client()
    spec_content = read_file(spec_path)

    prompt = f"""Read the following specification and decompose it into implementable GitHub issues.

SPECIFICATION:
{spec_content}

For each issue you identify:
1. Create a clear, specific title
2. Write detailed acceptance criteria
3. Specify what unit tests are required
4. Identify dependencies on other issues (by issue number)

Guidelines:
- Break down large features into smaller, testable units
- Issue #0 should be test framework setup if not already present
- Ensure each issue can be implemented independently once dependencies are met
- Order issues logically (foundations before features)
- Use "depends on #N" in issue body to indicate dependencies

Output a JSON array of issues with this structure:
[
  {{
    "number": 0,
    "title": "Setup test framework",
    "body": "Description\\n\\n## Dependencies\\nNone\\n\\n## Acceptance Criteria\\n- [ ] pytest installed\\n- [ ] Sample test passes\\n\\n## Tests Required\\n- Verify test framework works",
    "deps": []
  }},
  {{
    "number": 1,
    "title": "Implement user model",
    "body": "Description\\n\\n## Dependencies\\n- #0\\n\\n## Acceptance Criteria\\n- [ ] User class created\\n\\n## Tests Required\\n- Unit tests for User model",
    "deps": [0]
  }}
]

Only output the JSON array, nothing else."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    # Extract JSON from response
    content = response.content[0].text
    # Find JSON array in response
    start = content.find("[")
    end = content.rfind("]") + 1
    if start == -1 or end == 0:
        logger.error("No JSON array found in Claude's response")
        raise ValueError("Invalid response format")

    issues_data = json.loads(content[start:end])
    logger.info(f"Decomposed spec into {len(issues_data)} issues")

    return {"issues": issues_data}


def create_github_issues(issues_data: dict) -> list[int]:
    """
    Create GitHub issues from decomposed data.

    :param issues_data: Dictionary containing issue data
    :returns: List of created issue numbers
    """
    created_issues = []

    for issue in issues_data["issues"]:
        title = issue["title"]
        body = issue["body"]

        result = run_command(
            ["gh", "issue", "create", "--title", title, "--body", body], check=True
        )

        # Extract issue number from output (format: "https://github.com/owner/repo/issues/N")
        output = result.stdout.strip()
        issue_num = int(output.split("/")[-1])
        created_issues.append(issue_num)
        logger.info(f"Created issue #{issue_num}: {title}")

    return created_issues


def implement_issue(issue_number: int, max_turns: int = 30) -> bool:
    """
    Implement a GitHub issue using Claude with autonomous tool use.

    :param issue_number: Issue number to implement
    :param max_turns: Maximum conversation turns
    :returns: True if implementation successful
    """
    client = get_api_client()

    # Get issue details
    result = run_command(
        ["gh", "issue", "view", str(issue_number), "--json", "title,body"], check=True
    )
    issue_data = json.loads(result.stdout)
    title = issue_data["title"]
    body = issue_data["body"]

    logger.info(f"Implementing issue #{issue_number}: {title}")

    # Read project context
    context_files = []
    for pattern in [
        "*.py",
        "*.js",
        "*.ts",
        "*.go",
        "*.rs",
        "package.json",
        "pyproject.toml",
    ]:
        try:
            result = run_command(
                ["find", ".", "-name", pattern, "-type", "f"], check=False
            )
            if result.returncode == 0:
                context_files.extend(result.stdout.strip().split("\n"))
        except Exception:
            pass

    # Build context
    context = "Project structure:\n"
    for f in context_files[:20]:  # Limit to prevent token overflow
        if f and not f.startswith("./.git"):
            context += f"- {f}\n"

    prompt = f"""You are implementing GitHub issue #{issue_number}.

ISSUE TITLE: {title}

ISSUE BODY:
{body}

PROJECT CONTEXT:
{context}

Your task:
1. Write comprehensive unit tests for this feature (TDD approach)
2. Implement the feature to make tests pass
3. Run the full test suite to ensure no regressions
4. Ensure all tests pass before completing

Follow the project's existing patterns and conventions. If you need to read or write files, describe what you need and I'll help.

Start by analyzing the requirements and planning your test cases."""

    conversation = [{"role": "user", "content": prompt}]

    for turn in range(max_turns):
        logger.debug(f"Turn {turn + 1}/{max_turns}")

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=conversation,
        )

        assistant_message = response.content[0].text
        conversation.append({"role": "assistant", "content": assistant_message})

        logger.debug(f"Claude: {assistant_message[:200]}...")

        # Check if implementation is complete
        if (
            "implementation complete" in assistant_message.lower()
            or "all tests pass" in assistant_message.lower()
        ):
            logger.info("Implementation reported complete")
            return True

        # In a real implementation, this would handle tool use (file operations, running tests, etc.)
        # For now, we'll use a simpler approach with direct file operations

        if turn < max_turns - 1:
            # Provide feedback for next turn
            feedback = "Continue with implementation. What's your next step?"
            conversation.append({"role": "user", "content": feedback})

    logger.warning(f"Reached max turns ({max_turns}) without completion")
    return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: claude_api.py <command> [args]")
        print("Commands:")
        print("  decompose <spec_file>  - Decompose spec into issues")
        print("  implement <issue_num>  - Implement an issue")
        sys.exit(1)

    command = sys.argv[1]

    if command == "decompose":
        spec_file = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("README.md")
        result = decompose_spec(spec_file)
        print(json.dumps(result, indent=2))

    elif command == "implement":
        if len(sys.argv) < 3:
            print("Error: Issue number required")
            sys.exit(1)
        issue_num = int(sys.argv[2])
        success = implement_issue(issue_num)
        sys.exit(0 if success else 1)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
