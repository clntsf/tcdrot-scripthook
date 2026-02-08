import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Install dependencies
requirements_path = os.path.join(REPO_ROOT, "requirements.txt")
if os.path.isfile(requirements_path):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
else:
    print("No requirements.txt found, skipping dependency install.")

# Append rot() shell function to ~/.zshrc
zshrc_path = os.path.expanduser("~/.zshrc")
block = f"""
ROTPATH={REPO_ROOT}
rot() {{
  python3 $ROTPATH/src/"$1".py "${{@:2}}"
}}
"""

marker = "# >>> tcdrot-scripthook >>>"
end_marker = "# <<< tcdrot-scripthook <<<"

# Read existing zshrc content
existing = ""
if os.path.isfile(zshrc_path):
    with open(zshrc_path, "r") as f:
        existing = f.read()

# Replace existing block or append new one
if marker in existing:
    start = existing.index(marker)
    end = existing.index(end_marker) + len(end_marker)
    existing = existing[:start] + marker + block + end_marker + existing[end:]
    with open(zshrc_path, "w") as f:
        f.write(existing)
    print("Updated existing rot() block in ~/.zshrc")
else:
    with open(zshrc_path, "a") as f:
        f.write(f"\n{marker}{block}{end_marker}\n")
    print("Appended rot() block to ~/.zshrc")

print("Done. Run `source ~/.zshrc` or open a new terminal to use `rot`.")
