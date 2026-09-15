from pathlib import Path

project_name = ""

directories = [
    "app",
    "data/raw",
    "data/processed",
    "models"
]

files = [
    "app/__init__.py",
    "app/main.py",
    "app/ingest.py",
    "app/validation.py",
    "app/preprocessing.py",
    "app/features.py",
    "app/train.py",
    "app/evaluate.py",
    "app/predict.py",
    "requirements.txt",
    "Dockerfile",
    ".dockerignore",
    ".gitignore",
    "README.md"
]

root = Path(project_name)

root.mkdir(exist_ok=True)

for directory in directories:
    (root / directory).mkdir(parents=True, exist_ok=True)

for file in files:
    path = root / file
    path.touch(exist_ok=True)

print(f"Project created: {root.resolve()}")

print("\nProject structure:")
for path in sorted(root.rglob("*")):
    if path.is_dir():
        print(f"[DIR]  {path.relative_to(root)}")
    else:
        print(f"[FILE] {path.relative_to(root)}")