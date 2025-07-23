#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Project root (run inside your cloned repo)
echo "Setting up project structure..."

# Create main folders
mkdir -p model/src
mkdir -p web/src/templates
mkdir -p web/src/static

# Create core files
touch docker-compose.yml

# Model service files
touch model/Dockerfile
touch model/requirements.txt
touch model/src/model.ipynb

# Web service files
touch web/Dockerfile
touch web/requirements.txt
touch web/src/app.py
touch web/src/templates/index.html
touch web/src/static/style.css

# Create .gitignore
cat <<EOL > .gitignore
__pycache__/
*.py[cod]
*$py.class

.ipynb_checkpoints/
.vscode/

.env
*.env

venv/
env/
.venv/
ENV/
Pipfile.lock

*.csv
*.tsv
*.xlsx
*.xls
*.db
*.sqlite

*.pkl
*.joblib
*.h5

*.log

*.pid
*.tar
*.sock
docker-compose.override.yml

.DS_Store
Thumbs.db
ehthumbs.db
desktop.ini

.cache/
.mypy_cache/
~$*

build/
dist/
*.egg-info/
EOL

echo "Project setup complete! 🎉"
