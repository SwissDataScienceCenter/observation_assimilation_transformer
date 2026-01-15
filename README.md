# Python Project Template

![Python Version](https://img.shields.io/badge/python-3.12-blue)
![Black](https://img.shields.io/badge/black-24.0-000000)
![Ruff](https://img.shields.io/badge/ruff-0.4.0-red)
![mypy](https://img.shields.io/badge/mypy-1.8.0-blue)

A production-ready Python project template with Dev Container support, pre-configured with:

- 🛠️ Black for code formatting
- 🔍 Ruff for linting
- ✅ mypy for static type checking
- 💡 Pylance for intelligent code completion
- 🐳 Docker-based development environment

## Table of Contents
- [Features](#features)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Tool Configuration](#tool-configuration)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)

## Features

### Core Components
- Python 3.12 base environment
- VS Code Dev Container configuration
- Pre-configured:
  - Black 24.0 (code formatting)
  - Ruff 0.4.0 (linting)
  - mypy 1.8.0 (static typing)
  - Pylance (type checking)

### Workflow Automation
- Auto-formatting on save
- Auto-linting on save
- Pre-configured VS Code settings
- Isolated development environment

### Standardized Structure
```
oat/
├── .devcontainer/        # Dev Container configuration
│   ├── devcontainer.json
│   └── Dockerfile
├── .vscode/              # VS Code settings
│   └── settings.json
├── data/                 # Datasets (raw & processed)
├── notebooks/            # Jupyter notebooks
├── src/                  # Source code
│   └── oat/
│       └── __init__.py   # Package initialization
├── .gitignore            # Git ignore rules
├── pyproject.toml        # Project config & dependencies
└── README.md             # Project documentation
```

## Quick Start

### 1. Prerequisites
- [Git](https://git-scm.com/)
- [Python 3.12+](https://python.org)
- [VS Code](https://code.visualstudio.com/)
- [Dev Containers Extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

### 2. Create Project
```bash
# Install Cookiecutter
pip install cookiecutter

# Generate project
cookiecutter https://github.com/christiando/cookiecutter-python-template
```
You'll be prompted for:
- project_name: Your project's display name (e.g., "Data Analysis")
- project_slug: Directory name (auto-generated, e.g., "data_analysis")
- package_name: Python package name (defaults to project_slug)
- python_version: Python version (default: 3.11)
- Tool versions (Black, Ruff, mypy)
- include_notebooks: Whether to include notebooks directory (y/n)

### 3. Open and Setup
```bash
cd your-project-name
code .
```
When VS Code opens:
- Click "Reopen in Container" in the popup
- Wait for container to build (2-5 minutes)
- Install dependencies in terminal:

```bash
pip install -e .[dev]
```

## Development Workflow

### Standard Commands
```bash
# Run formatting
black src/

# Run linting
ruff check src/

# Run type checking
mypy src/

# Install in development mode
pip install -e .
```

### VS Code Features
- **Format on Save:** Automatically formats with Black
- **Lint on Save:** Runs Ruff and fixes fixable issues
- **Type Checking:** Real-time mypy feedback via Pylance

### Recommended Flow
1. Explore ideas in `notebooks/`
2. Refactor working code into `src/`
3. Commit changes with:
   ```bash
   git add .
   git commit -m "Descriptive message"
   ```

## Tool Configuration

### Black (Formatting)
Configure in `pyproject.toml`:
```toml
[tool.black]
line-length = 88
target-version = ['py311']
```

### Ruff (Linting)
Configure in `pyproject.toml`:
```toml
[tool.ruff]
select = ["E", "F", "W", "I", "B", "UP", "RUF"]
ignore = []
line-length = 88
```

### mypy (Type Checking)
Configure in `pyproject.toml`:
```toml
[tool.mypy]
python_version = "3.11"
strict = true
```

## Troubleshooting

### Common Issues
- **Dev Container fails to build:**
  - Ensure Docker is running
  - Allocate at least 2GB RAM to Docker
- **Formatting/linting not working:**
  - Check extensions are installed: Ruff, Black Formatter
  - Reinstall dependencies:
    ```bash
    pip install -e .[dev]
    ```
  - Reload VS Code (`Ctrl/Cmd+Shift+P > "Reload Window"`)
- **Type checking errors:**
  - Run manually to see errors:
    ```bash
    mypy src/
    ```
  - Add type annotations to resolve issues.

## FAQ

**Q: How do I update Python or tool versions?**
A: Edit `cookiecutter.json` before generating projects, or update `pyproject.toml` after generation.

**Q: Can I use this without VS Code?**
A: Yes, but Dev Container and auto-formatting features won't be available. Run tools manually:
```bash
black src/
ruff check src/
mypy src/
```

**Q: How do I add more dependencies?**
A: Add to `pyproject.toml` under `[project.optional-dependencies]` or `dependencies`.

**Q: Can I use this for production projects?**
A: Absolutely! The template enforces production-grade practices with strict type checking and linting.

📌 **Pro Tip:** Use `Ctrl/Cmd+Shift+P > "Python: Select Interpreter"` to ensure you're using the Dev Container's Python environment.

---

This comprehensive README includes:

1. **Badges** for quick version visibility
2. **Table of Contents** for easy navigation
3. **Visual Directory Structure** with ASCII tree
4. **Step-by-Step Setup Instructions**
5. **Workflow Recommendations**
6. **Configuration Details** for all tools
7. **Troubleshooting Guide**
8. **FAQ Section**

The document uses:
- Consistent formatting
- Clear section headers
- Code blocks for commands/configs
- Emoji for visual scanning
- Practical examples
- Pro tips for better usage
