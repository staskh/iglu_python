# Development Notes

## Package Build and Upload Commands

### Building the Package

The project uses Hatch as the build system (configured in `pyproject.toml`).

```bash
# Build both wheel and source distributions
hatch build
```

This creates two files in the `dist/` directory:
- `iglu_python-0.1.0-py3-none-any.whl` - Wheel distribution (preferred for installation)
- `iglu_python-0.1.0.tar.gz` - Source distribution

### Uploading to PyPI

#### Prerequisites
1. Create an account at [pypi.org](https://pypi.org)
2. Generate an API token:
   - Go to Account Settings → API tokens
   - Create a new token with appropriate scope
3. Set up authentication (choose one method):

**Method 1: Environment Variables**
```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-your-api-token-here
```

**Method 2: ~/.pypirc file**
```ini
[distutils]
index-servers = pypi

[pypi]
username = __token__
password = pypi-your-api-token-here
```

#### Upload Commands
```bash
# Upload all distribution files to PyPI
twine upload dist/*

# Upload with verbose output for debugging
twine upload --verbose dist/*

# Upload to test PyPI first (recommended)
twine upload --repository testpypi dist/*
```

### Complete Build and Upload Workflow

```bash
# 1. Clean previous builds (optional)
rm -rf dist/

# 2. Build the package
hatch build

# 3. Check the package
twine check dist/*

# 4. Upload to test PyPI (recommended first)
twine upload --repository testpypi dist/*

# 5. Upload to production PyPI
twine upload dist/*
```

### Testing Installation

```bash
# Install from local wheel
pip install dist/iglu_python-0.1.0-py3-none-any.whl

# Install from PyPI (after upload)
pip install iglu_python

# Install from test PyPI
pip install --index-url https://test.pypi.org/simple/ iglu_python
```

### Troubleshooting

- **403 Forbidden**: Check your API token and authentication setup
- **Invalid distribution format**: Ensure only `.whl` and `.tar.gz` files are in `dist/`
- **Package already exists**: Increment version in `pyproject.toml` and rebuild

### Dependencies

Required development dependencies for building and uploading:
- `hatch` - Build system
- `twine` - Upload tool

These are included in `requirements-dev.txt`.

## Code Quality with Ruff

### What is Ruff?

Ruff is an extremely fast Python linter and code formatter written in Rust. It replaces multiple Python development tools with a single, lightning-fast alternative (10-100x faster than traditional Python linters).

**Replaces multiple tools:**
- `flake8` (linting)
- `isort` (import sorting)
- `pydocstyle` (docstring linting)
- `pyupgrade` (syntax modernization)
- `bandit` (security linting)
- And many more...

### Current Project Configuration

The project is configured in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 88
target-version = "py38"
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings  
    "F",   # pyflakes
    "I",   # isort
    "C",   # flake8-comprehensions
    "B",   # flake8-bugbear
]
ignore = []

[tool.ruff.isort]
known-first-party = ["iglu_python"]

[tool.ruff.per-file-ignores]
"tests/*" = ["E501"]  # Allow long lines in tests
```

### Basic Ruff Commands

```bash
# Check code for issues
ruff check .

# Check and auto-fix issues
ruff check --fix .

# Format code (like black)
ruff format .

# Check specific files
ruff check iglu_python/

# Check with output format options
ruff check --output-format=json .
ruff check --output-format=github .  # For GitHub Actions

# Show all available rules
ruff linter

# Show configuration
ruff config
```

### Integration with Development Workflow

```bash
# Check and fix both linting and formatting
ruff check --fix . && ruff format .

# Add to pre-commit workflow
ruff check --diff .        # Show what would be changed
ruff format --check .      # Check if formatting is needed

# Generate reports
ruff check --statistics .  # Show rule violation statistics
ruff check --show-files .  # Show which files have issues
```

### Rule Categories

The project uses these rule categories:

- **E/W**: PyCodeStyle errors and warnings (PEP 8 compliance)
- **F**: Pyflakes (unused imports, undefined variables)
- **I**: Import sorting (replaces isort)
- **C**: Comprehensions (list/dict comprehension improvements)
- **B**: Bugbear (common Python gotchas and anti-patterns)

### Common Rule Examples

```python
# E203: Whitespace before ':'
bad:  x[1 : 2]
good: x[1:2]

# F401: Unused import
bad:  import os  # if 'os' is never used
good: # Remove unused imports

# I001: Import order
bad:  import sys
      import os
      import numpy
good: import os
      import sys
      
      import numpy

# C400: List comprehension
bad:  list(x for x in items)
good: [x for x in items]
```

### IDE Integration

**VS Code:**
1. Install "Ruff" extension
2. Configure in settings.json:
```json
{
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "none",
    "ruff.format.args": ["--config=pyproject.toml"]
}
```

**PyCharm:**
1. Install "Ruff" plugin
2. Configure in File → Settings → Tools → Ruff

### Performance Benefits

- **Speed**: 10-100x faster than flake8/pylint
- **Single tool**: Replaces multiple linting tools
- **Auto-fixing**: Can automatically fix many issues
- **Modern**: Written in Rust, actively developed

### Common Workflow

```bash
# Daily development
ruff check --fix .     # Fix auto-fixable issues
ruff format .          # Format code

# Before committing
ruff check .           # Ensure no remaining issues
ruff format --check .  # Ensure proper formatting

# CI/CD pipeline
ruff check --output-format=github .  # For GitHub Actions
ruff format --check .                # Fail if not formatted
```

### Troubleshooting

**Ignore specific rules temporarily:**
```python
# ruff: noqa: E501
very_long_line_that_exceeds_88_characters_and_cannot_be_broken_easily()

# Multiple rules
# ruff: noqa: E501, F401
```

**Configure per-file ignores in pyproject.toml:**
```toml
[tool.ruff.per-file-ignores]
"tests/*" = ["E501", "F401"]  # Allow long lines and unused imports in tests
"__init__.py" = ["F401"]      # Allow unused imports in __init__.py files
```

### Dependencies

Ruff is included in the project's development dependencies:
- Listed in `requirements-dev.txt` (if using pip)
- Or use: `pip install ruff`

## Testing with pytest and Coverage

### Overview

The project uses `pytest` for testing and `pytest-cov` for code coverage analysis. This combination provides comprehensive testing capabilities with detailed coverage reporting.

### Current Project Configuration

Testing is configured in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --cov=iglu_python --cov-report=term-missing"
```

**Configuration explained:**
- `testpaths`: Look for tests in the `tests/` directory
- `python_files`: Test files must start with `test_`
- `addopts`: Auto-enable verbose mode and coverage with missing lines report

### Basic pytest Commands

```bash
# Run all tests with coverage (uses pyproject.toml config)
pytest

# Run tests without coverage (faster for debugging)
pytest --no-cov

# Run specific test file
pytest tests/test_hbgi.py

# Run specific test function
pytest tests/test_hbgi.py::test_hbgi_basic

# Run tests matching pattern
pytest -k "hbgi"

# Stop on first failure
pytest -x

# Show local variables on failures
pytest -l

# Run in parallel (if pytest-xdist installed)
pytest -n auto
```

### Coverage Commands and Reports

```bash
# Basic coverage (terminal summary)
pytest --cov-report=term

# Coverage with missing line numbers
pytest --cov-report=term-missing

# Generate interactive HTML report
pytest --cov-report=html
# Opens: htmlcov/index.html

# Multiple report formats
pytest --cov-report=term --cov-report=html --cov-report=xml

# JSON report for automation/CI
pytest --cov-report=json

# Coverage for specific modules only
pytest --cov=iglu_python.hbgi --cov=iglu_python.lbgi

# Fail if coverage below threshold
pytest --cov-fail-under=80

# Branch coverage (more detailed)
pytest --cov-branch
```

### Understanding Coverage Reports

**Terminal Output Example:**
```
Name                                 Stmts   Miss  Cover   Missing
------------------------------------------------------------------
iglu_python/hbgi.py                     28      3    89%   66, 82, 95
iglu_python/lbgi.py                     26      2    92%   27, 109
iglu_python/metrics.py                 155    155     0%   1-465
------------------------------------------------------------------
TOTAL                                 1288    334    74%
```

**Columns explained:**
- **Stmts**: Total lines of executable code
- **Miss**: Lines not covered by tests
- **Cover**: Percentage of lines covered
- **Missing**: Specific line numbers not tested

### Coverage Report Types

| Report Type | Command | Output | Best For |
|-------------|---------|--------|----------|
| `term` | `--cov-report=term` | Terminal summary | Quick overview |
| `term-missing` | `--cov-report=term-missing` | Terminal + line numbers | Debugging |
| `html` | `--cov-report=html` | Interactive web page | Detailed analysis |
| `xml` | `--cov-report=xml` | XML file | CI/CD integration |
| `json` | `--cov-report=json` | JSON file | Automation/scripting |

### Test Organization

```
tests/
├── test_hbgi.py              # Tests for hbgi module
├── test_lbgi.py              # Tests for lbgi module
├── test_utils.py             # Tests for utility functions
├── data/                     # Test data files
│   └── example_data.csv
└── conftest.py               # Shared fixtures (if any)
```

### Writing Tests

**Basic test structure:**
```python
import pytest
import pandas as pd
import iglu_python as iglu

def test_hbgi_basic():
    """Test basic HBGI calculation."""
    data = pd.DataFrame({
        'id': ['subject1'] * 4,
        'time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:05:00',
                               '2020-01-01 00:10:00', '2020-01-01 00:15:00']),
        'gl': [150, 155, 160, 165]
    })
    
    result = iglu.hbgi(data)
    
    assert isinstance(result, pd.DataFrame)
    assert 'HBGI' in result.columns
    assert len(result) == 1
    assert result['HBGI'].iloc[0] > 0

def test_hbgi_missing_values():
    """Test HBGI with missing glucose values."""
    # Test implementation here
    pass

@pytest.mark.parametrize("glucose_values,expected", [
    ([100, 120, 140], True),
    ([50, 60, 70], False),
])
def test_hbgi_parametrized(glucose_values, expected):
    """Parametrized test for different glucose ranges."""
    # Test implementation here
    pass
```

### Development Workflow

```bash
# Quick test run during development
pytest tests/test_specific_module.py -v

# Test with coverage analysis
pytest tests/test_specific_module.py --cov=iglu_python.module_name

# Full test suite before committing
pytest

# Generate HTML report for detailed analysis
pytest --cov-report=html
open htmlcov/index.html  # macOS
```

### CI/CD Integration

```bash
# For GitHub Actions / CI environments
pytest --cov-report=xml --cov-report=term

# Fail build if coverage below threshold
pytest --cov-fail-under=75

# Generate reports for coverage services (Codecov, Coveralls)
pytest --cov-report=xml
```

### Improving Test Coverage

**Identify uncovered code:**
1. Run: `pytest --cov-report=html`
2. Open: `htmlcov/index.html`
3. Click on files with low coverage
4. Review highlighted uncovered lines

**Focus areas in this project:**
- `episode_calculation.py` (11% coverage)
- `metrics.py` (0% coverage)
- `pgs.py` (59% coverage)

**Coverage goals:**
- **Minimum**: 70% overall coverage
- **Target**: 85% overall coverage
- **Critical modules**: 90%+ coverage

### Common Testing Patterns

```bash
# Test only failed tests from last run
pytest --lf

# Test only changed files (requires pytest-testmon)
pytest --testmon

# Run tests in random order (requires pytest-randomly)
pytest --randomly-seed=1234

# Profile slow tests
pytest --durations=10

# Generate test report
pytest --html=report.html --self-contained-html
```

### Debugging Tests

```bash
# Drop into debugger on failure
pytest --pdb

# Drop into debugger on first failure
pytest -x --pdb

# Show print statements
pytest -s

# Increase verbosity
pytest -vv

# Show local variables in tracebacks
pytest --tb=long -l
```

### Dependencies

Testing dependencies in `requirements-dev.txt`:
- `pytest` - Testing framework
- `pytest-cov` - Coverage plugin
- Optional: `pytest-xdist` (parallel testing)
- Optional: `pytest-html` (HTML reports)

### Integration with Other Tools

```bash
# Run linting and tests together
ruff check . && pytest

# Pre-commit workflow
ruff check --fix . && ruff format . && pytest --cov-fail-under=70
``` 