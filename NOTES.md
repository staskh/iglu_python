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