# Starlight Agent Guidelines

## Development Commands
- **Test single file**: `python -m pytest tests/test_file.py::test_function -v`
- **Run all tests**: `python -m pytest tests/ -v`
- **Lint**: `ruff check .` or `flake8 .`
- **Format**: `ruff format .` or `black .`
- **Type check**: `mypy .` or `pyright .`

## Code Style Guidelines
- **Imports**: Group stdlib, third-party, local imports with blank lines
- **Formatting**: Use ruff/black with 88-character line length
- **Types**: Add type hints for function signatures and complex variables
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **Error handling**: Use try/except with specific exceptions, include meaningful error messages
- **Documentation**: Add docstrings for public functions and classes

## Project Structure
- Models go in `model/` directory with standardized `inference.py` interface
- Dataset generators use `data_generator.py` naming pattern
- Steganography modules: `lsb_steganography.py`, `exif_steganography.py`
- Training scripts: `train.py` with CFG dict for configuration

## Testing
- Test both clean and stego images for detection/extraction
- Verify ONNX model compatibility after export
- Test with multiple image formats (PNG, JPEG, WEBP)