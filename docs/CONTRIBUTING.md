# Contributing to Local LLM Testing Framework

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

### Prerequisites

- Python 3.9+
- Git
- Docker with NVIDIA Container Toolkit
- Access to a GPU (for testing)

### Setting Up Development Environment

```bash
# Clone the repository
git clone <repository-url>
cd llm-testing

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r scripts/requirements.txt

# Install development dependencies (if available)
pip install pytest pytest-cov black isort mypy
```

## How to Contribute

### Reporting Bugs

1. Check existing issues to avoid duplicates
2. Use the bug report template if available
3. Include:
   - Python version
   - GPU model and driver version
   - Steps to reproduce
   - Expected vs actual behavior
   - Error messages and logs

### Suggesting Features

1. Open an issue describing the feature
2. Explain the use case and benefits
3. Discuss implementation approach if you have ideas

### Submitting Code Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Write or update tests if applicable
5. Ensure code passes linting
6. Commit with clear messages
7. Push and create a Pull Request

## Code Standards

### Python Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Write docstrings for public functions (Google style)
- Keep functions focused and reasonably sized

### Example Function

```python
def calculate_metrics(
    gpu_index: int = 0,
    include_memory: bool = True
) -> Optional[GPUMetrics]:
    """
    Calculate GPU metrics for the specified GPU.

    Args:
        gpu_index: Index of the GPU to query (default: 0)
        include_memory: Whether to include memory metrics

    Returns:
        GPUMetrics object if successful, None if GPU query fails

    Raises:
        ValueError: If gpu_index is negative
    """
    if gpu_index < 0:
        raise ValueError("gpu_index must be non-negative")
    # Implementation...
```

### Commit Messages

Use clear, descriptive commit messages:

```
Add support for vLLM tensor parallelism

- Implement multi-GPU distribution for vLLM engine
- Add --tensor-parallel flag to batch_test.py
- Update documentation with examples
```

### Testing

- Add tests for new functionality when possible
- Run existing tests before submitting: `pytest tests/`
- Include both unit tests and integration tests where appropriate

## Project Structure

```
llm-testing/
├── scripts/
│   ├── core/           # Core modules (questions, results, metrics, runner)
│   ├── engines/        # Engine adapters (ollama, llamacpp, vllm)
│   ├── data/           # Test data (questions, inventory)
│   ├── run_test.py     # Single model testing
│   ├── batch_test.py   # Batch testing
│   ├── analyze_results.py
│   └── score_results.py
├── gpus/               # GPU-specific configs and results
├── docs/               # Documentation
└── tests/              # Test files (when added)
```

## Areas for Contribution

### High Priority

- Unit tests for core modules
- Integration tests for engine adapters
- Improved error handling and logging
- Multi-GPU support enhancements

### Medium Priority

- Additional inference engine support
- Web UI for results visualization
- Performance optimization
- Docker Compose setup

### Documentation

- API documentation
- Tutorial guides
- Video walkthroughs
- Translation to other languages

## Questions?

Open an issue with your question or reach out to the maintainers.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
