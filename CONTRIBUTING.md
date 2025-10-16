# Contributing to ToolForge

First off, thank you for considering contributing to ToolForge! It's people like you that make ToolForge such a great tool.

## Code of Conduct

This project and everyone participating in it is governed by our commitment to providing a welcoming and inspiring community for all.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* **Use a clear and descriptive title** for the issue to identify the problem.
* **Describe the exact steps which reproduce the problem** in as many details as possible.
* **Provide specific examples to demonstrate the steps**.
* **Describe the behavior you observed after following the steps** and point out what exactly is the problem with that behavior.
* **Explain which behavior you expected to see instead and why.**
* **Include screenshots and animated GIFs** if possible.
* **Include your Python version, OS, and any other relevant information.**

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* **Use a clear and descriptive title** for the issue to identify the suggestion.
* **Provide a step-by-step description of the suggested enhancement** in as many details as possible.
* **Provide specific examples to demonstrate the steps**.
* **Describe the current behavior** and **explain which behavior you expected to see instead** and why.
* **Explain why this enhancement would be useful** to most ToolForge users.

### Pull Requests

* Fill in the required template
* Do not include issue numbers in the PR title
* Include screenshots and animated GIFs in your pull request whenever possible.
* Follow the Python style guide (PEP 8).
* Include thoughtfully-worded, well-structured tests.
* Document new code based on the Documentation Styleguide
* End all files with a newline

## Development Setup

### Prerequisites

* Python 3.8 or higher
* pip
* git

### Setting Up Your Development Environment

1. Fork the repo and clone your fork:
   ```bash
   git clone https://github.com/YOUR-USERNAME/ToolForge.git
   cd ToolForge
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r gradio_webui/requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```

4. Set up your API keys as described in SETUP_GUIDE.md

5. Create a branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

### Making Changes

1. Make your changes in your feature branch
2. Add or update tests as needed
3. Update documentation if you're changing functionality
4. Run tests to make sure everything works
5. Commit your changes:
   ```bash
   git add .
   git commit -m "feat: add amazing feature"
   ```

### Commit Message Guidelines

We follow the Conventional Commits specification:

* `feat:` - A new feature
* `fix:` - A bug fix
* `docs:` - Documentation only changes
* `style:` - Changes that don't affect the meaning of the code
* `refactor:` - A code change that neither fixes a bug nor adds a feature
* `perf:` - A code change that improves performance
* `test:` - Adding missing tests or correcting existing tests
* `chore:` - Changes to the build process or auxiliary tools

Example:
```
feat: add support for Claude 3 models

- Add Claude 3 model detection
- Update API client for Claude compatibility
- Add configuration examples
```

### Submitting Your Pull Request

1. Push your changes to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Open a Pull Request from your fork to the main repository

3. In your PR description:
   * Describe what changes you made and why
   * Reference any related issues
   * Include screenshots if UI changes are involved
   * List any breaking changes

4. Wait for review and address any feedback

## Style Guidelines

### Python Style Guide

* Follow PEP 8
* Use meaningful variable and function names
* Add docstrings to functions and classes
* Keep functions focused and concise
* Use type hints where appropriate

Example:
```python
def process_data(input_file: str, max_lines: int = 100) -> dict:
    """
    Process data from input file.
    
    Args:
        input_file: Path to the input JSONL file
        max_lines: Maximum number of lines to process
        
    Returns:
        Dictionary containing processing results
        
    Raises:
        FileNotFoundError: If input file doesn't exist
    """
    # Implementation here
    pass
```

### Documentation Style Guide

* Use clear, concise language
* Include code examples where helpful
* Use proper markdown formatting
* Keep the tone friendly and helpful
* Update the README if you're adding a new feature

### Git Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line

## Project Structure

```
ToolForge/
├── stage_1_label/         # Tool labeling
├── stage_2_generate/      # Data generation
├── stage_3_judge/         # Validation
├── gradio_webui/          # Web interface
└── generate_virtual_tool/ # Virtual tools
```

When adding new features:
* Place code in the appropriate directory
* Add tests for new functionality
* Update relevant documentation
* Follow existing patterns and conventions

## Adding New Features to Gradio UI

See `gradio_webui/HOW_TO_ADD_FEATURE.md` for detailed instructions on adding new features to the web interface.

Basic steps:
1. Copy `feature_template.py`
2. Implement your feature logic
3. Add UI in `quick_fast.py`
4. Test thoroughly
5. Document the feature

## Testing

Before submitting a PR:

1. Test your changes manually
2. Run any existing automated tests
3. Add new tests for new functionality
4. Ensure all tests pass

```bash
# Example test run (if tests are available)
python -m pytest tests/
```

## Documentation

When adding new features:

* Update README.md if adding major functionality
* Add entries to SETUP_GUIDE.md if configuration is needed
* Create feature-specific docs in the appropriate directory
* Update inline code comments and docstrings

## Questions?

Feel free to:
* Open an issue for questions
* Start a discussion in GitHub Discussions
* Check existing documentation and issues

## Recognition

Contributors will be recognized in:
* The project README
* Release notes
* Special thanks in documentation

Thank you for contributing to ToolForge! 🚀

