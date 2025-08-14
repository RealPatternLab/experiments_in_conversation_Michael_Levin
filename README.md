# Virtual Person Pipeline

An AI-powered system that creates a digital twin of an individual by processing various types of media content and building both a RAG system and a fine-tuned language model.

## Quick Start

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd virtual-person-pipeline
   ```

2. **Create and activate virtual environment**
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   uv pip install -e .
   ```

4. **Install development dependencies (optional)**
   ```bash
   uv pip install -e ".[dev]"
   ```

### Development

- **Run tests**: `uv run pytest`
- **Format code**: `uv run black .`
- **Lint code**: `uv run flake8 .`
- **Type check**: `uv run mypy .`

### Running the Application

```bash
uv run streamlit run src/ui/streamlit_app.py
```

## Project Structure

```
src/
├── data_ingestion/      # Media processing and validation
├── content_processing/  # Text extraction and chunking
├── rag_system/         # Retrieval and generation
├── fine_tuning/        # Model training pipeline
├── api/                # FastAPI backend
└── ui/                 # Streamlit frontend
```

## Development Workflow

1. **Create feature branch**: `git checkout -b feature/your-feature`
2. **Make changes**: Follow PEP 8 and type hints
3. **Run tests**: `uv run pytest`
4. **Format code**: `uv run black .`
5. **Commit**: Use conventional commit messages
6. **Push**: `git push origin feature/your-feature`

## Contributing

- Follow PEP 8 coding standards
- Add type hints to all functions
- Write tests for new features
- Update documentation as needed

## License

[Your License Here]
