# Testing

The test suite covers audio generation, trial structure, ROI tracking, data management, configuration, and integration scenarios. All hardware dependencies are mocked.

`python -m pytest` on its own runs both test folders, as `pyproject.toml` lists them. On GitHub, the workflow `.github/workflows/tests.yml` runs it on Windows for every push to `main` and every pull request, and the Tests badge in the README shows the latest result.

```bash
# Run all tests (harness + grammar stimulus package)
python -m pytest tests/ src/amazeing/auditory/grammar_stimuli/tests -v

# Run with coverage
python -m pytest tests/ --cov=amazeing --cov-report=term-missing

# Run a specific test file
python -m pytest tests/test_audio.py -v
```

---
