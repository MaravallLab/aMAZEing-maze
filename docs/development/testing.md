# Testing

The test suite covers audio generation, trial structure, ROI tracking, data management, configuration, and integration scenarios. All hardware dependencies are mocked.

```bash
# Run all tests (harness + grammar stimulus package)
python -m pytest tests/ src/amazeing/auditory/grammar_stimuli/tests -v

# Run with coverage
python -m pytest tests/ --cov=amazeing --cov-report=term-missing

# Run a specific test file
python -m pytest tests/test_audio.py -v
```

---
