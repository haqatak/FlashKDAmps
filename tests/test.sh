set -e
pip install --no-build-isolation -e .
pip install flash-linear-attention matplotlib pytest
python tests/test_fwd.py
pytest tests/test_fallback.py
pytest tests/test_fwd_full.py
