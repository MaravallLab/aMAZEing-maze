import os
import sys

# Put <repo>/analysis/auditory on the path so `import model_validation` works
# regardless of where pytest is invoked from.
_AUDITORY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _AUDITORY not in sys.path:
    sys.path.insert(0, _AUDITORY)
