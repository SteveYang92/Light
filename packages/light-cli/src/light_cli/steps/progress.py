"""Progress stage names for run reporting.

The web backend streams these stage strings to clients over SSE, so the
VALUES are an external contract — never rename them.  Stage-specific
progress callbacks live with their step modules (``steps/asr.py`` etc.).
"""

STAGE_ASR = "asr"
STAGE_CORRECT = "correct"
STAGE_PUNCT = "punct"
STAGE_SEGMENT = "segment"
STAGE_CONTEXT = "context"
STAGE_COMPOSE = "compose"
STAGE_TRANSLATE = "translate"
STAGE_ANNOTATE = "annotate"
STAGE_FORMAT = "format"
