from fastapi import APIRouter
from api.utils import BASELINE_DIR, SCORES_DIR, TRANSCRIPTS_DIR

router = APIRouter()


@router.get("/stats")
def global_stats():
    return {
        "total_transcripts":   len(list(TRANSCRIPTS_DIR.glob("*.json"))),
        "total_baselines":     len(list(BASELINE_DIR.glob("*.json"))),
        "total_scored_calls":  len(list(SCORES_DIR.glob("*.json"))),
    }
