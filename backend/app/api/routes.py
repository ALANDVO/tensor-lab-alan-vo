from fastapi import APIRouter

from app.api.tokenizer_routes import router as tokenizer_router
from app.api.prediction_routes import router as prediction_router


router = APIRouter()

router.include_router(tokenizer_router)
router.include_router(prediction_router)


@router.get("/health")
async def health():
    return {"status": "ok"}
