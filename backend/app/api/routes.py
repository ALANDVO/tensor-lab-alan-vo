from fastapi import APIRouter

from app.api.tokenizer_routes import router as tokenizer_router


router = APIRouter()

router.include_router(tokenizer_router)


@router.get("/health")
async def health():
    return {"status": "ok"}