import pytest
from httpx import AsyncClient, ASGITransport
from app.main import create_app
import pytest_asyncio


@pytest_asyncio.fixture
async def client():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac