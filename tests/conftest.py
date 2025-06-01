# tests/conftest.py
import pytest
import asyncio
from httpx import AsyncClient
from app.main import app # Import your FastAPI app

@pytest.fixture(scope="session")
def event_loop():
    # Pytest-asyncio needs a session-scoped event loop for some cases.
    # For newer versions, this might not be strictly necessary if using `asyncio_mode = auto`.
    # However, it's good practice for compatibility.
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def test_client():
    async with AsyncClient(app=app, base_url="http://test") as client:
        # You might want to override dependencies here for testing, e.g., mock model loading.
        # For now, this client assumes the app can start.
        # Model loading (app.main.load_pipeline_models) is called at app startup.
        # For unit tests of endpoints, this loading should ideally be mocked to avoid
        # actual model downloads and GPU initialization, especially if those are slow
        # or resource-intensive. This can be done using pytest-mock or by structuring
        # the app to allow dependency injection for the model loading part.
        # For this example, we assume that if load_models_on_startup fails,
        # tests that don't rely on models might still pass, or those that do will fail
        # gracefully if the pipeline isn't actually called (e.g. mocked).
        yield client
