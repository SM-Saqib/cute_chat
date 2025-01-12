# processing time api imports

from fastapi import FastAPI

from cute_chat.app.api import whatsapp_communication
from fastapi_pagination import add_pagination
from fastapi.openapi.docs import get_redoc_html

from app.services import Scheduler

scheduler = Scheduler()
scheduler.start_scheduler()


app = FastAPI(
    title="cute_chat",
    description="cute_chat API",
    version="0.1.0",
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(whatsapp_communication.router, prefix="/cute_chat/v1")


add_pagination(app)


@app.get("/redoc", include_in_schema=False)
async def get_redoc_documentation(current_user: str = None):
    return get_redoc_html(openapi_url="/openapi.json", title="docs")


if __name__ == "__main__":
    import uvicorn
    from app.services import Scheduler

    scheduler = Scheduler()
    scheduler.start_scheduler()

    uvicorn.run(app, host="0.0.0.0", port=8000)
