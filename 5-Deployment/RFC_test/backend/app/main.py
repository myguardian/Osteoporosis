import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from core import config
from api.v1.api import api_router


app = FastAPI(
    title=config.PROJECT_NAME,
    openapi_url="/api/v1/openapi.json",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    )

app.include_router(api_router)


# CORS
origins = []


@app.get("/")
async def root():
    return {"message": "Hello RFC Model"}

if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="0.0.0.0")