from fastapi import APIRouter
from .endpoints import risk_level
from core import config

api_prefix = config.API_V1_STR

api_router = APIRouter()
api_router.include_router(risk_level.router, prefix=api_prefix, tags=["rfc_risk_level"])