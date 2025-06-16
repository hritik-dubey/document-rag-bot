from app.core.security import get_current_user, User
from fastapi import APIRouter, Depends

router = APIRouter()

@router.post("/login", summary="User Authentication")
async def login(current_user: User = Depends(get_current_user)):
    return current_user
