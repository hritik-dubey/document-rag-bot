import firebase_admin
from firebase_admin import credentials, auth
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging
from pydantic import BaseModel

from app.core.config import settings # To get FIREBASE_CREDENTIALS_PATH

logger = logging.getLogger(__name__)
logging.basicConfig(level=settings.LOG_LEVEL.upper())

firebase_initialized = False
try:
    if hasattr(settings, 'FIREBASE_CREDENTIALS_PATH') and settings.FIREBASE_CREDENTIALS_PATH:
        cred = credentials.Certificate(settings.FIREBASE_CREDENTIALS_PATH)
        firebase_admin.initialize_app(cred)
        firebase_initialized = True
        logger.info("Firebase Admin SDK initialized successfully.")
    else:
        logger.warning("FIREBASE_CREDENTIALS_PATH not set in environment variables or is empty. Firebase Admin SDK not initialized.")
except ValueError as ve:
    logger.error(f"Error initializing Firebase Admin SDK: {ve}. This often means the credentials path is incorrect or the JSON file is malformed.")
except Exception as e:
    logger.error(f"Unexpected error initializing Firebase Admin SDK: {e}")

class User(BaseModel):
    uid: str
    email: str | None = None
    name: str | None = None
    picture: str | None = None

# Custom bearer scheme to extract token
bearer_scheme = HTTPBearer(auto_error=False) # auto_error=False to handle missing token more gracefully

async def get_current_user(token: HTTPAuthorizationCredentials | None = Depends(bearer_scheme)) -> User:
    if not firebase_initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Firebase is not initialized. Cannot authenticate users."
        )
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated. No bearer token provided.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        id_token = token.credentials
        decoded_token = auth.verify_id_token(id_token)
        return User(
            uid=decoded_token.get("uid"),
            email=decoded_token.get("email"),
            name=decoded_token.get("name"),
            picture=decoded_token.get("picture")
        )
    except auth.ExpiredIdTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except auth.InvalidIdTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}", # Provide more specific error from Firebase if possible
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e: # Catch any other Firebase auth errors
        logger.error(f"Error during token verification: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not verify token due to an internal error."
        )
