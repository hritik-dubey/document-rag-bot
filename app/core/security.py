import firebase_admin
from firebase_admin import credentials, auth
from fastapi import Depends, HTTPException, status
from app.core.db import get_async_session
from app.models.db_models import User as DBUser # Alias to avoid Pydantic model conflict
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
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

class AuthUser(BaseModel):
    uid: str # Firebase UID
    id: str # Our internal DB User ID (UUID)
    email: str | None = None
    name: str | None = None
    picture: str | None = None

# Custom bearer scheme to extract token
bearer_scheme = HTTPBearer(auto_error=False) # auto_error=False to handle missing token more gracefully

async def get_current_user(token: HTTPAuthorizationCredentials | None = Depends(bearer_scheme), db: AsyncSession = Depends(get_async_session)) -> AuthUser:
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

        # Token is valid, now get/create user in our database
        firebase_uid = decoded_token.get("uid")
        email = decoded_token.get("email")
        name = decoded_token.get("name") # Often present, good to store
        picture = decoded_token.get("picture") # Often present

        # Try to fetch user by firebase_uid
        statement = select(DBUser).where(DBUser.firebase_uid == firebase_uid)
        result = await db.exec(statement)
        db_user = result.one_or_none()

        if not db_user:
            # User does not exist, create them
            logger.info(f"User with firebase_uid {firebase_uid} not found in DB. Creating new user.")
            db_user = DBUser(
                firebase_uid=firebase_uid,
                email=email,
                # name=name, # Add if DBUser model gets a name field
                # picture=picture # Add if DBUser model gets a picture field
                # subscription_tier can be set to default
            )
            db.add(db_user)
            try:
                await db.commit()
                await db.refresh(db_user)
                logger.info(f"New user created in DB with id: {db_user.id} for firebase_uid: {firebase_uid}")
            except Exception as e_create_user:
                await db.rollback()
                logger.error(f"Failed to create user in DB for firebase_uid {firebase_uid}: {e_create_user}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Could not create user profile after authentication."
                )
        else:
            # Optionally update user info if it changed in Firebase (e.g., email, name, picture)
            # For now, just log that user was found
            logger.debug(f"User found in DB with id: {db_user.id} for firebase_uid: {firebase_uid}")
            # Example update logic (if fields were added to DBUser):
            # needs_update = False
            # if email and db_user.email != email: db_user.email = email; needs_update = True
            # if name and db_user.name != name: db_user.name = name; needs_update = True
            # if needs_update:
            #     db.add(db_user)
            #     await db.commit()
            #     await db.refresh(db_user)

        return AuthUser(
            id=db_user.id, # This is our internal DB User ID
            uid=firebase_uid, # Firebase UID
            email=db_user.email, # Email from our DB
            name=name, # Name from Firebase token (not yet in our AuthUser model, but could be)
            picture=picture # Picture from Firebase token (not yet in our AuthUser model)
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
