
from passlib.context import CryptContext
from jose import jwt, JWTError
from datetime import datetime, timedelta, timezone
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.user import User as UserModel
from app.models.schemas import UserOut
from app.models.revoked import RevokedToken
import uuid
import os
import sentry_sdk
from fastapi import status

import logging

logger = logging.getLogger(__name__)

SECRET_KEY = os.getenv("SECRET_KEY", "dev-key")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_EXPIRE_MIN", 60))
REFRESH_EXPIRE_MINUTES = int(os.getenv("REFRESH_EXPIRE_MIN", 10080))
ALGORITHM = os.getenv("ALGORITHM", "HS256")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(
    data: dict,
    expires_delta: timedelta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
) -> str:
    now = datetime.now(timezone.utc)
    to_encode = data.copy()
    to_encode.update({
        "iat": now,
        "exp": now + expires_delta,
        "jti": uuid.uuid4().hex,
    })
    
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(
    data: dict,
    expires_delta: timedelta = timedelta(minutes=REFRESH_EXPIRE_MINUTES),
) -> str:
    now = datetime.now(timezone.utc)
    to_encode = data.copy()
    to_encode.update({
        "iat": now,
        "exp": now + expires_delta,
        "jti": uuid.uuid4().hex,
    })
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> UserOut:
    logger.debug("Validating token", extra={"token": token[:8] + "...",})
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        jti: str = payload.get("jti")
        if username is None:
            logger.warning("Token missing subject", extra={"jti": jti})
            raise HTTPException(status_code=401, detail="Invalid token")
        if is_token_revoked(jti, db):
            logger.warning("Token revoked", extra={"jti": jti})
            raise HTTPException(status_code=401, detail="Token has been revoked")
    except JWTError as e:
        sentry_sdk.capture_exception(e)
        logger.error("Token decode error", exc_info=e)
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(UserModel).filter(UserModel.username == username).first()
    if user is None:
        logger.warning("User not found for token", extra={"username": username})
        raise HTTPException(status_code=401, detail="User not found")
    logger.info("Authenticated user", extra={"username": username, "jti": jti})
    return UserOut(username=user.username, role=user.role)


def revoke_token(jti: str, db: Session):
    if not db.query(RevokedToken).filter_by(jti=jti).first():
        db.add(RevokedToken(jti=jti))
        db.commit()


def is_token_revoked(jti: str, db: Session) -> bool:
    return db.query(RevokedToken).filter_by(jti=jti).first() is not None

def require_roles(*allowed_roles: str):

    def role_checker(current_user: UserOut = Depends(get_current_user)):
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Operation not permitted",
            )
        return current_user
    return Depends(role_checker)