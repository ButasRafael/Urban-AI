
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from jose import jwt, JWTError
import sentry_sdk

from app.models.schemas import UserCreate, Token, UserOut, RefreshIn
from app.core.database import get_db
from app.models.user import User as UserModel
from app.core.security import (
    get_password_hash,
    verify_password,
    create_access_token,
    get_current_user,
    create_refresh_token,
    is_token_revoked,
    revoke_token,
    SECRET_KEY,
    ALGORITHM,
    oauth2_scheme,
)
import logging
logger = logging.getLogger(__name__)

router = APIRouter()

def unauthorized(detail: str):
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


@router.post("/register", status_code=201)
def register(user: UserCreate, db: Session = Depends(get_db)):
    logger.info("Register attempt", extra={"username": user.username})

    # early check (optional, but catches sooner)
    if db.query(UserModel).filter_by(username=user.username).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists",
        )

    new_user = UserModel(
        username=user.username,
        hashed_password=get_password_hash(user.password),
        role=user.role,
    )
    db.add(new_user)

    try:
        db.commit()
    except IntegrityError as e:
        # e.g. unique constraint failure
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists",
        )
    except Exception as e:
        # anything else is a genuine 500
        sentry_sdk.capture_exception(e)
        logger.exception("DB commit failed", extra={"username": user.username})
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        )

    db.refresh(new_user)
    logger.info("User registered", extra={"username": new_user.username})
    return {"message": "User registered successfully"}

@router.post("/login", response_model=Token)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    logger.info("Login attempt", extra={"username": form_data.username})
    user = db.query(UserModel).filter_by(username=form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        logger.warning("Failed login", extra={"username": form_data.username})
        raise HTTPException(status_code=400, detail="Invalid username or password")
    
    logger.info("Login success", extra={"username": user.username})
    access = create_access_token({"sub": user.username})
    refresh = create_refresh_token({"sub": user.username})
    return {"access_token": access, "refresh_token": refresh, "token_type": "bearer"}

@router.get("/me", response_model=UserOut)
def me(current_user: UserOut = Depends(get_current_user)):
    logger.info("User info requested", extra={"username": current_user.username})
    return current_user

@router.post("/refresh", response_model=Token)
def refresh_token(payload: RefreshIn, db: Session = Depends(get_db)):
    logger.info("Refresh token request", extra={"refresh_token": payload.refresh_token})
    try:
        data = jwt.decode(payload.refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError as e:
        sentry_sdk.capture_exception(e)
        logger.warning("Refresh token decode failed", exc_info=e)
        unauthorized("Invalid refresh token")

    jti = data.get("jti")
    if not jti or is_token_revoked(jti, db):
        logger.warning("Refresh token already revoked", extra={"jti": jti})
        unauthorized("Refresh token revoked")

    new_access = create_access_token({"sub": data["sub"]})
    new_refresh = create_refresh_token({"sub": data["sub"]})
    revoke_token(jti, db) 
    logger.info("Refresh token granted", extra={"new_jti": new_refresh})
    return {
        "access_token": new_access,
        "refresh_token": new_refresh,
        "token_type": "bearer",
    }

@router.post("/logout")
def logout(
    current_user=Depends(get_current_user),
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        jti = payload.get("jti", token)
    except JWTError as e:
        sentry_sdk.capture_exception(e)
        jti = token

    logger.info("Logout", extra={"username": current_user.username, "jti": jti})
    revoke_token(jti, db)
    return {"detail": "Token revoked"}
