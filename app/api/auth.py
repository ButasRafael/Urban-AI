from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from jose import jwt, JWTError
import sentry_sdk
import secrets
import os
from datetime import datetime, timedelta, timezone

from app.models.schemas import UserCreate, Token, UserOut, RefreshIn, PasswordResetRequest
from app.core.database import get_db
from app.models.user import User as UserModel, RoleEnum
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
from app.services.notification_channels import send_email
import logging
logger = logging.getLogger(__name__)

router = APIRouter()

def unauthorized(detail: str):
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )

def get_verification_url(token: str, platform: str = 'web') -> str:
    """Generate verification URL based on platform"""
    if platform == 'mobile':
        # Expo Go format: exp://HOST/--/verify-email?token=TOKEN
        expo_host = os.getenv('EXPO_DEV_HOST', 'localhost:8081')
        return f"exp://{expo_host}/--/verify-email?token={token}"
    else:
        web_url = os.getenv('WEB_APP_URL', 'http://localhost:5173')
        return f"{web_url}/verify-email?token={token}"

def get_reset_password_url(token: str, platform: str = 'web') -> str:
    """Generate password reset URL based on platform"""
    if platform == 'mobile':
        # Expo Go format: exp://HOST/--/reset-password?token=TOKEN
        expo_host = os.getenv('EXPO_DEV_HOST', 'localhost:8081')
        return f"exp://{expo_host}/--/reset-password?token={token}"
    else:
        web_url = os.getenv('WEB_APP_URL', 'http://localhost:5173')
        return f"{web_url}/reset-password?token={token}"


@router.post("/register", status_code=201)
def register(user: UserCreate, platform: str = 'web', db: Session = Depends(get_db)):
    logger.info("Register attempt", extra={"username": user.username, "email": user.email, "platform": platform})

    # Check if username already exists
    if db.query(UserModel).filter_by(username=user.username).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists",
        )

    # Check if email already exists
    if db.query(UserModel).filter_by(email=user.email).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already exists",
        )

    # Generate email verification token (valid for 24 hours)
    verification_token = secrets.token_urlsafe(32)
    verification_expires = datetime.now(timezone.utc) + timedelta(hours=24)

    # Web app can create authority/admin users, mobile app restricted to user role
    if platform == 'mobile':
        user_role = RoleEnum.user
    else:
        # Web app: use the role from the request
        user_role = RoleEnum[user.role]

    new_user = UserModel(
        username=user.username,
        email=user.email,
        hashed_password=get_password_hash(user.password),
        role=user_role,
        email_verified=False,
        email_verification_token=verification_token,
        email_verification_expires=verification_expires,
    )
    db.add(new_user)

    try:
        db.commit()
        db.refresh(new_user)
    except IntegrityError as e:
        db.rollback()
        error_msg = str(e).lower()
        if "username" in error_msg:
            detail = "Username already exists"
        elif "email" in error_msg:
            detail = "Email already exists"
        else:
            detail = "User already exists"
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
        )
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.exception("DB commit failed", extra={"username": user.username})
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        )

    # Send verification email
    try:
        verification_url = get_verification_url(verification_token, platform)
        email_subject = "Verify your Urban AI account"
        email_body = f"""
Welcome to Urban AI!

Please verify your email address by clicking the link below:

{verification_url}

This link will expire in 24 hours.

If you didn't create an account, please ignore this email.

Best regards,
Urban AI Team
"""
        send_email(new_user.email, email_subject, email_body)
        logger.info("Verification email sent", extra={"username": new_user.username, "email": new_user.email})
    except Exception as e:
        logger.error(f"Failed to send verification email: {e}", exc_info=True)
        # Don't fail registration if email fails

    logger.info("User registered", extra={"username": new_user.username})
    return {
        "message": "User registered successfully. Please check your email to verify your account.",
        "email": new_user.email
    }

@router.post("/login", response_model=Token)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    logger.info("Login attempt", extra={"username_or_email": form_data.username})

    # Try to find user by username or email
    user = db.query(UserModel).filter(
        (UserModel.username == form_data.username) | (UserModel.email == form_data.username)
    ).first()

    if not user or not verify_password(form_data.password, user.hashed_password):
        logger.warning("Failed login", extra={"username_or_email": form_data.username})
        raise HTTPException(status_code=400, detail="Invalid username/email or password")

    # Check if email is verified
    if not user.email_verified:
        logger.warning("Login attempt with unverified email", extra={"username": user.username})
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email not verified. Please check your email and verify your account before logging in."
        )

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

@router.get("/verify-email")
def verify_email(token: str, db: Session = Depends(get_db)):
    """
    Verify user email with the verification token
    """
    logger.info("Email verification attempt", extra={"token": token[:10] + "..."})

    # Find user with this verification token
    user = db.query(UserModel).filter_by(email_verification_token=token).first()

    if not user:
        logger.warning("Invalid verification token", extra={"token": token[:10] + "..."})
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification token"
        )

    # Check if already verified
    if user.email_verified:
        logger.info("Email already verified", extra={"username": user.username})
        return {"message": "Email already verified. You can now log in."}

    # Check if token has expired
    if not user.email_verification_expires or user.email_verification_expires < datetime.now(timezone.utc):
        logger.warning("Expired verification token", extra={"username": user.username})
        # Clear expired token
        user.email_verification_token = None
        user.email_verification_expires = None
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Verification token has expired. Please request a new verification email."
        )

    # Verify the email
    user.email_verified = True
    user.email_verification_token = None  # Clear the token
    user.email_verification_expires = None  # Clear the expiry

    try:
        db.commit()
        logger.info("Email verified successfully", extra={"username": user.username, "email": user.email})
        return {
            "message": "Email verified successfully! You can now log in.",
            "username": user.username
        }
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.exception("Failed to verify email", extra={"username": user.username})
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify email. Please try again."
        )

@router.post("/resend-verification")
def resend_verification_email(email: str, platform: str = 'web', db: Session = Depends(get_db)):
    """
    Resend verification email to a user
    """
    logger.info("Resend verification email request", extra={"email": email, "platform": platform})

    user = db.query(UserModel).filter_by(email=email).first()

    if not user:
        # Don't reveal if email exists or not for security
        return {"message": "If the email exists, a verification link has been sent."}

    if user.email_verified:
        return {"message": "Email is already verified. You can log in."}

    # Generate new verification token (valid for 24 hours)
    verification_token = secrets.token_urlsafe(32)
    verification_expires = datetime.now(timezone.utc) + timedelta(hours=24)
    user.email_verification_token = verification_token
    user.email_verification_expires = verification_expires

    try:
        db.commit()

        # Send verification email
        verification_url = get_verification_url(verification_token, platform)
        email_subject = "Verify your Urban AI account"
        email_body = f"""
Welcome to Urban AI!

Please verify your email address by clicking the link below:

{verification_url}

This link will expire in 24 hours.

If you didn't create an account, please ignore this email.

Best regards,
Urban AI Team
"""
        send_email(user.email, email_subject, email_body)
        logger.info("Verification email resent", extra={"username": user.username, "email": user.email})

        return {"message": "Verification email sent. Please check your inbox."}
    except Exception as e:
        logger.error(f"Failed to resend verification email: {e}", exc_info=True)
        db.rollback()
        return {"message": "If the email exists, a verification link has been sent."}

@router.post("/forgot-password")
def forgot_password(email: str, platform: str = 'web', db: Session = Depends(get_db)):
    """
    Request a password reset email
    """
    logger.info("Password reset request", extra={"email": email, "platform": platform})

    user = db.query(UserModel).filter_by(email=email).first()

    if not user:
        # Don't reveal if email exists or not for security
        return {"message": "If the email exists, a password reset link has been sent."}

    # Check if email is verified
    if not user.email_verified:
        logger.warning("Password reset attempt for unverified email", extra={"email": email})
        # Don't reveal that email is unverified for security
        return {"message": "If the email exists, a password reset link has been sent."}

    # Generate password reset token (valid for 1 hour)
    reset_token = secrets.token_urlsafe(32)
    reset_expires = datetime.now(timezone.utc) + timedelta(hours=1)

    user.password_reset_token = reset_token
    user.password_reset_expires = reset_expires

    try:
        db.commit()

        # Send password reset email
        reset_url = get_reset_password_url(reset_token, platform)
        email_subject = "Reset your Urban AI password"
        email_body = f"""
Hello,

We received a request to reset your password for your Urban AI account.

Click the link below to reset your password:

{reset_url}

This link will expire in 1 hour.

If you didn't request a password reset, please ignore this email. Your password will remain unchanged.

Best regards,
Urban AI Team
"""
        send_email(user.email, email_subject, email_body)
        logger.info("Password reset email sent", extra={"username": user.username, "email": user.email})

        return {"message": "If the email exists, a password reset link has been sent. Please check your inbox."}
    except Exception as e:
        logger.error(f"Failed to send password reset email: {e}", exc_info=True)
        db.rollback()
        return {"message": "If the email exists, a password reset link has been sent."}

@router.post("/reset-password")
def reset_password(reset_request: PasswordResetRequest, db: Session = Depends(get_db)):
    """
    Reset user password with a valid reset token
    """
    logger.info("Password reset attempt", extra={"token": reset_request.token[:10] + "..."})

    # Find user with this reset token
    user = db.query(UserModel).filter_by(password_reset_token=reset_request.token).first()

    if not user:
        logger.warning("Invalid reset token", extra={"token": reset_request.token[:10] + "..."})
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token"
        )

    # Check if token has expired
    if not user.password_reset_expires or user.password_reset_expires < datetime.now(timezone.utc):
        logger.warning("Expired reset token", extra={"username": user.username})
        # Clear expired token
        user.password_reset_token = None
        user.password_reset_expires = None
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Reset token has expired. Please request a new password reset."
        )

    # Reset the password
    user.hashed_password = get_password_hash(reset_request.new_password)
    user.password_reset_token = None  # Clear the token
    user.password_reset_expires = None  # Clear the expiry

    try:
        db.commit()
        logger.info("Password reset successfully", extra={"username": user.username})

        # Send confirmation email
        try:
            email_subject = "Your Urban AI password has been reset"
            email_body = f"""
Hello,

Your password for your Urban AI account has been successfully reset.

If you did not make this change, please contact support immediately.

Best regards,
Urban AI Team
"""
            send_email(user.email, email_subject, email_body)
        except Exception as e:
            logger.error(f"Failed to send password reset confirmation email: {e}", exc_info=True)
            # Don't fail the reset if email fails

        return {
            "message": "Password reset successfully. You can now log in with your new password.",
            "username": user.username
        }
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.exception("Failed to reset password", extra={"username": user.username})
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset password. Please try again."
        )
