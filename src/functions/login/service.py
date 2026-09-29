from sqlalchemy import or_, select
from sqlalchemy.orm import Session
from datetime import datetime, timedelta, timezone

from src.database.models.schema import AppUser, UserSession
from src.core.security import (
    verify_password,
    create_access_token,
    create_refresh_token
)

class AuthService:

    @staticmethod
    def signin(
        db: Session,
        data
    ):
        # Find user by email OR phone
        stmt = select(AppUser).where(
            or_(
                AppUser.email == data.emailOrPhone,
                AppUser.phone == data.emailOrPhone
            )
        )

        user = db.scalar(stmt)

        # User not found
        if not user:
            raise ValueError("Invalid email/phone or password")

        # Verify password
        if not verify_password(
            data.password,
            user.password_hash
        ):
            raise ValueError("Invalid email/phone or password")

        # Create access token
        access_token = create_access_token(
            {
                "sub": str(user.id)
            }
        )

        # Create refresh token
        refresh_token = create_refresh_token()
        
        refresh_expires_at = (
            datetime.now(timezone.utc)
            + timedelta(days=30)
        )

        # create a user session object to store refresh token
        user_session = UserSession(
            user_id=user.id,
            session_token=refresh_token,
            is_revoked=False,
            expires_at=refresh_expires_at,
        )
        db.add(user_session)
        db.commit()

        return {
            "accessToken": access_token,
            "refreshToken": refresh_token,
            "user": user
        }


    @staticmethod
    def refresh(
        db: Session,
        data
    ):
        # 1. Find the refresh-token session
        stmt = select(UserSession).where(
            UserSession.session_token == data.refreshToken
        )

        session = db.scalar(stmt)

        # 2. Refresh token doesn't exist
        if not session:
            raise ValueError("Invalid refresh token")

        # 3. Check if token has been revoked
        if session.is_revoked:
            raise ValueError("Refresh token has been revoked")

        # 4. Check expiration
        now = datetime.now(timezone.utc)

        if session.expires_at <= now:
            raise ValueError("Refresh token has expired")

        # 5. Create a new access token
        access_token = create_access_token(
            {
                "sub": str(session.user_id)
            }
        )

        # 6. Return new access token
        return {
            "accessToken": access_token
        }