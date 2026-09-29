from pydantic import BaseModel, Field

class SigninRequest(BaseModel):
    emailOrPhone: str = Field(min_length=1)
    password: str = Field(min_length=8)

class SigninUserResponse(BaseModel):
    id: str
    email: str
    phone: str | None=None
    firstName: str | None=None
    lastName: str | None=None

    model_config={
        'from_attributes':True
    }

class SigninResponse(BaseModel):
    accessToken: str
    refreshToken: str
    user: SigninUserResponse

class RefreshRequest(BaseModel):
    refreshToken: str = Field(min_length=1)


class RefreshResponse(BaseModel):
    accessToken: str