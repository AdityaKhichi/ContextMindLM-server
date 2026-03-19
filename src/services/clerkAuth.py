from fastapi import Request, HTTPException
from clerk_backend_api import AuthenticateRequestOptions, Clerk

from src.config.index import appConfig


def get_current_user(request: Request):
    try:
        sdk = Clerk(bearer_auth=appConfig["clerk_secret_key"])

        # request_state = JWT Token
        request_state = sdk.authenticate_request(
            request,
            options=AuthenticateRequestOptions(authorized_parties=appConfig["domain"]),
        )

        if not request_state.is_signed_in:
            raise HTTPException(status_code=401, detail="Not authenticated")

        clerk_id = request_state.payload.get("sub")

        if not clerk_id:
            raise HTTPException(status_code=401, detail="Invalid token")

        return clerk_id

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Authentication Failed. {str(e)}",
        )