from typing import List

from fastapi import APIRouter, FastAPI, Request, HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

from service.api.exceptions import UserNotFoundError, ModelNotFoundError
from service.log import app_logger
from service.reco_models import HW4


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


class LoginData(BaseModel):
    access_token: str
    token_type: str


HW4_model = HW4("checkpoints/main_model.joblib", "checkpoints/popular_model.joblib",
                "checkpoints/dataset.joblib")

router = APIRouter()

USER_DATABASE = {'test_user': {'login': 'test_user', 'password': 'qwerty'}}
MODEL_LIST = ['dummy_model', 'HW4']
AUTH_TOKEN = "test_auth"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


@router.get(
    path="/health",
    tags=["Health"],
    responses={401: {"description": "Not authenticated"}},
)
async def health(
    token: str = Depends(oauth2_scheme)
) -> str:
    return "I am alive"


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
    responses={401: {"description": "Not authenticated"},
               404: {"description": "Incorrect user_id or model_name"}},
)
async def get_reco(
    request: Request,
    model_name: str,
    user_id: int,
    token: str = Depends(oauth2_scheme)
) -> RecoResponse:
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")

    if model_name not in MODEL_LIST:
        raise ModelNotFoundError(
            error_message=f"Model name '{model_name}' not found"
        )

    if user_id > 10 ** 9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    if model_name == 'dummy_model':
        k_recs = request.app.state.k_recs
        reco = list(range(k_recs))
        return RecoResponse(user_id=user_id, items=reco)

    if model_name == 'HW4':
        reco = HW4_model(user_id)
        return RecoResponse(user_id=user_id, items=reco.tolist())


@router.post(
    "/login",
    tags=["Login"],
    responses={400: {"description": "Wrong login data"}}
)
async def login(form_data: OAuth2PasswordRequestForm = Depends()) -> LoginData:
    user_record = USER_DATABASE.get(form_data.username)
    if not user_record:
        raise HTTPException(status_code=400, detail="Wrong login data")

    if not form_data.password == user_record['password']:
        raise HTTPException(status_code=400, detail="Wrong login data")

    return LoginData(access_token=user_record['login'], token_type="bearer")


def add_views(app: FastAPI) -> None:
    app.include_router(router)
