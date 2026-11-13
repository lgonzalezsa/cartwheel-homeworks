from __future__ import annotations

import pytest
from fastapi import HTTPException


@pytest.fixture
def server_app():
    from server import app

    app._SESSIONS.clear()
    yield app
    app._SESSIONS.clear()


def test_create_session_rejects_role_that_differs_from_database(server_app, world):
    with pytest.raises(HTTPException) as exc_info:
        server_app.create_session(server_app.SessionCreate(user_id=1, role="merchant"))

    assert exc_info.value.status_code == 403


def test_create_session_rejects_unsupported_role(server_app, world):
    with pytest.raises(HTTPException) as exc_info:
        server_app.create_session(server_app.SessionCreate(user_id=1, role="admin"))

    assert exc_info.value.status_code == 400


def test_create_session_rejects_unknown_user(server_app, world):
    with pytest.raises(HTTPException) as exc_info:
        server_app.create_session(
            server_app.SessionCreate(user_id=999_999, role="shopper")
        )

    assert exc_info.value.status_code == 404


def test_session_token_cannot_authorize_another_session(server_app, world):
    from agent.auth import AuthContext

    first = server_app.create_session(
        server_app.SessionCreate(user_id=1, role="shopper")
    )
    second = server_app.create_session(
        server_app.SessionCreate(user_id=2, role="shopper")
    )

    context, session = server_app._SESSIONS[first["session_id"]]
    assert context == AuthContext(user_id=1, role="shopper", store_id=None)
    assert session is not None
    payload = server_app.verify_token(first["token"])
    assert payload is not None
    assert payload["session_id"] == first["session_id"]
    assert payload["user_id"] == 1
    assert payload["role"] == "shopper"
    assert payload["store_id"] is None
    assert isinstance(payload["issued_at"], int)

    with pytest.raises(HTTPException) as exc_info:
        server_app._authorize(
            second["session_id"], f"Bearer {first['token']}"
        )

    assert exc_info.value.status_code == 403
