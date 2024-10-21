"""Tests for the Letta Backend APIs."""

from unittest.mock import AsyncMock, patch

import pytest
import json

from homeassistant.components.openai_conversation.letta_api import (
    list_agents,
    list_llm_backends,
    send_message,
)
from homeassistant.core import HomeAssistant


@pytest.mark.asyncio
@patch("homeassistant.components.openai_conversation.letta_api._get_request")
async def test_list_llm_backends(mock_get_request, hass: HomeAssistant) -> None:
    """Test fetching list of LLM backends."""
    mock_response = AsyncMock()
    mock_response.json.return_value = {"models": ["model1", "model2"]}
    mock_get_request.return_value = mock_response

    response = await list_llm_backends(hass, None, None)
    mock_get_request.assert_called_once_with(
        "http://localhost:8283/v1/models/", None, 6.0
    )
    assert await response.json() == {"models": ["model1", "model2"]}


@pytest.mark.asyncio
@patch("homeassistant.components.openai_conversation.letta_api._get_request")
async def test_list_agents(mock_get_request, hass: HomeAssistant) -> None:
    """Test listing agents associated with a user."""
    mock_response = AsyncMock()
    mock_response.json.return_value = {"agents": ["agent1", "agent2"]}
    mock_get_request.return_value = mock_response

    response = await list_agents(hass, None, "user123")
    mock_get_request.assert_called_once_with(
        "http://localhost:8283/v1/agents/",
        {"user_id": "user123", "Authorization": "Bearer <token>"},
        6.0,
    )
    assert await response.json() == {"agents": ["agent1", "agent2"]}


@pytest.mark.asyncio
@patch("homeassistant.components.openai_conversation.letta_api._post_request")
async def test_send_message(mock_post_request, hass: HomeAssistant) -> None:
    """Test sending a message to an agent."""
    mock_response = AsyncMock()
    mock_response.json.return_value = {"response": "response text"}
    mock_post_request.return_value = mock_response

    data = {"message": "hello"}
    response = await send_message(hass, None, "user123", "agent456", data)
    mock_post_request.assert_called_once_with(
        "http://localhost:8283/v1/agents/agent456/messages",
        {
            "user_id": "user123",
            "Authorization": "Bearer <token>",
            "Content-Type": "application/json",
        },
        data,
        6.0,
    )
    assert await response.json() == {"response": "response text"}
