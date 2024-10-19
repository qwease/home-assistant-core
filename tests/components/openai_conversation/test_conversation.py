"""Tests for conversation.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from homeassistant.components.openai_conversation import OpenAIConfigEntry
from homeassistant.components.openai_conversation.conversation import (
    OpenAIConversationEntity,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent

# Mock constants
DOMAIN = "openai_conversation"
CONF_LLM_HASS_API = "llm_hass_api"


@pytest.fixture
def mock_entry():
    """Create a mock OpenAIConfigEntry."""
    entry = MagicMock(spec=OpenAIConfigEntry)
    entry.entry_id = "test_entry_id"
    entry.title = "Test Entry"
    entry.options = {CONF_LLM_HASS_API: "test_api"}
    return entry


@pytest.fixture
def mock_hass():
    """Create a mock HomeAssistant instance."""
    return MagicMock(spec=HomeAssistant)


@pytest.fixture
def mock_intent_response():
    """Create a mock IntentResponse."""
    return MagicMock(spec=intent.IntentResponse)


@pytest.fixture
def openai_conversation_entity(mock_entry):
    """Create an instance of OpenAIConversationEntity."""
    return OpenAIConversationEntity(mock_entry)


def test_initialize_conversation(openai_conversation_entity):
    """Test the _initialize_conversation method."""
    user_input = MagicMock()
    user_input.conversation_id = "valid_id"
    conversation_id, messages = openai_conversation_entity._initialize_conversation(
        user_input
    )
    assert conversation_id == "valid_id"
    assert messages == []


@pytest.mark.asyncio
async def test_setup_llm_api(openai_conversation_entity, mock_intent_response):
    """Test the _setup_llm_api method."""
    user_input = MagicMock()
    with patch(
        "homeassistant.helpers.llm.async_get_api", new_callable=AsyncMock
    ) as mock_get_api:
        mock_get_api.return_value = MagicMock()
        llm_api, tools = await openai_conversation_entity._setup_llm_api(
            user_input, mock_intent_response
        )
        assert llm_api is not None
        assert tools is not None


@pytest.mark.asyncio
async def test_generate_prompt(openai_conversation_entity, mock_intent_response):
    """Test the _generate_prompt method."""
    user_input = MagicMock()
    llm_api = MagicMock()
    prompt = await openai_conversation_entity._generate_prompt(
        user_input, llm_api, mock_intent_response
    )
    assert prompt is not None


@pytest.mark.asyncio
async def test_generate_response(openai_conversation_entity, mock_intent_response):
    """Test the _generate_response method."""
    user_input = MagicMock()
    conversation_id = "test_id"
    messages = []
    prompt = "Test prompt"
    tools = None
    with patch("openai.OpenAIError", new_callable=AsyncMock) as mock_openai_error:
        mock_openai_error.side_effect = Exception("Test error")
        result = await openai_conversation_entity._generate_response(
            user_input, conversation_id, messages, prompt, tools, mock_intent_response
        )
        assert result is not None
