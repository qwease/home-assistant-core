"""Tests for the OpenAI Conversation integration."""

from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

from homeassistant.components.openai_conversation import (
    SERVICE_GENERATE_IMAGE,
    async_setup,
    async_setup_entry,
    async_unload_entry,
)
from homeassistant.components.openai_conversation.const import DOMAIN
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady, HomeAssistantError

OPENAI_BASE_URL = "https://api.openai.com/v1"
PROMPT = "A beautiful sunset"


# Test setup for OpenAI entry with memory enabled
@pytest.mark.asyncio
async def test_async_setup_entry_with_memory(hass: HomeAssistant) -> None:
    """Test setup entry for OpenAI with memory enabled."""
    mock_entry = MagicMock(spec=ConfigEntry)
    mock_entry.data = {
        "api_key": "test_api_key",
        "base_url": OPENAI_BASE_URL,
        "enable_memory": True,
    }
    mock_entry.options = {}
    mock_entry.domain = DOMAIN
    mock_entry.entry_id = "test_entry_id"
    mock_entry.setup_lock = MagicMock()
    mock_entry.title = "Test Title"

    with patch(
        "homeassistant.components.openai_conversation.letta_api.list_llm_backends",
        new=AsyncMock(),
    ) as mock_llm_backends:
        mock_llm_backends.return_value = MagicMock(status_code=200)

        result = await async_setup_entry(hass, mock_entry)
        assert result is True
        mock_llm_backends.assert_called_once()


# Test the image generation service
@pytest.mark.asyncio
async def test_render_image_service(hass: HomeAssistant) -> None:
    """Test the image generation service."""
    mock_entry = MagicMock(spec=ConfigEntry)
    mock_entry.data = {
        "api_key": "test_api_key",
        "base_url": OPENAI_BASE_URL,
    }
    mock_entry.domain = DOMAIN
    mock_entry.entry_id = "test_entry_id"

    # Mock the OpenAI client
    mock_openai_client = AsyncMock()
    mock_openai_client.images.generate = AsyncMock(
        return_value=MagicMock(
            data=[MagicMock(model_dump=lambda exclude: {"url": "http://image.url"})]
        )
    )

    # Register the service
    await async_setup(hass, {})
    hass.config_entries.async_get_entry = MagicMock(return_value=mock_entry)
    mock_entry.runtime_data = mock_openai_client

    service_data = {
        "config_entry": mock_entry.entry_id,
        "prompt": PROMPT,
        "size": "1024x1024",
        "quality": "standard",
        "style": "vivid",
    }

    # Call the service with return_response=True to prevent ServiceValidationError
    await hass.services.async_call(
        DOMAIN,
        SERVICE_GENERATE_IMAGE,
        service_data,
        blocking=True,
        return_response=True,
    )

    # Check if the service generated an image
    mock_openai_client.images.generate.assert_called_once_with(
        model="dall-e-3",
        prompt=PROMPT,
        size="1024x1024",
        quality="standard",
        style="vivid",
        response_format="url",
        n=1,
    )


# Test setup failure when the API is not reachable
@pytest.mark.asyncio
async def test_setup_entry_connection_error(hass: HomeAssistant) -> None:
    """Test setup failure due to connection error."""
    mock_entry = MagicMock(spec=ConfigEntry)
    mock_entry.data = {
        "api_key": "test_api_key",
        "base_url": OPENAI_BASE_URL,
    }
    mock_entry.domain = DOMAIN
    mock_entry.options = {}

    with patch("openai.AsyncClient") as mock_openai_client:
        mock_openai_client().models.list = AsyncMock(side_effect=ConnectionRefusedError)

        with pytest.raises(ConfigEntryNotReady):
            await async_setup_entry(hass, mock_entry)


# Test unloading an entry
@pytest.mark.asyncio
async def test_async_unload_entry(hass: HomeAssistant) -> None:
    """Test unloading the OpenAI entry."""
    mock_entry = MagicMock(spec=ConfigEntry)
    mock_entry.domain = DOMAIN
    mock_entry.title = "Test OpenAI Entry"
    mock_entry.entry_id = "test_entry_id"

    with patch.object(hass.config_entries, "async_unload_platforms", return_value=True):
        result = await async_unload_entry(hass, mock_entry)
        assert result is True


# Test image generation service error handling
@pytest.mark.asyncio
async def test_render_image_service_api_error(hass: HomeAssistant) -> None:
    """Test the image generation service when the API returns an error."""
    mock_entry = MagicMock(spec=ConfigEntry)
    mock_entry.data = {
        "api_key": "test_api_key",
        "base_url": OPENAI_BASE_URL,
    }
    mock_entry.domain = DOMAIN
    mock_entry.entry_id = "test_entry_id"

    # Mock the OpenAI client to raise an API error
    mock_openai_client = AsyncMock()
    mock_openai_client.images.generate = AsyncMock(
        side_effect=openai.OpenAIError("API error")
    )

    # Register the service
    await async_setup(hass, {})
    hass.config_entries.async_get_entry = MagicMock(return_value=mock_entry)
    mock_entry.runtime_data = mock_openai_client

    service_data = {
        "config_entry": mock_entry.entry_id,
        "prompt": PROMPT,
        "size": "1024x1024",
        "quality": "standard",
        "style": "vivid",
    }

    # Call the service and check if it raises a HomeAssistantError
    with pytest.raises(HomeAssistantError):
        await hass.services.async_call(
            DOMAIN,
            SERVICE_GENERATE_IMAGE,
            service_data,
            blocking=True,
            return_response=True,
        )


# Test memory functionality
@pytest.mark.asyncio
async def test_memory_functionality_enabled(hass: HomeAssistant) -> None:
    """Test setup entry for OpenAI with memory enabled and validate backend check."""
    mock_entry = MagicMock(spec=ConfigEntry)
    mock_entry.data = {
        "api_key": "test_api_key",
        "base_url": OPENAI_BASE_URL,
        "enable_memory": True,
    }
    mock_entry.options = {}
    mock_entry.domain = DOMAIN
    mock_entry.entry_id = "test_entry_id"
    mock_entry.setup_lock = MagicMock()
    mock_entry.title = "Test Title"

    with patch(
        "homeassistant.components.openai_conversation.letta_api.list_llm_backends",
        new=AsyncMock(),
    ) as mock_llm_backends:
        mock_llm_backends.return_value = MagicMock(status_code=200)

        result = await async_setup_entry(hass, mock_entry)
        assert result is True
        mock_llm_backends.assert_called_once()


# Test invalid API key
@pytest.mark.asyncio
async def test_setup_entry_invalid_api_key(hass: HomeAssistant) -> None:
    """Test setup entry failure due to an invalid API key."""
    mock_entry = MagicMock(spec=ConfigEntry)
    mock_entry.data = {
        "api_key": "invalid_api_key",
        "base_url": OPENAI_BASE_URL,
    }
    mock_entry.domain = DOMAIN
    mock_entry.options = {}

    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.body = "Invalid API key"

    with patch("openai.AsyncClient") as mock_openai_client:
        mock_openai_client().models.list = AsyncMock(
            side_effect=openai.AuthenticationError(
                "Invalid API key", response=mock_response, body=mock_response.body
            )
        )

        with pytest.raises(ConfigEntryNotReady):
            await async_setup_entry(hass, mock_entry)
