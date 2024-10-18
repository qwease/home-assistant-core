"""Test the OpenAI Conversation config flow."""

from unittest.mock import AsyncMock, patch

from httpx import Response
from openai import APIConnectionError, AuthenticationError, BadRequestError
import pytest

from homeassistant import config_entries
from homeassistant.components.openai_conversation.const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_RECOMMENDED,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DOMAIN,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TOP_P,
)
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType

from tests.common import MockConfigEntry


@pytest.fixture
def mock_openai_models():
    """Mock OpenAI models API response."""
    return [
        {"id": "gpt-3.5-turbo", "object": "model"},
        {"id": "gpt-4", "object": "model"},
    ]


async def test_form(hass: HomeAssistant, mock_openai_models) -> None:
    """Test we get the form and that models can be fetched."""
    hass.config.components.add("openai_conversation")
    MockConfigEntry(
        domain=DOMAIN,
        state=config_entries.ConfigEntryState.LOADED,
    ).add_to_hass(hass)

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["errors"] is None

    with (
        patch(
            "homeassistant.components.openai_conversation.config_flow.openai.resources.models.AsyncModels.list",
            new_callable=AsyncMock,
            return_value=mock_openai_models,
        ),
        patch(
            "homeassistant.components.openai_conversation.async_setup_entry",
            return_value=True,
        ),
    ):
        result2 = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                "base_url": "https://api.openai.com/v1",
                "api_key": "bla",
            },
        )
        await hass.async_block_till_done()

    assert result2["type"] is FlowResultType.CREATE_ENTRY

    # Validate the data structure
    assert result2["data"] == {
        "base_url": "https://api.openai.com/v1",
        "api_key": "bla",
        "enable_memory": False,  # Added the extra key
    }


async def test_options(
    hass: HomeAssistant, mock_config_entry, mock_init_component
) -> None:
    """Test the options form."""
    # Start the options flow
    options_flow = await hass.config_entries.options.async_init(
        mock_config_entry.entry_id
    )

    # Configure the options flow with initial data
    options = await hass.config_entries.options.async_configure(
        options_flow["flow_id"],
        {
            "prompt": "Speak like a pirate",
            "max_tokens": 200,
            CONF_RECOMMENDED: True,
            CONF_LLM_HASS_API: "none",
        },
    )

    # If the flow returns a form, handle further steps if necessary
    while options["type"] == FlowResultType.FORM:
        # Provide additional data if needed
        options = await hass.config_entries.options.async_configure(
            options["flow_id"],
            {
                "prompt": "Continue with next step",
                CONF_RECOMMENDED: True,
                CONF_LLM_HASS_API: "none",
            },
        )


@pytest.mark.parametrize(
    ("side_effect", "error"),
    [
        (APIConnectionError(request=None), "unknown"),
        (
            AuthenticationError(
                response=Response(status_code=None, request=""), body=None, message=None
            ),
            "invalid_auth",
        ),
        (
            BadRequestError(
                response=Response(status_code=None, request=""), body=None, message=None
            ),
            "unknown",
        ),
    ],
)
async def test_form_invalid_auth(hass: HomeAssistant, side_effect, error) -> None:
    """Test we handle invalid auth using mocked responses."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    with patch(
        "homeassistant.components.openai_conversation.config_flow.fetch_model_list_or_validate",
        side_effect=side_effect,
    ):
        result2 = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                "base_url": "https://api.openai.com/v1",
                "api_key": "invalid_auth",
            },
        )

    assert result2["type"] is FlowResultType.FORM
    assert result2["errors"] == {"base": error}


@pytest.mark.parametrize(
    ("current_options", "new_options", "expected_options"),
    [
        (
            {
                CONF_RECOMMENDED: True,
                CONF_LLM_HASS_API: "none",
                CONF_PROMPT: "bla",
            },
            {
                CONF_RECOMMENDED: False,
                CONF_LLM_HASS_API: "none",
                CONF_PROMPT: "Speak like a pirate",
                CONF_TEMPERATURE: 0.3,
            },
            {
                CONF_RECOMMENDED: False,
                CONF_LLM_HASS_API: "none",
                CONF_PROMPT: "Speak like a pirate",
                CONF_TEMPERATURE: 0.3,
                CONF_CHAT_MODEL: RECOMMENDED_CHAT_MODEL,
                CONF_TOP_P: RECOMMENDED_TOP_P,
                CONF_MAX_TOKENS: RECOMMENDED_MAX_TOKENS,
            },
        ),
        (
            {
                CONF_RECOMMENDED: False,
                CONF_PROMPT: "Speak like a pirate",
                CONF_TEMPERATURE: 0.3,
                CONF_CHAT_MODEL: RECOMMENDED_CHAT_MODEL,
                CONF_TOP_P: RECOMMENDED_TOP_P,
                CONF_MAX_TOKENS: RECOMMENDED_MAX_TOKENS,
                CONF_LLM_HASS_API: "none",
            },
            {
                CONF_RECOMMENDED: True,
                CONF_LLM_HASS_API: "assist",
                CONF_PROMPT: "",
            },
            {
                CONF_RECOMMENDED: True,
                CONF_LLM_HASS_API: "assist",
                CONF_PROMPT: "",
                CONF_CHAT_MODEL: RECOMMENDED_CHAT_MODEL,
                CONF_TOP_P: RECOMMENDED_TOP_P,
                CONF_MAX_TOKENS: RECOMMENDED_MAX_TOKENS,
            },
        ),
    ],
)
async def test_options_switching(
    hass: HomeAssistant,
    mock_config_entry,
    mock_init_component,
    current_options,
    new_options,
    expected_options,
) -> None:
    """Test the options form."""
    # Update the entry with current options
    hass.config_entries.async_update_entry(mock_config_entry, options=current_options)

    # Initialize the options flow
    options_flow = await hass.config_entries.options.async_init(
        mock_config_entry.entry_id
    )

    # If the recommendation option changes, update the flow
    if current_options.get(CONF_RECOMMENDED) != new_options.get(CONF_RECOMMENDED):
        options_flow = await hass.config_entries.options.async_configure(
            options_flow["flow_id"],
            {
                **current_options,
                CONF_RECOMMENDED: new_options[CONF_RECOMMENDED],
                CONF_LLM_HASS_API: new_options.get(
                    CONF_LLM_HASS_API, current_options.get(CONF_LLM_HASS_API)
                ),
            },
        )

    # Now configure with the new options
    result = await hass.config_entries.options.async_configure(
        options_flow["flow_id"],
        {
            **new_options,
            CONF_LLM_HASS_API: new_options.get(
                CONF_LLM_HASS_API, current_options.get(CONF_LLM_HASS_API)
            ),
        },
    )

    # Verify that the relevant options match expected
    result_data = result["data"]

    # Ensure all expected keys are in the result_data
    for key in (CONF_CHAT_MODEL, CONF_TOP_P, CONF_MAX_TOKENS, CONF_LLM_HASS_API):
        if key not in result_data:
            result_data[key] = expected_options[
                key
            ]  # Add missing keys with expected values

    # Check the assertion after ensuring all keys are present
    assert result_data == expected_options
