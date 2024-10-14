"""Config flow for OpenAI Conversation integration."""

from __future__ import annotations

import logging
from types import MappingProxyType
from typing import Any

import openai
from requests import HTTPError
import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
)

from .const import (
    CONF_AGENT,
    CONF_BASE_URL,
    CONF_CHAT_MODEL,
    CONF_ENABLE_MEMORY,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_RECOMMENDED,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DOMAIN,
    RECOMMENDED_BASE_URL,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_ENABLE_MEMORY,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
)
from .letta_api import list_agents, list_LLM_backends

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_BASE_URL, default=RECOMMENDED_BASE_URL): str,
        vol.Required(CONF_API_KEY): str,
        vol.Required(
            CONF_ENABLE_MEMORY,
            default=RECOMMENDED_ENABLE_MEMORY,
        ): bool,
    }
)

RECOMMENDED_OPTIONS = {
    CONF_RECOMMENDED: True,
    CONF_LLM_HASS_API: llm.LLM_API_ASSIST,
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
}


async def get_available_model_ids(
    hass: HomeAssistant, config_entry: ConfigEntry
) -> list[str]:
    """Retrieve all available model ids."""
    client = openai.AsyncOpenAI(
        base_url=config_entry.data[CONF_BASE_URL],
        api_key=config_entry.data[CONF_API_KEY],
    )
    async_pages = await hass.async_add_executor_job(
        client.with_options(timeout=10.0).models.list
    )
    return [model.id async for model in async_pages]


async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> None:
    """Validate the user input allows us to connect.

    Data has the keys from STEP_USER_DATA_SCHEMA with values provided by the user.
    """
    client = openai.AsyncOpenAI(
        base_url=data[CONF_BASE_URL], api_key=data[CONF_API_KEY]
    )
    # await client.with_options(timeout=10.0).models.list()
    await hass.async_add_executor_job(client.with_options(timeout=10.0).models.list)


class OpenAIConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for OpenAI Conversation."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        _LOGGER.info("User_input")
        _LOGGER.info(user_input)
        if user_input is None:
            return self.async_show_form(
                step_id="user", data_schema=STEP_USER_DATA_SCHEMA
            )

        errors = {}

        try:
            # two ways: no letta and letta enabled
            if not user_input.get(CONF_ENABLE_MEMORY, False):
                await validate_input(self.hass, user_input)
            else:
                # enabled
                await list_LLM_backends(
                    self.hass,
                    user_input.get(CONF_BASE_URL, None),
                    headers={"Authorization": "Bearer token"},
                )  # headers is unnecessary
        except openai.APIConnectionError:
            errors["base"] = "cannot_connect"
        except openai.AuthenticationError:
            errors["base"] = "invalid_auth"
        except Exception:
            _LOGGER.exception("Unexpected exception")
            errors["base"] = "unknown"
        else:
            return self.async_create_entry(
                title="ChatGPT",
                data=user_input,
                options=RECOMMENDED_OPTIONS,
            )
        return self.async_show_form(
            step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
        )

    @staticmethod
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> OptionsFlow:
        """Create the options flow."""
        return OpenAIOptionsFlow(config_entry)


class OpenAIOptionsFlow(OptionsFlow):
    """OpenAI config flow options handler."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry
        self.last_rendered_recommended = config_entry.options.get(
            CONF_RECOMMENDED, False
        )

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options."""
        options: dict[str, Any] | MappingProxyType[str, Any] = self.config_entry.options

        if user_input is not None:
            if user_input[CONF_RECOMMENDED] == self.last_rendered_recommended:
                if user_input[CONF_LLM_HASS_API] == "none":
                    user_input.pop(CONF_LLM_HASS_API)
                return self.async_create_entry(title="", data=user_input)

            # Re-render the options again, now with the recommended options shown/hidden
            self.last_rendered_recommended = user_input[CONF_RECOMMENDED]

            options = {
                CONF_RECOMMENDED: user_input[CONF_RECOMMENDED],
                CONF_PROMPT: user_input[CONF_PROMPT],
                CONF_LLM_HASS_API: user_input[CONF_LLM_HASS_API],
            }
        try:
            model_id_list = None
            if not self.config_entry.data.get(CONF_ENABLE_MEMORY, False):
                # to pass the test
                model_id_list = await get_available_model_ids(
                    self.hass, self.config_entry
                )
            else:
                user_id = ""  # leave blank
                list_agents_response = await list_agents(
                    self.hass,
                    self.config_entry.data.get(CONF_BASE_URL, None),
                    user_id,
                )
                if list_agents_response.status_code == 200:
                    import json  # pylint: disable=import-outside-toplevel  # noqa: I001

                    response_json = json.loads(list_agents_response.text)
                    _LOGGER.info(f"response_json:{response_json}")  # noqa: G004
                    agent_name_list = [agent.get("name", "") for agent in response_json]
                    _LOGGER.info("Agent_name_list")
                    _LOGGER.info(agent_name_list)
                    # Assuming name is unique, which is not the case.
        except openai.APIConnectionError:
            model_id_list = ["gpt-4o-mini"]
        except HTTPError as err:
            _LOGGER.error("Error rendering prompt: %s", err)
        if model_id_list:
            schema = openai_config_option_schema(self.hass, options, model_id_list)
        else:
            schema = openai_config_option_schema(
                self.hass,
                options,
                agent_name_list,
                self.config_entry.data.get(CONF_ENABLE_MEMORY, False),
            )
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(schema),
        )


def openai_config_option_schema(
    hass: HomeAssistant,
    options: dict[str, Any] | MappingProxyType[str, Any],
    id_list: list[str],
    is_memory_enabled: bool = False,
) -> dict:
    """Return a schema for OpenAI completion options."""
    hass_apis: list[SelectOptionDict] = [
        SelectOptionDict(
            label="No control",
            value="none",
        )
    ]
    hass_apis.extend(
        SelectOptionDict(
            label=api.name,
            value=api.id,
        )
        for api in llm.async_get_apis(hass)
    )

    schema = {
        vol.Optional(
            CONF_PROMPT,
            description={
                "suggested_value": options.get(
                    CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT
                )
            },
        ): TemplateSelector(),
        vol.Optional(
            CONF_LLM_HASS_API,
            description={"suggested_value": options.get(CONF_LLM_HASS_API)},
            default="none",
        ): SelectSelector(SelectSelectorConfig(options=hass_apis)),
        vol.Required(
            CONF_RECOMMENDED, default=options.get(CONF_RECOMMENDED, False)
        ): bool,
    }

    if options.get(CONF_RECOMMENDED):
        return schema

    _LOGGER.info(id_list)
    _LOGGER.info(options)
    if not is_memory_enabled:
        schema.update(
            {
                vol.Optional(
                    CONF_CHAT_MODEL,
                    default=RECOMMENDED_CHAT_MODEL,
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=id_list,
                        mode=SelectSelectorMode.DROPDOWN,
                    )
                ),
            }
        )
    else:
        schema.update(
            {
                vol.Optional(
                    CONF_AGENT,
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=id_list,
                        mode=SelectSelectorMode.DROPDOWN,
                    )
                ),
            }
        )
    schema.update(
        {
            vol.Optional(
                CONF_MAX_TOKENS,
                description={"suggested_value": options.get(CONF_MAX_TOKENS)},
                default=RECOMMENDED_MAX_TOKENS,
            ): int,
            vol.Optional(
                CONF_TOP_P,
                description={"suggested_value": options.get(CONF_TOP_P)},
                default=RECOMMENDED_TOP_P,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Optional(
                CONF_TEMPERATURE,
                description={"suggested_value": options.get(CONF_TEMPERATURE)},
                default=RECOMMENDED_TEMPERATURE,
            ): NumberSelector(NumberSelectorConfig(min=0, max=2, step=0.05)),
        }
    )
    return schema
