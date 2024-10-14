"""Conversation support for OpenAI."""

from collections.abc import Callable
import json
from typing import Any, Literal

import openai
from openai._types import NOT_GIVEN
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import Function
from openai.types.shared_params import FunctionDefinition
import voluptuous as vol
from voluptuous_openapi import convert

from homeassistant.components import assist_pipeline, conversation
from homeassistant.components.conversation import trace
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError, TemplateError
from homeassistant.helpers import device_registry as dr, intent, llm, template
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.util import ulid

from . import OpenAIConfigEntry
from .const import (
    CONF_AGENT,
    CONF_CHAT_MODEL,
    CONF_ENABLE_MEMORY,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DOMAIN,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
)

# Max number of back and forth with the LLM to generate a response
MAX_TOOL_ITERATIONS = 10


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: OpenAIConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    agent = OpenAIConversationEntity(config_entry)
    async_add_entities([agent])


def _format_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> ChatCompletionToolParam:
    """Format tool specification."""
    tool_spec = FunctionDefinition(
        name=tool.name,
        parameters=convert(tool.parameters, custom_serializer=custom_serializer),
    )
    if tool.description:
        tool_spec["description"] = tool.description
    return ChatCompletionToolParam(type="function", function=tool_spec)


class OpenAIConversationEntity(
    conversation.ConversationEntity, conversation.AbstractConversationAgent
):
    """OpenAI conversation agent."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, entry: OpenAIConfigEntry) -> None:
        """Initialize the agent."""
        self.entry = entry
        self.history: dict[str, list[ChatCompletionMessageParam]] = {}
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="OpenAI",
            model="ChatGPT",
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        if self.entry.options.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        assist_pipeline.async_migrate_engine(
            self.hass, "conversation", self.entry.entry_id, self.entity_id
        )
        conversation.async_set_agent(self.hass, self.entry, self)
        self.entry.async_on_unload(
            self.entry.add_update_listener(self._async_entry_update_listener)
        )

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    async def async_process(  # noqa: C901
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Process a sentence."""
        is_memory_enabled = self.entry.data.get(CONF_ENABLE_MEMORY, False)
        options = self.entry.options
        intent_response = intent.IntentResponse(language=user_input.language)
        llm_api: llm.APIInstance | None = None
        tools: list[ChatCompletionToolParam] | None = None
        user_name: str | None = None
        llm_context = llm.LLMContext(
            platform=DOMAIN,
            context=user_input.context,
            user_prompt=user_input.text,
            language=user_input.language,
            assistant=conversation.DOMAIN,
            device_id=user_input.device_id,
        )

        if options.get(CONF_LLM_HASS_API):
            try:
                llm_api = await llm.async_get_api(
                    self.hass,
                    options[CONF_LLM_HASS_API],
                    llm_context,
                )
            except HomeAssistantError as err:
                LOGGER.error("Error getting LLM API: %s", err)
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN,
                    f"Error preparing LLM API: {err}",
                )
                return conversation.ConversationResult(
                    response=intent_response, conversation_id=user_input.conversation_id
                )
            tools = [
                _format_tool(tool, llm_api.custom_serializer) for tool in llm_api.tools
            ]

        if user_input.conversation_id is None:
            conversation_id = ulid.ulid_now()
            messages = []

        elif user_input.conversation_id in self.history:
            conversation_id = user_input.conversation_id
            messages = self.history[conversation_id]

        else:
            # Conversation IDs are ULIDs. We generate a new one if not provided.
            # If an old OLID is passed in, we will generate a new one to indicate
            # a new conversation was started. If the user picks their own, they
            # want to track a conversation and we respect it.
            try:
                ulid.ulid_to_bytes(user_input.conversation_id)
                conversation_id = ulid.ulid_now()
            except ValueError:
                conversation_id = user_input.conversation_id

            messages = []

        if (
            user_input.context
            and user_input.context.user_id
            and (
                user := await self.hass.auth.async_get_user(user_input.context.user_id)
            )
        ):
            user_name = user.name

        try:
            prompt_parts = [
                template.Template(
                    llm.BASE_PROMPT
                    + options.get(CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT),
                    self.hass,
                ).async_render(
                    {
                        "ha_name": self.hass.config.location_name,
                        "user_name": user_name,
                        "llm_context": llm_context,
                    },
                    parse_result=False,
                )
            ]

        except TemplateError as err:
            LOGGER.error("Error rendering prompt: %s", err)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Sorry, I had a problem with my template: {err}",
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )

        if llm_api:
            prompt_parts.append(llm_api.api_prompt)

        prompt = "\n".join(prompt_parts)

        # Create a copy of the variable because we attach it to the trace
        messages = [
            ChatCompletionSystemMessageParam(role="system", content=prompt),
            *messages[1:],
            ChatCompletionUserMessageParam(role="user", content=user_input.text),
        ]

        LOGGER.info("Prompt: %s", messages)
        LOGGER.info("Tools: %s", tools)
        trace.async_conversation_trace_append(
            trace.ConversationTraceEventType.AGENT_DETAIL,
            {"messages": messages, "tools": llm_api.tools if llm_api else None},
        )

        # letta api kicks in
        if not is_memory_enabled:
            client = self.entry.runtime_data

            # To prevent infinite loops, we limit the number of iterations
            for _iteration in range(MAX_TOOL_ITERATIONS):
                try:
                    result = await client.chat.completions.create(
                        model=options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
                        messages=messages,
                        tools=tools or NOT_GIVEN,
                        max_tokens=options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
                        top_p=options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
                        temperature=options.get(
                            CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE
                        ),
                        user=conversation_id,
                    )
                except openai.OpenAIError as err:
                    intent_response = intent.IntentResponse(
                        language=user_input.language
                    )
                    intent_response.async_set_error(
                        intent.IntentResponseErrorCode.UNKNOWN,
                        f"Sorry, I had a problem talking to OpenAI: {err}",
                    )
                    return conversation.ConversationResult(
                        response=intent_response, conversation_id=conversation_id
                    )

                LOGGER.debug("Response %s", result)
                response = result.choices[0].message

                def message_convert(
                    message: ChatCompletionMessage,
                ) -> ChatCompletionMessageParam:
                    """Convert from class to TypedDict."""
                    tool_calls: list[ChatCompletionMessageToolCallParam] = []
                    if message.tool_calls:
                        tool_calls = [
                            ChatCompletionMessageToolCallParam(
                                id=tool_call.id,
                                function=Function(
                                    arguments=tool_call.function.arguments,
                                    name=tool_call.function.name,
                                ),
                                type=tool_call.type,
                            )
                            for tool_call in message.tool_calls
                        ]
                    param = ChatCompletionAssistantMessageParam(
                        role=message.role,
                        content=message.content,
                    )
                    if tool_calls:
                        param["tool_calls"] = tool_calls
                    return param

                messages.append(message_convert(response))
                tool_calls = response.tool_calls

                if not tool_calls or not llm_api:
                    break

                for tool_call in tool_calls:
                    tool_input = llm.ToolInput(
                        tool_name=tool_call.function.name,
                        tool_args=json.loads(tool_call.function.arguments),
                    )
                    LOGGER.debug(
                        "Tool call: %s(%s)", tool_input.tool_name, tool_input.tool_args
                    )

                    try:
                        tool_response = await llm_api.async_call_tool(tool_input)
                    except (HomeAssistantError, vol.Invalid) as e:
                        tool_response = {"error": type(e).__name__}
                        if str(e):
                            tool_response["error_text"] = str(e)

                    LOGGER.debug("Tool response: %s", tool_response)
                    messages.append(
                        ChatCompletionToolMessageParam(
                            role="tool",
                            tool_call_id=tool_call.id,
                            content=json.dumps(tool_response),
                        )
                    )

            self.history[conversation_id] = messages

            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_speech(response.content or "")
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )

        from .const import CONF_BASE_URL  # pylint: disable=import-outside-toplevel  # noqa: I001
        from .letta_api import list_agents, send_message  # pylint: disable=import-outside-toplevel  # noqa: I001

        agent_name = options.get(CONF_AGENT, "")
        agent_id = ""
        user_id = ""  # leave blank
        modified_prompt = (
            messages[0].get("content") + " Following dict is the tools you can use: "  # type: ignore[operator]
        )
        if tools is not None:
            for tool in tools:
                modified_prompt += str(json.dumps(tool))
            available_tools = [tool["function"]["name"] for tool in tools]
        modified_prompt += f'\n The user prompt is: "{messages[1].get("content")}"'  # \nIf your tool call failed, DO NOT complain, since the tool may be specifically for Home Assistant System, just ignore it.

        data = {
            "messages": [{"role": "user", "text": modified_prompt}],
            "return_message_object": True,
        }
        LOGGER.info("Data")
        LOGGER.info(data)
        list_agents_response = await list_agents(
            self.hass,
            self.entry.data.get(CONF_BASE_URL, None),
            user_id,
        )
        if list_agents_response.status_code == 200:
            # LOGGER.info(f"list_agents_response.text:{list_agents_response.text}")  # noqa: G004
            for agent in json.loads(list_agents_response.text):
                if agent.get("name", "") == agent_name:
                    user_id = agent.get("user_id", "")
                    if user_id is None:
                        user_id = ""
                    agent_id = agent.get("id", "")
        if agent_id == "":
            # agent was deleted and disappeared
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                "Required Agent Disappeared. Throwing an error",
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )
        speech = ""

        # To prevent infinite loops, we limit the number of iterations
        for _iteration in range(MAX_TOOL_ITERATIONS):
            try:
                result = await send_message(  # type: ignore[assignment]
                    self.hass,
                    self.entry.data.get(CONF_BASE_URL, None),
                    user_id,
                    agent_id,
                    data,
                    60,  # It needs time...
                )
                if result.status_code == 200:  # type: ignore[attr-defined]
                    response_json = json.loads(result.text)  # type: ignore[attr-defined]
                    LOGGER.info(response_json)
            except ConnectionRefusedError as err:
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN,
                    f"Sorry, I had a problem talking to Local LLM: {err}",
                )
                return conversation.ConversationResult(
                    response=intent_response, conversation_id=conversation_id
                )
            called_tools = []
            message_list = response_json.get("messages")
            LOGGER.info(len(message_list))
            shall_break = False
            for message in message_list:
                LOGGER.info("LOGGER.info(message)")
                LOGGER.info(message.keys())
                tool_calls = message.get("tool_calls")
                if not llm_api:
                    shall_break = True
                if tool_calls is not None:
                    for tool_call in tool_calls:
                        LOGGER.info(tool_call)
                        tool_name = tool_call["function"]["name"]  # type: ignore[index]
                        called_tools.append(tool_call)

                        if (
                            available_tools is not None
                            and tool_name not in available_tools
                        ):
                            shall_break = True
                            break

                        tool_args: dict = json.loads(tool_call["function"]["arguments"])  # type: ignore[index]

                        if "request_heartbeat" in tool_args:
                            del tool_args["request_heartbeat"]

                        tool_input = llm.ToolInput(
                            tool_name=tool_name,
                            tool_args=tool_args,
                        )
                        LOGGER.info(
                            "Tool call: %s(%s)",
                            tool_input.tool_name,
                            tool_input.tool_args,
                        )

                        try:
                            tool_response = await llm_api.async_call_tool(tool_input)  # type: ignore[union-attr]
                        except (HomeAssistantError, vol.Invalid) as e:
                            tool_response = {"error": type(e).__name__}
                            if str(e):
                                tool_response["error_text"] = str(e)
                        if tool_response["response_type"] == "action_done":
                            speech = "Action Performed"
                        LOGGER.info("Tool response: %s", tool_response)
            if shall_break:
                break
        if speech == "":
            # Find the first appearance of send_message()
            LOGGER.info(called_tools)
            for tool in called_tools:  # type: ignore[assignment]
                LOGGER.info(tool)
                tool_name = tool["function"]["name"]
                LOGGER.info(tool_name)
                if tool_name == "send_message":
                    speech = json.loads(tool["function"]["arguments"])["message"]  # type: ignore[typeddict-item]
                    LOGGER.info(speech)
                    break

        LOGGER.info("Speech")
        LOGGER.info(speech)
        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(speech or "Agent has nothing to say.")
        return conversation.ConversationResult(
            response=intent_response, conversation_id=conversation_id
        )

    async def _async_entry_update_listener(
        self, hass: HomeAssistant, entry: ConfigEntry
    ) -> None:
        """Handle options update."""
        # Reload as we update device info + entity name + supported features
        await hass.config_entries.async_reload(entry.entry_id)
