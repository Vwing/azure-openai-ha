"""Constants for the Azure OpenAI Conversation Integration."""

import logging

DOMAIN = "azure_openai_conversation"
LOGGER = logging.getLogger(__package__)

CONF_RECOMMENDED = "recommended"
CONF_PROMPT = "prompt"
CONF_CHAT_MODEL = "chat_model"
CONF_MAX_TOKENS = "max_tokens"
CONF_TOP_P = "top_p"
CONF_TEMPERATURE = "temperature"
CONF_REASONING_EFFORT = "reasoning_effort"
CONF_WEB_SEARCH = "web_search"
CONF_WEB_SEARCH_CITY = "web_search_city"
CONF_WEB_SEARCH_CONTEXT_SIZE = "web_search_context_size"
CONF_WEB_SEARCH_COUNTRY = "web_search_country"
CONF_WEB_SEARCH_REGION = "web_search_region"
CONF_WEB_SEARCH_TIMEZONE = "web_search_timezone"
CONF_WEB_SEARCH_USER_LOCATION = "web_search_user_location"
# Custom configuration entries
CONF_API_BASE = "api_base"
CONF_API_VERSION = "api_version"

RECOMMENDED_CHAT_MODEL = "gpt-4o-mini"
RECOMMENDED_MAX_TOKENS = 150
RECOMMENDED_TOP_P = 1.0
RECOMMENDED_TEMPERATURE = 1.0
RECOMMENDED_REASONING_EFFORT = "medium"
RECOMMENDED_WEB_SEARCH_CONTEXT_SIZE = 5000

DEFAULT_CHAT_MODEL = "gpt-3.5-turbo"
DEFAULT_TOP_P = 1
DEFAULT_TEMPERATURE = 0.5
DEFAULT_PROMPT = """This smart home is controlled by Home Assistant.

An overview of the areas and the devices in this smart home:
{%- for area in areas() %}
  {%- set area_info = namespace(printed=false) %}
  {%- for device in area_devices(area) -%}
    {%- if not device_attr(device, "disabled_by") and not device_attr(device, "entry_type") and device_attr(device, "name") %}
      {%- if not area_info.printed %}

{{ area_name(area) }}:
        {%- set area_info.printed = true %}
      {%- endif %}
- {{ device_attr(device, "name") }}{% if device_attr(device, "model") and (device_attr(device, "model") | string) not in (device_attr(device, "name") | string) %} ({{ device_attr(device, "model") }}){% endif %}
    {%- endif %}
  {%- endfor %}
{%- endfor %}

Answer the user's questions about the world truthfully.

If the user wants to control a device, reject the request and suggest using the Home Assistant app.
"""
