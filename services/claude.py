"""
Async Claude API client — singleton for use across all routers.

All Anthropic API calls in this codebase must go through this module.
Using a single AsyncAnthropic instance ensures connection pooling and
avoids the event-loop blocking caused by the legacy sync client.
"""

import os
import anthropic

# Single shared async client. Initialized at import time; the API key is
# read from the environment (set via .env or container secret).
_api_key = os.getenv("ANTHROPIC_API_KEY")

async_client: anthropic.AsyncAnthropic = anthropic.AsyncAnthropic(api_key=_api_key)
