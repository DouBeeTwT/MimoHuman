"""Custom exception hierarchy for MimoHuman."""


class MimoHumanError(Exception):
    """Base exception for all MimoHuman errors."""

    pass


class ConfigurationError(MimoHumanError):
    """Invalid configuration."""

    pass


class ProviderError(MimoHumanError):
    """Base for provider-related errors."""

    pass


class ProviderAuthenticationError(ProviderError):
    """Authentication failed with the provider."""

    pass


class ProviderRateLimitError(ProviderError):
    """Rate limit exceeded."""

    pass


class ProviderAPIError(ProviderError):
    """Generic API error from provider."""

    pass


class ToolError(MimoHumanError):
    """Base for tool-related errors."""

    pass


class ToolNotFoundError(ToolError):
    """Requested tool is not registered."""

    pass


class ToolExecutionError(ToolError):
    """Tool execution failed."""

    pass


class AgentError(MimoHumanError):
    """Base for agent-related errors."""

    pass


class MaxToolRoundsExceeded(AgentError):
    """Agent exceeded the maximum number of tool-calling rounds."""

    pass


class ConversationLimitError(AgentError):
    """Conversation exceeded configured limits."""

    pass
