"""Multi-agent orchestration."""

from typing import Any, AsyncGenerator

from pydantic import BaseModel, Field

from mimohuman.core.agent import Agent
from mimohuman.core.conversation import Conversation
from mimohuman.core.provider import StreamEvent, StreamEventType


class AgentRole(BaseModel):
    """A named role that wraps an Agent."""

    name: str
    agent: Agent
    description: str = ""

    model_config = {"arbitrary_types_allowed": True}


class OrchestratorConfig(BaseModel):
    """Configuration for the Orchestrator."""

    roles: list[AgentRole] = Field(default_factory=list)
    max_turns: int = 20


class Orchestrator:
    """Coordinates multiple agents.

    For the initial scaffold, uses simple sequential routing:
    each agent processes the output of the previous one.
    """

    def __init__(self, config: OrchestratorConfig) -> None:
        self.config = config
        self._agents: dict[str, Agent] = {r.name: r.agent for r in config.roles}

    def get_agent(self, name: str) -> Agent | None:
        """Get an agent by role name."""
        return self._agents.get(name)

    async def run(
        self,
        user_input: str,
        conversation: Conversation | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Run the orchestration for a user input.

        Currently uses simple sequential routing through all roles.
        Future versions can add router-model-based agent selection.
        """
        conv = conversation or Conversation()
        current_input = user_input

        for turn, role in enumerate(self.config.roles):
            if turn >= self.config.max_turns:
                break

            yield StreamEvent(
                type=StreamEventType.AGENT_START,
                data={"agent": role.name, "turn": turn},
            )

            agent = role.agent
            async for event in agent.run(current_input, conv):
                yield event

            # The last assistant message becomes input for the next agent
            for msg in reversed(conv.messages):
                if msg.role.value == "assistant":
                    current_input = msg.content
                    break

        yield StreamEvent(
            type=StreamEventType.AGENT_END,
            data={"agent": "orchestrator", "turns_completed": turn + 1},
        )
