# API Reference

Timestep MVP provides core functions for agent execution.

## run_agent

Run an agent with custom execution loop.

### Function Signature

```python
async def run_agent(
    agent: Agent,
    messages: List[ChatMessage],
    session: Session,
    stream: bool = False
) -> AsyncIterator[Dict[str, Any]]
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `agent` | `Agent` | The agent to run |
| `messages` | `List[ChatMessage]` | Initial messages for the conversation |
| `session` | `Session` | Session for conversation persistence |
| `stream` | `bool` | Whether to stream responses (default: False) |

### Returns

Returns an async iterator of execution events:
- `{"type": "content_delta", "content": str}` - Streaming content chunks
- `{"type": "tool_call", "tool": str, "args": dict}` - Tool call event
- `{"type": "tool_result", "tool": str, "result": dict}` - Tool result event
- `{"type": "tool_error", "tool": str, "error": str}` - Tool error event
- `{"type": "message", "content": str}` - Final message
- `{"type": "error", "error": str}` - Execution error

### Example

```python
from timestep import Agent, FileSession, run_agent

agent = Agent(
    name="Assistant",
    model="gpt-4o",
    instructions="You are a helpful assistant.",
    tools=[],
)

session = FileSession(
    agent_name=agent.name,
    conversation_id="my-conversation",
    agent_instructions=agent.instructions,
)

# Get raw events by passing result_processor=None
messages = [{"role": "user", "content": "Hello!"}]
events = run_agent(agent, messages, session, stream=False, result_processor=None)

async for event in events:
    print(event)
```

## default_result_processor

Default processor that collects all events and returns final result.

### Function Signature

```python
async def default_result_processor(
    events: AsyncIterator[Dict[str, Any]]
) -> Dict[str, Any]
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `events` | `AsyncIterator[Dict[str, Any]]` | Stream of execution events |

### Returns

Returns a dictionary with:
- `messages`: List of message contents
- `tool_calls`: List of tool calls made
- `errors`: List of errors encountered

### Example

```python
from timestep import Agent, FileSession, run_agent

agent = Agent(name="Assistant", model="gpt-4o", instructions="Helpful assistant", tools=[])
session = FileSession(agent_name=agent.name, conversation_id="test", agent_instructions=agent.instructions)

messages = [{"role": "user", "content": "Hello!"}]
result = await run_agent(agent, messages, session, stream=False)

print(result["messages"])
print(result["tool_calls"])
```

## Agent

Agent configuration for multi-agent system.

### Class Definition

```python
@dataclass
class Agent:
    name: str
    model: str
    instructions: str
    tools: List[Tool] = field(default_factory=list)
    handoffs: List["Agent"] = field(default_factory=list)
    guardrails: List[Any] = field(default_factory=list)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Agent name |
| `model` | `str` | OpenAI model name (e.g., "gpt-4o") |
| `instructions` | `str` | System instructions for the agent |
| `tools` | `List[Tool]` | List of callable tool functions |
| `handoffs` | `List[Agent]` | List of agents this agent can handoff to |
| `guardrails` | `List` | List of InputGuardrail/OutputGuardrail instances |

### Example

```python
from timestep import Agent, InputGuardrail

async def my_tool(args: dict) -> dict:
    return {"result": "done"}

async def my_guardrail(tool_name: str, args: dict):
    from timestep import GuardrailResult
    return GuardrailResult.proceed()

agent = Agent(
    name="My Agent",
    model="gpt-4o",
    instructions="You are helpful.",
    tools=[my_tool],
    guardrails=[InputGuardrail(my_guardrail)],
)
```

## FileSession

File-based session implementation using JSONL storage.

### Class Definition

```python
class FileSession:
    def __init__(
        self,
        agent_name: str,
        conversation_id: str,
        agent_instructions: str | None = None,
        storage_dir: str = "conversations",
    )
    
    @property
    def session_id(self) -> str
    
    async def get_items(self, limit: int | None = None) -> list[ChatMessage]
    async def add_items(self, items: list[ChatMessage]) -> None
    async def pop_item(self) -> ChatMessage | None
    async def clear_session(self) -> None
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `agent_name` | `str` | Name of the agent |
| `conversation_id` | `str` | Unique identifier for this conversation |
| `agent_instructions` | `str \| None` | Optional system instructions |
| `storage_dir` | `str` | Directory to store conversation files (default: "conversations") |

### Example

```python
from timestep import FileSession

session = FileSession(
    agent_name="Assistant",
    conversation_id="my-conversation",
    agent_instructions="You are helpful.",
)

# Get messages
messages = await session.get_items()

# Add messages
await session.add_items([{"role": "user", "content": "Hello"}])
```

## InputGuardrail

Input guardrail for tool execution.

### Class Definition

```python
class InputGuardrail:
    def __init__(
        self,
        handler: Callable[[str, Dict[str, Any]], Awaitable[GuardrailResult]]
    )
    
    async def check(self, tool_name: str, args: Dict[str, Any]) -> GuardrailResult
```

### Example

```python
from timestep import InputGuardrail, GuardrailResult, GuardrailInterrupt

async def approval_guardrail(tool_name: str, args: dict) -> GuardrailResult:
    if needs_approval(args):
        raise GuardrailInterrupt("Need approval", tool_name, args)
    return GuardrailResult.proceed()

guardrail = InputGuardrail(approval_guardrail)
```

## OutputGuardrail

Output guardrail for tool execution.

### Class Definition

```python
class OutputGuardrail:
    def __init__(
        self,
        handler: Callable[[str, Dict[str, Any], Dict[str, Any]], Awaitable[GuardrailResult]]
    )
    
    async def check(
        self,
        tool_name: str,
        args: Dict[str, Any],
        result: Dict[str, Any]
    ) -> GuardrailResult
```

### Example

```python
from timestep import OutputGuardrail, GuardrailResult

async def sanitize_guardrail(tool_name: str, args: dict, result: dict) -> GuardrailResult:
    if "password" in str(result):
        sanitized = str(result).replace("password=secret", "password=***")
        return GuardrailResult.modify_result({"result": sanitized})
    return GuardrailResult.proceed()

guardrail = OutputGuardrail(sanitize_guardrail)
```

## GuardrailResult

Result from a guardrail check.

### Class Definition

```python
@dataclass
class GuardrailResult:
    proceed: bool
    modified_args: Optional[Dict[str, Any]] = None
    modified_result: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None
    
    @classmethod
    def block(cls, reason: str) -> "GuardrailResult"
    
    @classmethod
    def proceed(cls) -> "GuardrailResult"
    
    @classmethod
    def modify_args(cls, new_args: Dict[str, Any]) -> "GuardrailResult"
    
    @classmethod
    def modify_result(cls, new_result: Dict[str, Any]) -> "GuardrailResult"
```

## GuardrailInterrupt

Exception raised when guardrail requires human input.

### Class Definition

```python
class GuardrailInterrupt(GuardrailError):
    def __init__(self, prompt: str, tool_name: str, args: Dict[str, Any])
```

### Attributes

- `prompt`: Prompt to show to user
- `tool_name`: Name of the tool
- `args`: Tool arguments

## request_approval

Request approval from user. Blocks until answered.

### Function Signature

```python
async def request_approval(prompt: str) -> bool
```

### Example

```python
from timestep import request_approval

approved = await request_approval("Approve this action? (y/n): ")
if approved:
    # Proceed
    pass
```

## with_guardrails

Execute a tool handler with optional pre and post guardrails.

### Function Signature

```python
async def with_guardrails(
    tool_handler: Callable[[Dict[str, Any]], Any],
    tool_name: str,
    args: Dict[str, Any],
    pre_guardrails: Optional[List[Guardrail]] = None,
    post_guardrails: Optional[List[Guardrail]] = None,
) -> Dict[str, Any]
```

### Example

```python
from timestep import with_guardrails

async def my_tool(args: dict) -> dict:
    return {"result": "done"}

result = await with_guardrails(
    my_tool,
    tool_name="my_tool",
    args={"key": "value"},
    pre_guardrails=[pre_guardrail],
    post_guardrails=[post_guardrail],
)
```
