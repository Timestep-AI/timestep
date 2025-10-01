"""MCP Server Proxy for fetching tools from MCP servers and providing built-in tools."""

from typing import Any, Callable, Optional
from dataclasses import dataclass
from mcp import ClientSession
from mcp.client.sse import sse_client
from agents import function_tool


@dataclass
class BuiltInMcpTool:
    """Built-in MCP tool definition."""
    name: str
    description: str
    input_schema: dict[str, Any]
    execute: Callable[[dict[str, Any]], str]
    needs_approval: bool = False


# Built-in tools available in the local MCP server
BUILT_IN_TOOLS: list[BuiltInMcpTool] = [
    BuiltInMcpTool(
        name="get_weather",
        description="Get the weather for a given city",
        input_schema={
            "type": "object",
            "properties": {
                "city": {"type": "string"},
            },
            "required": ["city"],
        },
        execute=lambda params: f"The weather in {params['city']} is sunny.",
        needs_approval=False,
    ),
]


async def fetch_mcp_tools(
    server_url: Optional[str],
    include_built_in: bool = False,
    require_approval: Optional[dict[str, dict[str, list[str]]]] = None,
) -> list[Any]:
    """
    Fetches tools from an MCP server and/or includes built-in tools.

    Args:
        server_url: The URL of the MCP server (e.g., 'https://gitmcp.io/timestep-ai/timestep'),
                   or None for built-in onl
        include_built_in: Whether to include built-in tools from the local MCP server
        require_approval: Configuration for which tools require approval
                         Example: {
                             "never": {"toolNames": ["tool1", "tool2"]},
                             "always": {"toolNames": ["tool3"]},
                         }

    Returns:
        Array of agent tools
    """
    agent_tools: list[Any] = []

    # Fetch remote tools if server_url is provided
    if server_url:
        try:
            async with sse_client(server_url, timeout=30.0) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    # List all available tools from the MCP server
                    tools_response = await session.list_tools()
                    mcp_tools = tools_response.tools if tools_response.tools else []

                    # Close connection after fetching tools
                    # (Connection will be reopened for each tool execution)
        except Exception as e:
            print(f"Warning: Could not connect to MCP server {server_url}: {e}")
            import traceback
            traceback.print_exc()
            print("Continuing without remote tools...")
            mcp_tools = []

        # Convert MCP tools to agent function tools
        for mcp_tool in mcp_tools:
            # Determine if this tool needs approval
            never_require = (
                mcp_tool.name in require_approval.get("never", {}).get("toolNames", [])
                if require_approval
                else False
            )
            always_require = (
                mcp_tool.name in require_approval.get("always", {}).get("toolNames", [])
                if require_approval
                else False
            )
            needs_approval = always_require or (not never_require and False)

            # Create a closure to capture the tool name and server URL
            def make_execute_fn(tool_name: str, srv_url: str, param_names: list[str]):
                async def execute_tool(*args, **kwargs) -> str:
                    """Execute a remote MCP tool."""
                    # Convert positional args to kwargs based on parameter names
                    if args and not kwargs:
                        kwargs = {param_names[i]: args[i] for i in range(min(len(args), len(param_names)))}

                    async with sse_client(srv_url, timeout=30.0) as (read, write):
                        async with ClientSession(read, write) as exec_session:
                            await exec_session.initialize()
                            result = await exec_session.call_tool(tool_name, kwargs)

                            if result.isError:
                                raise Exception(f"MCP tool error: {result.content}")

                            # Extract text content from result
                            text_content = "\n".join(
                                item.text for item in result.content if hasattr(item, 'text')
                            )
                            return text_content or str(result.content)
                return execute_tool

            # Create the function tool with proper annotations
            # For MCP tools, we'll create a wrapper function that accepts any parameters
            # and uses inspect to build the right signature dynamically

            # Build function signature from input schema
            import inspect
            from typing import Optional as Opt

            # Parse the input schema to create proper parameters
            properties = mcp_tool.inputSchema.get("properties", {}) if hasattr(mcp_tool, 'inputSchema') else {}
            required = mcp_tool.inputSchema.get("required", []) if hasattr(mcp_tool, 'inputSchema') else []

            # Get parameter names in order
            param_names = list(properties.keys())

            # Create the execute function with parameter names
            execute_fn = make_execute_fn(mcp_tool.name, server_url, param_names)
            execute_fn.__name__ = mcp_tool.name
            execute_fn.__doc__ = mcp_tool.description or ""

            # Create parameter list
            params = [inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] if False else []
            for param_name, param_schema in properties.items():
                param_type = str  # Default to str
                if param_schema.get("type") == "integer":
                    param_type = int
                elif param_schema.get("type") == "number":
                    param_type = float
                elif param_schema.get("type") == "boolean":
                    param_type = bool

                if param_name not in required:
                    # Optional parameter
                    params.append(inspect.Parameter(
                        param_name,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=None,
                        annotation=Opt[param_type]
                    ))
                else:
                    # Required parameter
                    params.append(inspect.Parameter(
                        param_name,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=param_type
                    ))

            # Create new signature
            new_sig = inspect.Signature(params, return_annotation=str)
            execute_fn.__signature__ = new_sig

            # Use function_tool decorator
            try:
                tool_fn = function_tool(execute_fn)
                # Note: Python agents library doesn't have built-in approval support like TS version
                # So we mark it for future reference but can't enforce it in the same way
                tool_fn._needs_approval = needs_approval
                agent_tools.append(tool_fn)
            except Exception as e:
                print(f"Warning: Could not create tool {mcp_tool.name}: {e}")
                continue

    # Add built-in tools if requested
    if include_built_in:
        for built_in_tool in BUILT_IN_TOOLS:
            # Determine if this tool needs approval (apply requireApproval config)
            never_require = (
                built_in_tool.name in require_approval.get("never", {}).get("toolNames", [])
                if require_approval
                else False
            )
            always_require = (
                built_in_tool.name in require_approval.get("always", {}).get("toolNames", [])
                if require_approval
                else False
            )
            needs_approval = always_require or (not never_require and built_in_tool.needs_approval)

            # Create the function tool with explicit parameter
            def make_builtin_fn(tool: BuiltInMcpTool):
                def execute_tool(city: str) -> str:
                    """Execute the built-in tool."""
                    return tool.execute({"city": city})
                execute_tool.__name__ = tool.name
                execute_tool.__doc__ = tool.description
                return execute_tool

            execute_fn = make_builtin_fn(built_in_tool)
            try:
                tool_fn = function_tool(execute_fn)
                tool_fn._needs_approval = needs_approval
                agent_tools.append(tool_fn)
            except Exception as e:
                print(f"Warning: Could not create built-in tool {built_in_tool.name}: {e}")
                continue

    return agent_tools
