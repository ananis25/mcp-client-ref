# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mcp[cli] @ git+https://github.com/modelcontextprotocol/python-sdk@c14ec2e",
#     "uvicorn==0.34.0",
# ]
# ///


import httpx
import mcp.types as types
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ServerSimple")


@mcp.tool()
async def fetch_website(
    url: str,
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Fetch a website."""
    headers = {
        "User-Agent": "MCP Test Server (github.com/modelcontextprotocol/python-sdk)"
    }
    async with httpx.AsyncClient(follow_redirects=True, headers=headers) as client:
        response = await client.get(url)
        response.raise_for_status()
        return [types.TextContent(type="text", text=response.text)]


@mcp.tool()
async def get_weather(city: str) -> str:
    """
    Get the weather for a city.

    :param city: The city to get the weather for
    """
    import random

    return f"The temperature in {city} is {random.randint(10, 30)}°C."


@mcp.resource("echo://{message}")
def echo_resource(message: str) -> str:
    """Echo a message as a resource"""
    return f"Resource echo: {message}"


@mcp.resource("ask://help")
def help_resource() -> str:
    """Get help"""
    return "This is a simple server with a few resources."


@mcp.resource("ask://about")
def about_resource() -> str:
    """Get information about the server"""
    return "This is a simple server implementation."


@mcp.prompt()
def create_messages(
    context: str | None = None, topic: str | None = None
) -> list[types.PromptMessage]:
    """
    Create the messages for the prompt.

    :param context: Additional context to consider for the prompt
    :param topic: Specific topic to focus on
    """
    messages = []

    # Add context if provided
    if context:
        messages.append(
            types.PromptMessage(
                role="user",
                content=types.TextContent(
                    type="text", text=f"Here is some relevant context: {context}"
                ),
            )
        )

    # Add the main prompt
    prompt = "Please help me with "
    if topic:
        prompt += f"the following topic: {topic}"
    else:
        prompt += "whatever questions I may have."

    messages.append(
        types.PromptMessage(
            role="user", content=types.TextContent(type="text", text=prompt)
        )
    )

    return messages


if __name__ == "__main__":
    mcp.run()
