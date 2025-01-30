# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mcp[cli] @ git+https://github.com/modelcontextprotocol/python-sdk@c14ec2e",
#     "uvicorn==0.34.0",
# ]
# ///

import random

import mcp.types as types
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ServerFortune")


@mcp.tool()
async def roll_dice_and_sum(sides: int = 6, num_dice: int = 1) -> str:
    """
    Roll some dice and sum the results.

    Args:
        sides: Number of sides on each die (default: 6)
        num_dice: Number of dice to roll (default: 1)
    """
    results = [random.randint(1, sides) for _ in range(num_dice)]
    total = sum(results)
    return f"Rolled {num_dice} {sides}-sided dice: {results}. Total: {total}"


@mcp.tool()
async def fortune_cookie() -> str:
    """Get a random fortune cookie message"""
    fortunes = [
        "A beautiful, smart, and loving person will be coming into your life.",
        "A dubious friend may be an enemy in camouflage.",
        "A golden egg of opportunity falls into your lap this month.",
        "A good time to finish up old tasks.",
    ]
    return random.choice(fortunes)


@mcp.resource("quote://random")
def random_quote() -> str:
    """Get a random inspirational quote"""
    quotes = [
        "Be the change you wish to see in the world.",
        "Life is what happens while you're busy making other plans.",
        "The only way to do great work is to love what you do.",
        "The best way to predict the future is to invent it.",
        "The only limit to our realization of tomorrow will be our doubts of today.",
        "Success is not the key to happiness. Happiness is the key to success. If you love what you are doing, you will be successful.",
        "Believe you can and you're halfway there.",
        "The only way to do great work is to love what you do.",
        "Your time is limited, don't waste it living someone else's life.",
        "The best way to predict the future is to invent it.",
    ]
    return random.choice(quotes)


@mcp.prompt()
def brainstorm(topic: str, num_ideas: int = 3) -> list[types.PromptMessage]:
    """
    Create a brainstorming prompt.

    Args:
        topic: Topic to brainstorm about
        num_ideas: Number of ideas to request (default: 3)
    """
    return [
        types.PromptMessage(
            role="user",
            content=types.TextContent(
                type="text",
                text=f"Let's brainstorm {num_ideas} creative ideas about: {topic}\n\nPlease provide unique and innovative suggestions.",
            ),
        )
    ]


if __name__ == "__main__":
    mcp.run()
