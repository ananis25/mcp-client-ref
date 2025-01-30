# mcp-client-ref

A [model context protocol](https://modelcontextprotocol.io/) client that runs in the terminal. Thanks to the [Textual](https://textual.textualize.io/) TUI library. 

## Installation

To run the client, install the dependencies and run `scripts/client.py`. Pass the path to the MCP servers config file as an argument.

```bash
uv sync
uv run scripts/client.py <mcp_servers_config_path>
```

- The MCP servers config file is a JSON file that lists the MCP servers you want the client to connect to. It follows the [format used](https://modelcontextprotocol.io/quickstart/user) by the `Claude Desktop` app.
- The client uses the Claude Haiku model by default, so you'll need to keep the `ANTHROPIC_API_KEY` in the `.env` file. It uses the magentic library to connect to the LLM APIs; you can configure a different model to use in the client code. 


### MCP server inspector 

You can debug/test the servers by opening up the inspector screen using the command palette in the app. 