import asyncio
import enum
import inspect
import json
import logging
import sys
import traceback
import typing as t
import uuid

import dotenv
import magentic as mg
import mcp
import textual as tx
from magentic.chat_model.anthropic_chat_model import AnthropicChatModel
from mcp.client.stdio import stdio_client
from rich.pretty import Pretty
from textual.app import App, ComposeResult, SystemCommand
from textual.containers import Grid, VerticalScroll
from textual.logging import TextualHandler
from textual.screen import ModalScreen, Screen
from textual.widget import Widget
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Markdown,
    Select,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
    Tree,
)

dotenv.load_dotenv()

logging.basicConfig(
    level="DEBUG",
    handlers=[TextualHandler()],
)


##########################
# MCP Client
##########################


def mcp_tool_to_function(tool_def: mcp.types.Tool) -> t.Callable[[t.Any], t.Any]:
    """
    Create a dummy function from a tool definition schema, thats what Magentic wants.

    :param tool_definition: Dictionary containing the tool definition with name,
        description, and input_schema
    """
    name = tool_def.name
    docstring = tool_def.description

    # Parse input schema
    properties = tool_def.inputSchema["properties"]
    required = tool_def.inputSchema.get("required", [])

    type_mapping = {
        "string": str,
        "number": float,
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    # Create parameters dictionary
    parameters = {}
    for param_name, param_info in properties.items():
        param_type = type_mapping.get(param_info["type"], t.Any)
        parameters[param_name] = param_type

    # Create signature parameters
    sig_parameters = []
    for param_name, param_type in parameters.items():
        # Make parameter required or optional based on the required list
        default = inspect.Parameter.empty if param_name in required else None
        param = inspect.Parameter(
            name=param_name,
            kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=param_type,
            default=default,
        )
        sig_parameters.append(param)

    # Create the signature
    sig = inspect.Signature(sig_parameters, return_annotation=t.Any)

    # Create the dummy function
    def dummy_func(*args, **kwargs): ...

    # Set the function's metadata
    dummy_func.__name__ = name
    dummy_func.__doc__ = docstring
    dummy_func.__signature__ = sig

    return dummy_func


class StateMCPClient(enum.Enum):
    INIT = enum.auto()
    READY = enum.auto()
    ERROR = enum.auto()
    EXIT = enum.auto()


class MCPClient:
    name: str
    server_params: mcp.StdioServerParameters
    session: mcp.ClientSession
    history: list
    capabilities: dict[str, t.Any] | None = None

    def __init__(self, name: str, server_params: dict):
        """Maintain a connection to the MCP server and tools

        Args:
            server_script_path: Path to the server script (.py)
        """
        self.name = name
        self.server_params = mcp.StdioServerParameters(**server_params)
        self.state = StateMCPClient.INIT
        self.mailbox = asyncio.Queue()

    async def connect(self):
        try:
            async with stdio_client(self.server_params) as (read, write):
                async with mcp.ClientSession(read, write) as session:
                    self.session = session
                    await self.session.initialize()
                    self.state = StateMCPClient.READY

                    # wait for an exit signal or to get cancelled
                    await self.mailbox.get()
                    self.state = StateMCPClient.EXIT

        except Exception as e:
            self.state = StateMCPClient.ERROR
            logging.error(
                f"Error connecting to MCP server {self.name}: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            )

    async def fetch_capabilities(self):
        """Fetch the capabilities of the MCP server"""

        if self.capabilities is None:
            tools = (await self.session.list_tools()).tools
            prompts = (await self.session.list_prompts()).prompts
            resources = (await self.session.list_resources()).resources
            self.capabilities = dict(tools=tools, prompts=prompts, resources=resources)

        return self.capabilities


##########################
# Textual UI
##########################


class UserMsg(Markdown):
    """User's prompt display widget"""

    BORDER_TITLE = "You"


class AssistantMsg(Markdown):
    """Assistant's response display widget"""

    BORDER_TITLE = "Assistant"


class ChatScreen(Screen):
    """Individual chat screen"""

    BINDINGS = [
        ("ctrl+r", "remove_chat", "Remove Chat"),
    ]

    CSS = """
    UserMsg {
        border: solid $accent;
        background: $primary 10%;
        color: $text;
        margin-right: 8;
        padding: 1 1 0 1;
    }
    AssistantMsg {
        border: solid $success;
        background: $success 10%;   
        color: $text;
        margin-left: 8;
        padding: 1 1 0 1;
    }
    LoadingIndicator {
        color: $text;
    }
    """

    app: "MCPTextualApp"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.name is not None
        self.title = self.name[:12]

    def compose(self) -> ComposeResult:
        """Compose the chat screen layout"""
        yield Header()
        with VerticalScroll(id="chat-view"):
            yield AssistantMsg("Loading tools...", id="tools-list")
        yield Input(placeholder="Enter your query...")
        yield Footer()

    async def initialize(self):
        """Fetch the list of available capabilities for this chat window"""

        tools_list = self.query_one("#tools-list")
        assert isinstance(tools_list, AssistantMsg)

        available_tools: list[mcp.types.Tool] = []
        for client in self.app.mcp_clients:
            if client.state == StateMCPClient.READY:
                capabilities = await client.fetch_capabilities()
                available_tools.extend(capabilities["tools"])  # type: ignore
                tools_list.update(
                    "Available tools\n\n"
                    + "\n".join([f"- {tool.name}" for tool in available_tools])
                )

        self.chat = mg.Chat(
            model=AnthropicChatModel(model="claude-3-5-haiku-latest"),
            output_types=[mg.AsyncStreamedResponse],
            functions=[mcp_tool_to_function(tool) for tool in available_tools],
        )
        self.chat = self.chat.add_system_message(
            "You are a helpful assistant. You can use the attached tools if needed to answer the user query."
        )

        chat_view = self.query_one("#chat-view")
        chat_view.mount(AssistantMsg("All set. How can I help you today?"))

    def on_mount(self) -> None:
        self.run_worker(self.initialize, group="mcp-server")

    async def action_remove_chat(self):
        """Remove the chat window"""
        assert self.name is not None
        await self.app.remove_chat_screen(self.name)

    @tx.on(Input.Submitted)
    async def on_input(self, event: Input.Submitted) -> None:
        """Handle user input submission"""
        query = event.value
        event.input.clear()

        chat_view = self.query_one("#chat-view")
        await chat_view.mount(UserMsg(query))

        response_widget = AssistantMsg()
        await chat_view.mount(response_widget)
        response_widget.anchor()

        self.process_chat(query)

    @tx.work(group="llm")
    async def process_chat(self, query: str) -> None:
        """Process a query using the LLM and available tools, maintaining chat history"""
        response_widget = self.query(AssistantMsg).last()
        try:
            # Add the user's message to the chat history
            self.chat = self.chat.add_user_message(query)

            response_content = ""
            while True:
                self.chat = await self.chat.asubmit()
                response_widget.loading = True

                tool_calls = []
                async for chunk in self.chat.last_message.content:
                    response_widget.loading = False
                    if isinstance(chunk, mg.AsyncStreamedStr):
                        async for part in chunk:
                            response_content += part
                            await response_widget.update(response_content)
                    elif isinstance(chunk, mg.FunctionCall):
                        tool_name = chunk.function.__name__
                        tool_args = chunk.arguments
                        tool_calls.append((tool_name, tool_args, chunk))

                        response_content += f"\n\n### Tool call {len(tool_calls)}\n\nCalling {tool_name} - {tool_args}\n\n"
                        await response_widget.update(response_content)

                if not tool_calls:
                    break
                else:
                    for idx, (tool_name, tool_args, fn_call_chunk) in enumerate(
                        tool_calls
                    ):
                        # Figure out which MCP client to use and call the tool
                        mcp_client = None
                        for client in self.app.mcp_clients:
                            client_tools = (await client.fetch_capabilities())["tools"]
                            for tool in client_tools:
                                if tool.name == tool_name:
                                    mcp_client = client
                                    break

                        assert mcp_client is not None
                        tool_result = await mcp_client.session.call_tool(
                            tool_name, tool_args
                        )
                        result = ",".join(
                            r.text
                            for r in tool_result.content
                            if isinstance(r, mcp.types.TextContent)
                        )
                        response_content += (
                            f"\n\n### Tool result {idx + 1}\n\n{result}\n\n"
                        )
                        await response_widget.update(response_content)

                        self.chat = self.chat.add_message(
                            mg.FunctionResultMessage(
                                content=str(result), function_call=fn_call_chunk
                            )
                        )

            if self.title is not None and self.title.startswith("chat."):
                fork_chat = self.chat.add_user_message(
                    "Create a quick slug-like ASCII title for this conversation. Do not use tools. Output only the title."
                )
                fork_chat = await fork_chat.asubmit()

                _new_title = ""
                async for chunk in fork_chat.last_message.content:
                    if isinstance(chunk, mg.AsyncStreamedStr):
                        async for part in chunk:
                            _new_title += part
                self.title = _new_title

        except Exception as e:
            self.log(f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}")
            await response_widget.update(
                "Error: An error occurred while processing your request."
            )


class ListChatWindows(ModalScreen):
    """Modal screen to list and select chat windows"""

    AUTO_FOCUS = "#chat-window-select"

    BINDINGS = [
        ("escape", "app.pop_screen", "Close"),
    ]

    app: "MCPTextualApp"

    def compose(self) -> ComposeResult:
        yield Header()
        yield Select(
            options=[
                (screen.title, screen.name)
                for screen in self.app.chat_screens
                if screen.title is not None
            ],
            prompt="Chat windows",
            id="chat-window-select",
        )
        yield Footer()

    @tx.on(Select.Changed)
    def select_changed(self, event: Select.Changed) -> None:
        self.app.switch_screen(screen=event.value)  # type: ignore


def _format_description(description: str) -> str:
    """Format a function description to be displayed in a table"""

    lines = [line for line in description.split("\n") if line.strip() != ""]
    common_whitespace = min(
        (len(line) - len(line.lstrip()) for line in lines if line.strip()), default=0
    )
    return "\n".join([line[common_whitespace:] for line in lines])


class EditParameter(ModalScreen):
    """Edit a parameter value"""

    BINDINGS = [
        ("escape", "app.pop_screen", "return to Inspector"),
        ("ctrl+enter", "submit_value", "submit value"),
    ]

    def __init__(self, *args, **kwargs):
        self.row_data = kwargs.pop("row_data")
        super().__init__(*args, **kwargs)

    def compose(self) -> ComposeResult:
        dialog = Grid(
            Markdown(id="description"),
            TextArea(id="input"),
            id="dialog",
        )
        dialog.border_title = "Edit parameter"
        yield dialog
        yield Footer()

    def on_mount(self) -> None:
        description = self.query_one("#description", Markdown)
        description.update(
            f"""
{self.row_data[0]} {"(required)" if self.row_data[2] else ""}

{self.row_data[3]}
"""
        )

        text_input = self.query_one("#input", TextArea)
        text_input.text = self.row_data[1]
        text_input.focus()

    async def action_submit_value(self) -> None:
        self.dismiss(self.query_one("#input", TextArea).text)


class ServerInspector(Screen):
    """List of MCP servers"""

    BINDINGS = [
        ("escape", "app.pop_screen", "Return to chat"),
    ]

    CSS = """
    .inspector-grid {
        grid-size: 2 2;
        grid-columns: 1fr 2fr;
        grid-rows: 1fr 1fr;
    }
    .inspector-box {
        padding: 0 1;
        margin: 0 1;
        layout: vertical;
        overflow: auto auto;
        border: solid $primary;
        border-title-align: right;
        border-title-background: $background;
        border-title-color: $text;
    }
    .capabilities {
        row-span: 2;
    }
    .capability-call {
        height: 1;
        min-width: 10;
        border: none;
    }
    .capability-description {
        margin: 1 0;
    }
    .error-response {
        border: solid $error;
    }
    """

    app: "MCPTextualApp"

    class RequestEditor(Widget):
        BORDER_TITLE = "Request"

        def compose(self):
            yield Static("")

    class ResponseViewer(Static):
        BORDER_TITLE = "Response"

        def compose(self):
            yield Label(classes="response-viewer-content")

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(classes="mcp-servers-list"):
            for mcp_client in self.app.mcp_clients:
                with TabPane(mcp_client.name, id=mcp_client.name):
                    with Grid(
                        classes="inspector-grid",
                        id=f"inspector-grid-{mcp_client.name}",
                    ):
                        yield Tree(
                            label="Capabilities",
                            classes="inspector-box capabilities",
                            id=f"capabilities-{mcp_client.name}",
                        )
                        yield ServerInspector.RequestEditor(
                            classes="inspector-box request-editor",
                            id=f"request-editor-{mcp_client.name}",
                        )
                        yield ServerInspector.ResponseViewer(
                            classes="inspector-box response-viewer",
                            id=f"response-viewer-{mcp_client.name}",
                        )
        yield Footer()

    async def load_capabilities(self):
        for mcp_client in self.app.mcp_clients:
            if mcp_client.state == StateMCPClient.READY:
                capabilities = await mcp_client.fetch_capabilities()

                tree = self.query_one(f"#capabilities-{mcp_client.name}")
                assert isinstance(tree, Tree)
                tree.root.expand()

                tools_node = tree.root.add("Tools", expand=True)
                for tool in capabilities["tools"]:
                    tools_node.add_leaf(
                        tool.name,
                        data={
                            "type": "tool",
                            "client": mcp_client.name,
                            "name": tool.name,
                        },
                    )

                prompts_node = tree.root.add("Prompts", expand=True)
                for prompt in capabilities["prompts"]:
                    prompts_node.add_leaf(
                        prompt.name,
                        data={
                            "type": "prompt",
                            "client": mcp_client.name,
                            "name": prompt.name,
                        },
                    )

                resources_node = tree.root.add("Resources", expand=True)
                for resource in capabilities["resources"]:
                    resources_node.add_leaf(
                        resource.name,
                        data={
                            "type": "resource",
                            "client": mcp_client.name,
                            "name": resource.name,
                        },
                    )

    def on_mount(self) -> None:
        self.run_worker(self.load_capabilities, group="mcp-server")

    @property
    def active_client(self) -> str:
        tabbed_content = self.query_one(".mcp-servers-list", TabbedContent)
        assert (
            tabbed_content.active_pane is not None
            and tabbed_content.active_pane.id is not None
        )
        return tabbed_content.active_pane.id

    async def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        if event.node.data is None:
            return

        client = [
            cli for cli in self.app.mcp_clients if cli.name == self.active_client
        ][0]
        assert client.capabilities is not None

        request_editor = self.query_one(
            f"#request-editor-{client.name}", ServerInspector.RequestEditor
        )
        await request_editor.remove_children()
        self.update_response("", is_error=False)

        # update the request editor
        submit = Button("Submit", classes="capability-call")
        description = Static(classes="capability-description")
        table = DataTable(classes="capability-parameters")

        EDITOR_WIDTH = request_editor.size.width - 6  # take out padding values
        COLUMNS = ("parameter", "value", "required", "description")
        SUM_LEN_COLUMNS = sum(len(col) for col in COLUMNS)
        for col in COLUMNS:
            table.add_column(
                col, key=col, width=(EDITOR_WIDTH * len(col)) // SUM_LEN_COLUMNS
            )
        table.cursor_type = "row"

        if event.node.data["type"] == "tool":
            tool: mcp.types.Tool = [
                x
                for x in client.capabilities["tools"]
                if x.name == event.node.data["name"]
            ][0]
            description.update(
                _format_description(tool.description or "No description available")
            )

            properties = tool.inputSchema.get("properties", {})
            required = tool.inputSchema.get("required", [])

            for param_name, param_info in properties.items():
                table.add_row(
                    param_name,
                    "",
                    "Yes" if param_name in required else "",
                    param_info.get("description", ""),
                    key=param_name,
                )

        elif event.node.data["type"] == "prompt":
            prompt: mcp.types.Prompt = [
                x
                for x in client.capabilities["prompts"]
                if x.name == event.node.data["name"]
            ][0]
            description.update(
                _format_description(prompt.description or "No description available")
            )

            if prompt.arguments:
                for arg in prompt.arguments:
                    table.add_row(
                        arg.name,
                        "",
                        "Yes" if arg.required else "",
                        arg.description,
                        key=arg.name,
                    )

        elif event.node.data["type"] == "resource":
            resource: mcp.types.Resource = [
                x
                for x in client.capabilities["resources"]
                if x.name == event.node.data["name"]
            ][0]
            description.update(
                _format_description(resource.description or "No description available")
            )

            # TODO: resource templates don't work with the MCP client SDK right now

        request_editor.mount(submit)
        request_editor.mount(description)
        request_editor.mount(table)

    def update_response(
        self,
        text: t.Any,
        is_error: bool = False,
        is_loading: bool = False,
    ) -> None:
        response_viewer_content = self.query_one(
            f"#response-viewer-{self.active_client}", ServerInspector.ResponseViewer
        ).query_one(".response-viewer-content", Label)

        if is_loading:
            response_viewer_content.loading = True
        else:
            response_viewer_content.loading = False
            response_viewer_content.update(text)
            if is_error:
                response_viewer_content.add_class("error-response")
            else:
                response_viewer_content.remove_class("error-response")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.row_key is None:
            return

        table = self.query_one(
            f"#request-editor-{self.active_client}", ServerInspector.RequestEditor
        ).query_one(".capability-parameters", DataTable)
        row_data = list(table.get_row(event.row_key))
        self.app.push_screen(
            EditParameter(row_data=row_data),
            callback=lambda val: table.update_cell(event.row_key, "value", val),
        )

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if not event.button.has_class("capability-call"):
            return

        tree = self.query_one(f"#capabilities-{self.active_client}", Tree)
        active_node = tree.cursor_node
        if not active_node or active_node.data is None:
            return

        table = self.query_one(
            f"#request-editor-{self.active_client}", ServerInspector.RequestEditor
        ).query_one(".capability-parameters", DataTable)
        parameters = {
            param.value: table.get_cell(param, "value") for param in table.rows
        }

        client = [
            cli for cli in self.app.mcp_clients if cli.name == self.active_client
        ][0]

        self.update_response("", is_loading=True)
        try:
            if active_node.data["type"] == "tool":
                result = await client.session.call_tool(
                    active_node.data["name"], parameters
                )

            elif active_node.data["type"] == "prompt":
                result = await client.session.get_prompt(
                    active_node.data["name"], parameters
                )

            elif active_node.data["type"] == "resource":
                result = await client.session.read_resource(active_node.data["name"])

            self.update_response(Pretty(result))
        except mcp.McpError as err:
            self.update_response(err.error.__rich_repr__(), is_error=True)


class MCPTextualApp(App):
    """Textual-based UI for the MCP Client"""

    AUTO_FOCUS = "Input"
    BINDINGS = [
        ("ctrl+n", "new_chat", "New Chat"),
        ("ctrl+l", "list_chats", "List Chats"),
    ]
    CSS = """
    EditParameter {
        align: center middle;
        
        #dialog {
            border: solid $primary;
            border-title-align: center;
            grid-size: 1 2;
            padding: 1 1;
            width: 40;
            height: 20
        }

        #description {
            border: solid $accent;
        }
    }
    """

    mcp_clients: list[MCPClient]
    chat_screens: list[ChatScreen]

    def __init__(
        self,
        path_config: str = "/Users/ishananand/repos/pypypy/mcp-client-ref/notebooks/mcp_config.json",
    ):
        super().__init__()
        with open(path_config, "r") as f:
            self.mcp_config = json.load(f)
        self.mcp_clients = []
        for server_name, server_params in self.mcp_config["mcpServers"].items():
            self.mcp_clients.append(MCPClient(server_name, server_params))
        self.chat_screens: list[ChatScreen] = []

    def get_system_commands(self, screen: Screen) -> t.Iterable[SystemCommand]:
        yield from super().get_system_commands(screen=screen)
        yield SystemCommand(
            title="inspect-mcp-servers",
            help="Inspect MCP servers",
            callback=lambda: self.push_screen(ServerInspector(name="server-inspector")),
        )

    def compose(self) -> ComposeResult:
        """Compose the UI layout"""
        yield Header()
        yield Footer()

    async def on_mount(self) -> None:
        """Initialize the MCP client when the app starts"""
        for mcp_client in self.mcp_clients:
            self.run_worker(mcp_client.connect, group="mcp-server")

        async def wait_for_client(cli: MCPClient):
            while cli.state not in (StateMCPClient.READY, StateMCPClient.ERROR):
                await asyncio.sleep(0.1)

        await asyncio.gather(
            *[wait_for_client(mcp_client) for mcp_client in self.mcp_clients]
        )
        await self.action_new_chat()

    async def on_unmount(self) -> None:
        """Clean up resources when the app closes"""
        # The MCP clients run as asyncio tasks, the context manager should unwind itself automatically
        pass

    async def action_new_chat(self) -> None:
        """Create a new chat window"""
        new_screen = ChatScreen(name=f"chat.{uuid.uuid4()}")
        self.chat_screens.append(new_screen)

        self.install_screen(new_screen, name=new_screen.name)  # type: ignore
        await self.push_screen(new_screen)

    async def action_list_chats(self) -> None:
        """Show the list of all chat windows"""
        await self.push_screen(ListChatWindows())

    async def remove_chat_screen(self, name: str) -> None:
        """Remove the active chat window."""

        await self.pop_screen()
        for screen in self.chat_screens:
            if screen.name == name:
                self.chat_screens.remove(screen)
                self.uninstall_screen(screen)
                break


def main():
    if len(sys.argv) != 2:
        print("Usage: uv run client.py <mcp_servers_config_path>")
        sys.exit(1)

    app = MCPTextualApp(sys.argv[1])
    app.run()


if __name__ == "__main__":
    main()
