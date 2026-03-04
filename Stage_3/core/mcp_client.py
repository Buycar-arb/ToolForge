from mcp import ClientSession
from mcp.client.sse import sse_client
class MCPCaller:
    def __init__(self):
        self.session = None

    async def connect_to_sse_server(self, server_url: str):
        """Connect to an SSE MCP server."""
        print(f"Connecting to SSE MCP server at {server_url}")

        self._streams_context = sse_client(url=server_url)
        streams = await self._streams_context.__aenter__()

        self._session_context = ClientSession(*streams)
        self.session = await self._session_context.__aenter__()

        # Initialize
        await self.session.initialize()
        
        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print(f"Connected to SSE MCP Server at {server_url}. Available tools: {[tool.name for tool in tools]}")

   
    async def connect_to_server(self, server_path_or_url: str):
        """Connect to an MCP server (either stdio or SSE)."""
        await self.connect_to_sse_server(server_path_or_url)

    async def cleanup(self):
        """Clean up resources."""
        if hasattr(self, '_session_context') and self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if hasattr(self, '_streams_context') and self._streams_context:
            await self._streams_context.__aexit__(None, None, None)

