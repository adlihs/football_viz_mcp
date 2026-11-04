import os
import sys

if "--stdio" in sys.argv or os.getenv("MCP_TRANSPORT") == "stdio":
    from .server import mcp

    mcp.run("stdio")
else:
    from .app import main

    main()
