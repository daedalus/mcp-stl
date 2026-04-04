from mcp_stl.adapters.mcp_server import create_mcp_server


def test_create_mcp_server() -> None:
    mcp = create_mcp_server()
    assert mcp is not None
    assert mcp.name == "mcp-stl"
