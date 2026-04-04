from mcp_stl.adapters.mcp_server import create_mcp_server


def main() -> int:
    mcp = create_mcp_server()
    mcp.run()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
