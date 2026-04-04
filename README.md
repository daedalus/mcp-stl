# mcp-stl

[![PyPI](https://img.shields.io/pypi/v/mcp-stl.svg)](https://pypi.org/project/mcp-stl/)
[![Python](https://img.shields.io/pypi/pyversions/mcp-stl.svg)](https://pypi.org/project/mcp-stl/)
[![Coverage](https://codecov.io/gh/daedalus/mcp-stl/branch/main/graph/badge.svg)](https://codecov.io/gh/daedalus/mcp-stl)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

MCP server for editing STL 3D model files. Provides tools for parsing, viewing, and manipulating STL files through the Model Context Protocol.

## Install

```bash
pip install mcp-stl
```

## MCP Server

mcp-name: io.github.daedalus/mcp-stl

## Tools

### Reading & Writing

- **read_stl_file**: Read an STL file and return mesh data (vertices, normals, bounding box)
- **get_mesh_info**: Get summary information about an STL file without loading full geometry
- **write_stl**: Write mesh data to an STL file (ASCII or binary format)

### Transformations

- **translate_stl**: Translate (move) the mesh by specified X, Y, Z offsets
- **rotate_stl**: Rotate the mesh around X, Y, or Z axis by a given angle in degrees
- **scale_stl**: Scale the mesh by specified X, Y, Z factors

### Primitives

- **create_cube**: Create a cube mesh
- **create_sphere**: Create a sphere mesh
- **create_cylinder**: Create a cylinder mesh
- **create_cone**: Create a cone mesh
- **create_torus**: Create a torus (donut) mesh
- **create_plane**: Create a plane mesh

## Usage

### Python API

```python
from mcp_stl import read_stl_file, create_cube, translate_stl

# Read an existing STL file
mesh = read_stl_file("model.stl")
print(f"Faces: {mesh.face_count}")

# Create a primitive
create_cube("cube.stl", size=2.0)

# Transform a mesh
translate_stl("input.stl", "output.stl", x=10.0, y=0.0, z=5.0)
```

### MCP Server

Configure in your MCP client:

```json
{
  "mcpServers": {
    "mcp-stl": {
      "command": "mcp-stl"
    }
  }
}
```

## Development

```bash
git clone https://github.com/daedalus/mcp-stl.git
cd mcp-stl
pip install -e ".[test]"

# run tests
pytest

# format
ruff format src/ tests/

# lint
ruff check src/ tests/

# type check
mypy src/
```
