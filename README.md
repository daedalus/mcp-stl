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
- **shear_stl**: Apply a shear transformation (shift one axis per unit along another)
- **mirror_stl**: Mirror (reflect) the mesh across the plane perpendicular to a given axis
- **rotate_stl_axis**: Rotate the mesh around an arbitrary axis vector (Rodrigues' formula)
- **combine_stl**: Merge multiple STL files into one
- **array_linear**: Create N copies of the mesh spaced by a fixed X/Y/Z offset
- **array_circular**: Create N copies of the mesh at equal angular intervals around an axis

### Primitives

- **create_cube**: Create a cube mesh
- **create_box**: Create a rectangular box (cuboid) with independent width/height/depth
- **create_sphere**: Create a sphere mesh
- **create_cylinder**: Create a cylinder mesh
- **create_cone**: Create a cone mesh
- **create_torus**: Create a torus (donut) mesh
- **create_plane**: Create a flat plane mesh
- **create_capsule**: Create a capsule (cylinder with hemispherical end caps)
- **create_ellipsoid**: Create an ellipsoid with independent radii on each axis
- **create_frustum**: Create a frustum (truncated cone)
- **create_tube**: Create a hollow cylinder (tube/pipe)
- **create_pyramid**: Create a regular n-sided pyramid
- **create_prism**: Create a regular n-sided prism (e.g. hexagonal prism)
- **create_hemisphere**: Create a hemisphere (dome with flat circular base)
- **create_wedge**: Create a right-triangular wedge (triangular prism)

### Engine-Design Shapes

- **create_gear**: Create a spur gear mesh (configurable module, teeth, pressure angle)
- **create_spring**: Create a helical coil spring
- **create_connecting_rod**: Create a connecting rod with big/small-end bores
- **create_crankshaft**: Create an N-throw crankshaft
- **create_valve**: Create a poppet valve (stem + head disc)
- **create_camshaft_lobe**: Create an eccentric cam lobe with cosine nose profile

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
