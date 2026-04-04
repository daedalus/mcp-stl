# SPEC.md — mcp-stl

## Purpose

An MCP server that provides tools for parsing, viewing, and editing STL (Stereolithography) 3D model files. It enables AI assistants to manipulate 3D meshes by reading STL geometry, modifying vertices and normals, and writing the results back to STL format.

## Scope

- **In scope:**
  - Parse ASCII STL files
  - Parse binary STL files
  - Extract mesh information (vertices, normals, face count, bounding box)
  - Modify mesh by translating, rotating, and scaling
  - Generate 3D modeling primitives (cube, sphere, cylinder, cone, torus, plane)
  - Write STL files (both ASCII and binary formats)
  - Provide tools as MCP server for AI integration

- **Not out of scope:**
  - GUI visualization
  - File format conversion to non-STL formats
  - Advanced mesh editing (vertex-level editing, edge operations)
  - Mesh validation or repair

## Public API / Interface

### MCP Tools

1. **`read_stl_file(path: str) -> dict`**
   - Reads an STL file and returns mesh data.
   - Args:
     - `path`: Path to the STL file
   - Returns:
     - Dictionary with: `vertex_count`, `face_count`, `normals`, `vertices`, `bounding_box`, `format` (ascii|binary)
   - Raises: `FileNotFoundError`, `ValueError` (invalid STL)

2. **`get_mesh_info(path: str) -> dict`**
   - Returns summary information about an STL file without loading full geometry.
   - Args:
     - `path`: Path to the STL file
   - Returns:
     - Dictionary with: `face_count`, `bounding_box`, `center`, `volume` (if binary), `format`

3. **`translate_stl(path: str, output_path: str, x: float, y: float, z: float) -> str`**
   - Translates (moves) the mesh by the specified offset.
   - Args:
     - `path`: Input STL file path
     - `output_path`: Output STL file path
     - `x`: Translation along X axis
     - `y`: Translation along Y axis
     - `z`: Translation along Z axis
   - Returns: Path to the output file

4. **`rotate_stl(path: str, output_path: str, axis: str, angle: float) -> str`**
   - Rotates the mesh around a specified axis.
   - Args:
     - `path`: Input STL file path
     - `output_path`: Output STL file path
     - `axis`: Rotation axis ('x', 'y', or 'z')
     - `angle`: Rotation angle in degrees
   - Returns: Path to the output file

5. **`scale_stl(path: str, output_path: str, x: float, y: float, z: float) -> str`**
   - Scales the mesh by the specified factors.
   - Args:
     - `path`: Input STL file path
     - `output_path`: Output STL file path
     - `x`: Scale factor for X axis
     - `y`: Scale factor for Y axis
     - `z`: Scale factor for Z axis
   - Returns: Path to the output file

6. **`write_stl(vertices: list, normals: list, output_path: str, format: str = "binary") -> str`**
   - Writes mesh data to an STL file.
   - Args:
     - `vertices`: List of vertices as [x, y, z] triplets
     - `normals`: List of face normals as [x, y, z] triplets
     - `output_path`: Output STL file path
     - `format`: Output format ('ascii' or 'binary')
   - Returns: Path to the output file

7. **`create_cube(output_path: str, size: float = 1.0, center: list = [0, 0, 0]) -> str`**
   - Creates a cube mesh.
   - Args:
     - `output_path`: Output STL file path
     - `size`: Edge length (default 1.0)
     - `center`: Center position as [x, y, z] (default [0, 0, 0])
   - Returns: Path to the output file

8. **`create_sphere(output_path: str, radius: float = 1.0, segments: int = 32) -> str`**
   - Creates a sphere mesh.
   - Args:
     - `output_path`: Output STL file path
     - `radius`: Sphere radius (default 1.0)
     - `segments`: Number of horizontal segments (default 32)
   - Returns: Path to the output file

9. **`create_cylinder(output_path: str, radius: float = 1.0, height: float = 2.0, segments: int = 32) -> str`**
   - Creates a cylinder mesh.
   - Args:
     - `output_path`: Output STL file path
     - `radius`: Cylinder radius (default 1.0)
     - `height`: Cylinder height (default 2.0)
     - `segments`: Number of radial segments (default 32)
   - Returns: Path to the output file

10. **`create_cone(output_path: str, radius: float = 1.0, height: float = 2.0, segments: int = 32) -> str`**
    - Creates a cone mesh.
    - Args:
      - `output_path`: Output STL file path
      - `radius`: Base radius (default 1.0)
      - `height`: Cone height (default 2.0)
      - `segments`: Number of radial segments (default 32)
    - Returns: Path to the output file

11. **`create_torus(output_path: str, major_radius: float = 1.0, minor_radius: float = 0.3, major_segments: int = 32, minor_segments: int = 16) -> str`**
    - Creates a torus (donut) mesh.
    - Args:
      - `output_path`: Output STL file path
      - `major_radius`: Distance from center to tube center (default 1.0)
      - `minor_radius`: Tube radius (default 0.3)
      - `major_segments`: Number of segments around the ring (default 32)
      - `minor_segments`: Number of segments around the tube (default 16)
    - Returns: Path to the output file

12. **`create_plane(output_path: str, width: float = 1.0, height: float = 1.0) -> str`**
    - Creates a plane mesh (2D flat surface).
    - Args:
      - `output_path`: Output STL file path
      - `width`: Plane width (default 1.0)
      - `height`: Plane height (default 1.0)
    - Returns: Path to the output file

### MCP Resources

1. **`stl://{filepath}`**
   - Resource URI to get mesh information for a file.
   - Returns mesh metadata as JSON.

## Data Formats

### STL ASCII Format
```
solid name
facet normal nx ny nz
  outer loop
    vertex x1 y1 z1
    vertex x2 y2 z2
    vertex x3 y3 z3
  endloop
endfacet
endsolid name
```

### STL Binary Format
- 80-byte header
- 4-byte face count (uint32)
- 50-byte per face: normal(3×float) + vertices(3×3×float) + 2-byte attribute byte count

## Edge Cases

1. Empty STL file (0 faces) - should handle gracefully
2. Malformed ASCII STL (missing vertex/endfacet) - raise ValueError
3. Binary STL with wrong face count - raise ValueError
4. Non-existent file path - raise FileNotFoundError
5. Zero scale factor - allowed (collapses mesh to plane/line/point)
6. Rotation by 0 or 360 degrees - no-op, should still work
7. Very large files (>100MB) - may require chunked processing
8. File with duplicate vertices - handle but don't deduplicate
9. Negative coordinates - valid and supported
10. Invalid axis for rotation - raise ValueError

## Performance & Constraints

- Must handle files up to 100MB efficiently
- Face count extraction should be O(1) for binary STL (seek to byte 80)
- Memory usage should be reasonable (<2x file size)
- Use numpy for vectorized operations where possible

