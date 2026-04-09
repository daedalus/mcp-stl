# SPEC.md — mcp-stl

## Purpose

An MCP server that provides tools for parsing, viewing, and editing STL (Stereolithography) 3D model files. It enables AI assistants to manipulate 3D meshes by reading STL geometry, modifying vertices and normals, and writing the results back to STL format.

## Scope

- **In scope:**
  - Parse ASCII STL files
  - Parse binary STL files
  - Extract mesh information (vertices, normals, face count, bounding box)
  - Modify mesh by translating, rotating, and scaling
  - Mirror mesh along a plane (X, Y, or Z)
  - Rotate mesh around an arbitrary axis vector (Rodrigues' rotation)
  - Merge multiple STL files into one
  - Generate 3D modeling primitives (cube, box, sphere, cylinder, cone, torus, plane, capsule, ellipsoid, frustum, tube, pyramid, prism, hemisphere, wedge)
  - Generate rocket engine components (bell nozzle, injector plate, pump housing)
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

13. **`create_box(output_path: str, width: float = 1.0, height: float = 1.0, depth: float = 1.0, center: list = [0, 0, 0]) -> str`**
    - Creates a rectangular box (cuboid) mesh with independent dimensions.
    - Args:
      - `output_path`: Output STL file path
      - `width`: Box width along X axis (default 1.0)
      - `height`: Box height along Y axis (default 1.0)
      - `depth`: Box depth along Z axis (default 1.0)
      - `center`: Center position as [x, y, z] (default [0, 0, 0])
    - Returns: Path to the output file

14. **`create_capsule(output_path: str, radius: float = 0.5, height: float = 2.0, segments: int = 32) -> str`**
    - Creates a capsule mesh (cylinder with hemispherical end caps).
    - Useful for robot arm link bodies with smooth rounded ends.
    - Args:
      - `output_path`: Output STL file path
      - `radius`: Radius of the cylinder and hemispheres (default 0.5)
      - `height`: Height of the cylindrical section only (default 2.0)
      - `segments`: Number of radial/latitude segments (default 32)
    - Returns: Path to the output file

15. **`create_ellipsoid(output_path: str, rx: float = 1.0, ry: float = 0.5, rz: float = 0.75, segments: int = 32) -> str`**
    - Creates an ellipsoid mesh with independent radii on each axis.
    - Useful for joint housings and actuator enclosures.
    - Args:
      - `output_path`: Output STL file path
      - `rx`: Radius along X axis (default 1.0)
      - `ry`: Radius along Y axis (default 0.5)
      - `rz`: Radius along Z axis (default 0.75)
      - `segments`: Number of latitude/longitude segments (default 32)
    - Returns: Path to the output file

16. **`create_frustum(output_path: str, bottom_radius: float = 1.0, top_radius: float = 0.5, height: float = 2.0, segments: int = 32) -> str`**
    - Creates a frustum (truncated cone) mesh.
    - Useful for tapered arm segments transitioning between joint diameters.
    - Args:
      - `output_path`: Output STL file path
      - `bottom_radius`: Radius of the bottom circle (default 1.0)
      - `top_radius`: Radius of the top circle (default 0.5)
      - `height`: Frustum height (default 2.0)
      - `segments`: Number of radial segments (default 32)
    - Returns: Path to the output file

17. **`create_tube(output_path: str, outer_radius: float = 1.0, inner_radius: float = 0.7, height: float = 2.0, segments: int = 32) -> str`**
    - Creates a hollow cylinder (tube) mesh.
    - Useful for hollow structural link bodies.
    - Args:
      - `output_path`: Output STL file path
      - `outer_radius`: Outer radius (default 1.0)
      - `inner_radius`: Inner (bore) radius, must be less than outer_radius (default 0.7)
      - `height`: Tube height (default 2.0)
      - `segments`: Number of radial segments (default 32)
    - Returns: Path to the output file
    - Raises: `ValueError` if `inner_radius >= outer_radius`

18. **`mirror_stl(path: str, output_path: str, axis: str) -> str`**
    - Mirrors (reflects) the mesh across the plane perpendicular to the given axis.
    - Useful for creating symmetric robot arm structures from a single side.
    - Args:
      - `path`: Input STL file path
      - `output_path`: Output STL file path
      - `axis`: Mirror axis ('x', 'y', or 'z')
    - Returns: Path to the output file
    - Raises: `ValueError` for invalid axis

19. **`rotate_stl_axis(path: str, output_path: str, ax: float, ay: float, az: float, angle: float) -> str`**
    - Rotates the mesh around an arbitrary axis vector using Rodrigues' rotation formula.
    - Essential for 6-DOF robot arm kinematic simulations where joints rotate
      around non-cardinal axes.
    - Args:
      - `path`: Input STL file path
      - `output_path`: Output STL file path
      - `ax`: X component of the rotation axis vector
      - `ay`: Y component of the rotation axis vector
      - `az`: Z component of the rotation axis vector
      - `angle`: Rotation angle in degrees
    - Returns: Path to the output file
    - Raises: `ValueError` if the axis vector is the zero vector

20. **`combine_stl(paths: list, output_path: str) -> str`**
    - Merges multiple STL files into a single STL file.
    - Used to assemble individual robot arm components (base, links, joints,
      end-effector) into a complete robot arm model.
    - Args:
      - `paths`: List of input STL file paths to merge
      - `output_path`: Output STL file path
    - Returns: Path to the output file
    - Raises: `ValueError` if `paths` is empty

21. **`create_pyramid(output_path: str, base_radius: float = 1.0, height: float = 2.0, segments: int = 4) -> str`**
    - Creates a regular pyramid mesh (apex at y=+height/2, base at y=-height/2).
    - Args:
      - `output_path`: Output STL file path
      - `base_radius`: Circumscribed radius of the base polygon (default 1.0)
      - `height`: Distance from base to apex (default 2.0)
      - `segments`: Number of base polygon sides (default 4 → square pyramid)
    - Returns: Path to the output file

22. **`create_prism(output_path: str, radius: float = 1.0, height: float = 2.0, segments: int = 6) -> str`**
    - Creates a regular n-sided prism mesh.
    - Args:
      - `output_path`: Output STL file path
      - `radius`: Circumscribed radius of the cross-section polygon (default 1.0)
      - `height`: Prism height along Y axis (default 2.0)
      - `segments`: Number of polygon sides (default 6 → hexagonal prism)
    - Returns: Path to the output file

23. **`create_hemisphere(output_path: str, radius: float = 1.0, segments: int = 32) -> str`**
    - Creates a hemisphere mesh (dome from y=0 to y=radius with a flat circular base at y=0).
    - Args:
      - `output_path`: Output STL file path
      - `radius`: Hemisphere radius (default 1.0)
      - `segments`: Number of latitude/longitude subdivisions (default 32)
    - Returns: Path to the output file

24. **`create_wedge(output_path: str, width: float = 1.0, height: float = 1.0, depth: float = 1.0) -> str`**
    - Creates a right-triangular wedge (triangular prism) mesh.
    - The right-angle corner is at the origin; width along X, height along Y, extruded along ±Z.
    - Args:
      - `output_path`: Output STL file path
      - `width`: Extent along X axis (default 1.0)
      - `height`: Extent along Y axis (default 1.0)
      - `depth`: Extent along Z axis (default 1.0)
    - Returns: Path to the output file

25. **`shear_stl(path: str, output_path: str, xy: float = 0.0, xz: float = 0.0, yx: float = 0.0, yz: float = 0.0, zx: float = 0.0, zy: float = 0.0) -> str`**
    - Applies a shear transformation to the mesh.
    - Each parameter shifts one axis by a given factor per unit along another axis.
    - Normals are transformed by the inverse-transpose of the shear matrix.
    - Args:
      - `path`: Input STL file path
      - `output_path`: Output STL file path
      - `xy`: X shear along Y (default 0.0)
      - `xz`: X shear along Z (default 0.0)
      - `yx`: Y shear along X (default 0.0)
      - `yz`: Y shear along Z (default 0.0)
      - `zx`: Z shear along X (default 0.0)
      - `zy`: Z shear along Y (default 0.0)
    - Returns: Path to the output file

26. **`create_bell_nozzle(output_path: str, throat_radius: float = 0.15, exit_radius: float = 0.75, chamber_radius: float = 0.35, chamber_length: float = 0.3, convergent_length: float = 0.2, bell_length: float = 1.0, wall_thickness: float = 0.04, segments: int = 32, profile_points: int = 16) -> str`**
    - Creates a convergent-divergent (de Laval) bell nozzle mesh.
    - The nozzle axis is aligned along Y. y=0 is the combustion-chamber inlet.
    - Profile: cylindrical chamber → cosine-blend convergent section → quarter-sine
      diverging bell section.
    - Mesh is a hollow thin-walled solid (outer + inner surface + two annular caps).
    - Essential component of any liquid rocket engine thrust chamber (e.g. RD-180).
    - Args:
      - `output_path`: Output STL file path
      - `throat_radius`: Nozzle throat (minimum) radius (default 0.15)
      - `exit_radius`: Nozzle exit plane radius (default 0.75)
      - `chamber_radius`: Combustion chamber bore radius (default 0.35)
      - `chamber_length`: Length of cylindrical chamber section (default 0.3)
      - `convergent_length`: Length of converging section (default 0.2)
      - `bell_length`: Length of diverging bell section (default 1.0)
      - `wall_thickness`: Nozzle wall thickness (default 0.04)
      - `segments`: Number of circumferential segments (default 32)
      - `profile_points`: Axial profile points per section (default 16)
    - Returns: Path to the output file
    - Raises: `ValueError` if `throat_radius >= chamber_radius`,
      `throat_radius >= exit_radius`, or `wall_thickness >= throat_radius`

27. **`create_injector_plate(output_path: str, radius: float = 0.35, thickness: float = 0.05, num_elements: int = 18, element_radius: float = 0.015, pattern_radius: float = 0.22, segments: int = 32) -> str`**
    - Creates a propellant injector plate mesh.
    - A solid flat disc (aligned along Y) with a circular array of small cylindrical
      injector-element stubs protruding from the top face (+Y direction).
    - Models the injector head of a liquid rocket combustion chamber (e.g. RD-180).
    - Args:
      - `output_path`: Output STL file path
      - `radius`: Outer radius of the plate (default 0.35)
      - `thickness`: Plate thickness along Y (default 0.05)
      - `num_elements`: Number of injector-element stubs around the ring (default 18)
      - `element_radius`: Radius of each cylindrical injector stub (default 0.015)
      - `pattern_radius`: Radial distance from plate centre to stub centreline (default 0.22)
      - `segments`: Circumferential segments for plate and stubs (default 32)
    - Returns: Path to the output file
    - Raises: `ValueError` if `pattern_radius + element_radius >= radius`

28. **`create_pump_housing(output_path: str, bore_radius: float = 0.25, housing_radius: float = 0.6, housing_height: float = 0.35, outlet_radius: float = 0.12, outlet_length: float = 0.25, segments: int = 32) -> str`**
    - Creates a centrifugal pump housing (volute casing) mesh.
    - A hollow cylindrical casing with the pump axis along Y, plus a cylindrical
      discharge outlet pipe extending in the +X direction.
    - Models the turbopump casing used in rocket engines such as the RD-180.
      Combine with `create_turbine_disk` and `create_turbine_blade` + `array_circular`
      to assemble a complete turbopump stage.
    - Args:
      - `output_path`: Output STL file path
      - `bore_radius`: Inner bore radius for the impeller cavity (default 0.25)
      - `housing_radius`: Outer casing radius (default 0.6)
      - `housing_height`: Axial height of the casing along Y (default 0.35)
      - `outlet_radius`: Radius of the discharge outlet pipe (default 0.12)
      - `outlet_length`: Length of the outlet pipe along X (default 0.25)
      - `segments`: Number of circumferential segments (default 32)
    - Returns: Path to the output file
    - Raises: `ValueError` if `bore_radius >= housing_radius` or
      `outlet_radius >= housing_radius`



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
10. Invalid axis for rotation/mirror - raise ValueError
11. Zero vector for arbitrary-axis rotation - raise ValueError
12. inner_radius >= outer_radius for tube - raise ValueError
13. Empty path list for combine_stl - raise ValueError

## Performance & Constraints

- Must handle files up to 100MB efficiently
- Face count extraction should be O(1) for binary STL (seek to byte 80)
- Memory usage should be reasonable (<2x file size)
- Use numpy for vectorized operations where possible

