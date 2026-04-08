import os

import fastmcp

from mcp_stl import _core


def create_mcp_server() -> fastmcp.FastMCP:
    mcp = fastmcp.FastMCP("mcp-stl")

    @mcp.tool()
    def read_stl_file(path: str) -> dict[str, object]:
        """Reads an STL file and returns mesh data including vertices, normals, and bounding box.

        Args:
            path: Path to the STL file.

        Returns:
            Dictionary containing: vertex_count, face_count, normals (as nested lists),
            vertices (as nested lists), bounding_box, and format (ascii|binary).

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the file is not a valid STL file.

        Example:
            >>> read_stl_file("/path/to/model.stl")
            {"vertex_count": 36, "face_count": 12, ...}
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"STL file not found: {path}")

        mesh = _core.read_stl_file(path)

        return {
            "vertex_count": len(mesh.vertices),
            "face_count": mesh.face_count,
            "normals": mesh.normals.tolist(),
            "vertices": mesh.vertices.tolist(),
            "bounding_box": mesh.bounding_box,
            "format": mesh.format,
        }

    @mcp.tool()
    def get_mesh_info(path: str) -> dict[str, object]:
        """Returns summary information about an STL file without loading full geometry.

        Args:
            path: Path to the STL file.

        Returns:
            Dictionary containing: face_count, bounding_box, center, and format.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the file is not a valid STL file.

        Example:
            >>> get_mesh_info("/path/to/model.stl")
            {"face_count": 12, "bounding_box": {...}, "center": {...}, "format": "binary"}
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"STL file not found: {path}")

        return _core.get_mesh_info(path)

    @mcp.tool()
    def translate_stl(path: str, output_path: str, x: float, y: float, z: float) -> str:
        """Translates (moves) the mesh by the specified offset.

        Args:
            path: Input STL file path.
            output_path: Output STL file path.
            x: Translation along X axis.
            y: Translation along Y axis.
            z: Translation along Z axis.

        Returns:
            Path to the output file.

        Raises:
            FileNotFoundError: If the input file does not exist.
            ValueError: If the input file is not a valid STL.

        Example:
            >>> translate_stl("input.stl", "output.stl", 10.0, 0.0, 5.0)
            "output.stl"
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"STL file not found: {path}")

        return _core.translate_stl(path, output_path, x, y, z)

    @mcp.tool()
    def rotate_stl(path: str, output_path: str, axis: str, angle: float) -> str:
        """Rotates the mesh around a specified axis.

        Args:
            path: Input STL file path.
            output_path: Output STL file path.
            axis: Rotation axis ('x', 'y', or 'z').
            angle: Rotation angle in degrees.

        Returns:
            Path to the output file.

        Raises:
            FileNotFoundError: If the input file does not exist.
            ValueError: If axis is invalid or input file is not a valid STL.

        Example:
            >>> rotate_stl("input.stl", "output.stl", "y", 90.0)
            "output.stl"
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"STL file not found: {path}")

        return _core.rotate_stl(path, output_path, axis, angle)

    @mcp.tool()
    def scale_stl(path: str, output_path: str, x: float, y: float, z: float) -> str:
        """Scales the mesh by the specified factors.

        Args:
            path: Input STL file path.
            output_path: Output STL file path.
            x: Scale factor for X axis.
            y: Scale factor for Y axis.
            z: Scale factor for Z axis.

        Returns:
            Path to the output file.

        Raises:
            FileNotFoundError: If the input file does not exist.
            ValueError: If the input file is not a valid STL.

        Example:
            >>> scale_stl("input.stl", "output.stl", 2.0, 2.0, 2.0)
            "output.stl"
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"STL file not found: {path}")

        return _core.scale_stl(path, output_path, x, y, z)

    @mcp.tool()
    def write_stl(
        vertices: list[list[float]],
        normals: list[list[float]],
        output_path: str,
        format: str = "binary",
    ) -> str:
        """Writes mesh data to an STL file.

        Args:
            vertices: List of vertices as [x, y, z] triplets.
            normals: List of face normals as [x, y, z] triplets.
            output_path: Output STL file path.
            format: Output format ('ascii' or 'binary'). Defaults to 'binary'.

        Returns:
            Path to the output file.

        Example:
            >>> write_stl([[0,0,0], [1,0,0], [0,1,0]], [[0,0,1]], "output.stl")
            "output.stl"
        """
        return _core.write_stl(vertices, normals, output_path, format)

    @mcp.tool()
    def create_cube(
        output_path: str, size: float = 1.0, center: list[float] | None = None
    ) -> str:
        """Creates a cube mesh.

        Args:
            output_path: Output STL file path.
            size: Edge length (default 1.0).
            center: Center position as [x, y, z] (default [0, 0, 0]).

        Returns:
            Path to the output file.

        Example:
            >>> create_cube("cube.stl", size=2.0)
            "cube.stl"
        """
        return _core.create_cube(output_path, size, center)

    @mcp.tool()
    def create_sphere(output_path: str, radius: float = 1.0, segments: int = 32) -> str:
        """Creates a sphere mesh.

        Args:
            output_path: Output STL file path.
            radius: Sphere radius (default 1.0).
            segments: Number of horizontal segments (default 32).

        Returns:
            Path to the output file.

        Example:
            >>> create_sphere("sphere.stl", radius=0.5, segments=16)
            "sphere.stl"
        """
        return _core.create_sphere(output_path, radius, segments)

    @mcp.tool()
    def create_cylinder(
        output_path: str,
        radius: float = 1.0,
        height: float = 2.0,
        segments: int = 32,
    ) -> str:
        """Creates a cylinder mesh.

        Args:
            output_path: Output STL file path.
            radius: Cylinder radius (default 1.0).
            height: Cylinder height (default 2.0).
            segments: Number of radial segments (default 32).

        Returns:
            Path to the output file.

        Example:
            >>> create_cylinder("cylinder.stl", radius=0.5, height=3.0)
            "cylinder.stl"
        """
        return _core.create_cylinder(output_path, radius, height, segments)

    @mcp.tool()
    def create_cone(
        output_path: str,
        radius: float = 1.0,
        height: float = 2.0,
        segments: int = 32,
    ) -> str:
        """Creates a cone mesh.

        Args:
            output_path: Output STL file path.
            radius: Base radius (default 1.0).
            height: Cone height (default 2.0).
            segments: Number of radial segments (default 32).

        Returns:
            Path to the output file.

        Example:
            >>> create_cone("cone.stl", radius=0.5, height=2.0)
            "cone.stl"
        """
        return _core.create_cone(output_path, radius, height, segments)

    @mcp.tool()
    def create_torus(
        output_path: str,
        major_radius: float = 1.0,
        minor_radius: float = 0.3,
        major_segments: int = 32,
        minor_segments: int = 16,
    ) -> str:
        """Creates a torus (donut) mesh.

        Args:
            output_path: Output STL file path.
            major_radius: Distance from center to tube center (default 1.0).
            minor_radius: Tube radius (default 0.3).
            major_segments: Number of segments around the ring (default 32).
            minor_segments: Number of segments around the tube (default 16).

        Returns:
            Path to the output file.

        Example:
            >>> create_torus("torus.stl", major_radius=2.0, minor_radius=0.5)
            "torus.stl"
        """
        return _core.create_torus(
            output_path, major_radius, minor_radius, major_segments, minor_segments
        )

    @mcp.tool()
    def create_plane(output_path: str, width: float = 1.0, height: float = 1.0) -> str:
        """Creates a plane mesh (2D flat surface).

        Args:
            output_path: Output STL file path.
            width: Plane width (default 1.0).
            height: Plane height (default 1.0).

        Returns:
            Path to the output file.

        Example:
            >>> create_plane("plane.stl", width=5.0, height=3.0)
            "plane.stl"
        """
        return _core.create_plane(output_path, width, height)

    @mcp.tool()
    def create_box(
        output_path: str,
        width: float = 1.0,
        height: float = 1.0,
        depth: float = 1.0,
        center: list[float] | None = None,
    ) -> str:
        """Creates a rectangular box (cuboid) mesh with independent dimensions.

        Args:
            output_path: Output STL file path.
            width: Box width along X axis (default 1.0).
            height: Box height along Y axis (default 1.0).
            depth: Box depth along Z axis (default 1.0).
            center: Center position as [x, y, z] (default [0, 0, 0]).

        Returns:
            Path to the output file.

        Example:
            >>> create_box("box.stl", width=2.0, height=0.5, depth=1.5)
            "box.stl"
        """
        return _core.create_box(output_path, width, height, depth, center)

    @mcp.tool()
    def create_capsule(
        output_path: str,
        radius: float = 0.5,
        height: float = 2.0,
        segments: int = 32,
    ) -> str:
        """Creates a capsule mesh (cylinder with hemispherical end caps).

        Useful for robot arm link bodies with smooth rounded ends.

        Args:
            output_path: Output STL file path.
            radius: Radius of the cylinder and hemispheres (default 0.5).
            height: Height of the cylindrical section only (default 2.0).
            segments: Number of radial/latitude segments (default 32).

        Returns:
            Path to the output file.

        Example:
            >>> create_capsule("link.stl", radius=0.3, height=1.5)
            "link.stl"
        """
        return _core.create_capsule(output_path, radius, height, segments)

    @mcp.tool()
    def create_ellipsoid(
        output_path: str,
        rx: float = 1.0,
        ry: float = 0.5,
        rz: float = 0.75,
        segments: int = 32,
    ) -> str:
        """Creates an ellipsoid mesh with independent radii on each axis.

        Useful for joint housings and actuator enclosures in robot arm designs.

        Args:
            output_path: Output STL file path.
            rx: Radius along X axis (default 1.0).
            ry: Radius along Y axis (default 0.5).
            rz: Radius along Z axis (default 0.75).
            segments: Number of latitude/longitude segments (default 32).

        Returns:
            Path to the output file.

        Example:
            >>> create_ellipsoid("joint.stl", rx=0.8, ry=0.4, rz=0.6)
            "joint.stl"
        """
        return _core.create_ellipsoid(output_path, rx, ry, rz, segments)

    @mcp.tool()
    def create_frustum(
        output_path: str,
        bottom_radius: float = 1.0,
        top_radius: float = 0.5,
        height: float = 2.0,
        segments: int = 32,
    ) -> str:
        """Creates a frustum (truncated cone) mesh.

        Useful for tapered robot arm segments transitioning between joint diameters.

        Args:
            output_path: Output STL file path.
            bottom_radius: Radius of the bottom circle (default 1.0).
            top_radius: Radius of the top circle (default 0.5).
            height: Frustum height (default 2.0).
            segments: Number of radial segments (default 32).

        Returns:
            Path to the output file.

        Example:
            >>> create_frustum("taper.stl", bottom_radius=0.8, top_radius=0.4, height=1.5)
            "taper.stl"
        """
        return _core.create_frustum(output_path, bottom_radius, top_radius, height, segments)

    @mcp.tool()
    def create_tube(
        output_path: str,
        outer_radius: float = 1.0,
        inner_radius: float = 0.7,
        height: float = 2.0,
        segments: int = 32,
    ) -> str:
        """Creates a hollow cylinder (tube) mesh.

        Useful for hollow structural link bodies in robot arm designs.

        Args:
            output_path: Output STL file path.
            outer_radius: Outer radius (default 1.0).
            inner_radius: Inner (bore) radius, must be less than outer_radius (default 0.7).
            height: Tube height (default 2.0).
            segments: Number of radial segments (default 32).

        Returns:
            Path to the output file.

        Raises:
            ValueError: If inner_radius >= outer_radius.

        Example:
            >>> create_tube("link.stl", outer_radius=0.5, inner_radius=0.35, height=3.0)
            "link.stl"
        """
        return _core.create_tube(output_path, outer_radius, inner_radius, height, segments)

    @mcp.tool()
    def mirror_stl(path: str, output_path: str, axis: str) -> str:
        """Mirrors (reflects) the mesh across the plane perpendicular to the given axis.

        Useful for creating symmetric robot arm structures from a single side.

        Args:
            path: Input STL file path.
            output_path: Output STL file path.
            axis: Mirror axis ('x', 'y', or 'z'). The mesh is reflected across
                  the plane perpendicular to this axis passing through the origin.

        Returns:
            Path to the output file.

        Raises:
            FileNotFoundError: If the input file does not exist.
            ValueError: If axis is invalid or input file is not a valid STL.

        Example:
            >>> mirror_stl("input.stl", "mirrored.stl", "x")
            "mirrored.stl"
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"STL file not found: {path}")

        return _core.mirror_stl(path, output_path, axis)

    @mcp.tool()
    def rotate_stl_axis(
        path: str,
        output_path: str,
        ax: float,
        ay: float,
        az: float,
        angle: float,
    ) -> str:
        """Rotates the mesh around an arbitrary axis vector using Rodrigues' rotation.

        Essential for 6-DOF robot arm kinematic simulations where joints rotate
        around non-cardinal axes.

        Args:
            path: Input STL file path.
            output_path: Output STL file path.
            ax: X component of the rotation axis vector.
            ay: Y component of the rotation axis vector.
            az: Z component of the rotation axis vector.
            angle: Rotation angle in degrees.

        Returns:
            Path to the output file.

        Raises:
            FileNotFoundError: If the input file does not exist.
            ValueError: If the axis vector is the zero vector or the file is invalid.

        Example:
            >>> rotate_stl_axis("input.stl", "output.stl", 1.0, 1.0, 0.0, 45.0)
            "output.stl"
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"STL file not found: {path}")

        return _core.rotate_stl_axis(path, output_path, ax, ay, az, angle)

    @mcp.tool()
    def combine_stl(paths: list[str], output_path: str) -> str:
        """Merges multiple STL files into a single STL file.

        Used to assemble individual robot arm components (base, links, joints,
        end-effector) into a complete robot arm model.

        Args:
            paths: List of input STL file paths to merge.
            output_path: Output STL file path.

        Returns:
            Path to the output file.

        Raises:
            FileNotFoundError: If any input file does not exist.
            ValueError: If paths is empty or any file is not a valid STL.

        Example:
            >>> combine_stl(["base.stl", "link1.stl", "link2.stl"], "robot.stl")
            "robot.stl"
        """
        for p in paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"STL file not found: {p}")

        return _core.combine_stl(paths, output_path)

    @mcp.resource("stl://{filepath}")
    def get_stl_info(filepath: str) -> dict[str, object]:
        """Resource URI to get mesh information for an STL file.

        Args:
            filepath: Path to the STL file.

        Returns:
            Mesh metadata as JSON dictionary.

        Example:
            Access via stl:///path/to/model.stl
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"STL file not found: {filepath}")

        return _core.get_mesh_info(filepath)

    return mcp
