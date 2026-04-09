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

    @mcp.tool()
    def create_gear(
        output_path: str,
        module: float = 1.0,
        teeth: int = 20,
        thickness: float = 1.0,
        pressure_angle_deg: float = 20.0,
        segments_per_tooth: int = 4,
    ) -> str:
        """Creates a spur gear mesh (used for timing gears, cam drives, etc.).

        Args:
            output_path: Output STL file path.
            module: Gear module — pitch diameter / tooth count (default 1.0).
            teeth: Number of teeth (default 20, minimum 4).
            thickness: Face width along Y axis (default 1.0).
            pressure_angle_deg: Pressure angle in degrees (default 20.0).
            segments_per_tooth: Polygon segments per tooth (default 4).

        Returns:
            Path to the output file.

        Raises:
            ValueError: If teeth < 4.

        Example:
            >>> create_gear("gear.stl", module=2.0, teeth=16, thickness=1.5)
            "gear.stl"
        """
        return _core.create_gear(
            output_path, module, teeth, thickness, pressure_angle_deg, segments_per_tooth
        )

    @mcp.tool()
    def create_spring(
        output_path: str,
        coil_radius: float = 1.0,
        wire_radius: float = 0.1,
        turns: float = 5.0,
        height: float = 5.0,
        segments: int = 32,
        wire_segments: int = 8,
    ) -> str:
        """Creates a helical coil spring mesh (used for valve springs, etc.).

        The spring axis is along Y, centred at the origin.

        Args:
            output_path: Output STL file path.
            coil_radius: Radius of the coil helix centre line (default 1.0).
            wire_radius: Radius of the wire cross-section (default 0.1).
            turns: Number of coil turns (default 5.0).
            height: Total spring height along Y axis (default 5.0).
            segments: Path segments per full turn (default 32).
            wire_segments: Polygon sides on wire cross-section (default 8).

        Returns:
            Path to the output file.

        Example:
            >>> create_spring("spring.stl", coil_radius=0.8, wire_radius=0.08, turns=6)
            "spring.stl"
        """
        return _core.create_spring(
            output_path, coil_radius, wire_radius, turns, height, segments, wire_segments
        )

    @mcp.tool()
    def create_connecting_rod(
        output_path: str,
        length: float = 6.0,
        big_end_outer_radius: float = 1.0,
        big_end_inner_radius: float = 0.6,
        small_end_outer_radius: float = 0.6,
        small_end_inner_radius: float = 0.35,
        beam_width: float = 0.4,
        beam_height: float = 0.8,
        segments: int = 32,
    ) -> str:
        """Creates a simplified connecting rod mesh.

        The rod is aligned along Y. Big end (crankshaft side) at y = -length/2,
        small end (piston-pin side) at y = +length/2.

        Args:
            output_path: Output STL file path.
            length: Centre-to-centre distance between end bores (default 6.0).
            big_end_outer_radius: Outer radius of the big end (default 1.0).
            big_end_inner_radius: Inner bore radius of the big end (default 0.6).
            small_end_outer_radius: Outer radius of the small end (default 0.6).
            small_end_inner_radius: Inner bore radius of the small end (default 0.35).
            beam_width: Width of connecting beam along X (default 0.4).
            beam_height: Depth of connecting beam along Z (default 0.8).
            segments: Radial segments for end bores (default 32).

        Returns:
            Path to the output file.

        Raises:
            ValueError: If inner radius >= outer radius for either end.

        Example:
            >>> create_connecting_rod("rod.stl", length=8.0)
            "rod.stl"
        """
        return _core.create_connecting_rod(
            output_path,
            length,
            big_end_outer_radius,
            big_end_inner_radius,
            small_end_outer_radius,
            small_end_inner_radius,
            beam_width,
            beam_height,
            segments,
        )

    @mcp.tool()
    def create_crankshaft(
        output_path: str,
        throws: int = 4,
        main_journal_radius: float = 0.5,
        rod_journal_radius: float = 0.4,
        journal_width: float = 0.4,
        crank_arm_thickness: float = 0.25,
        stroke: float = 2.0,
        segments: int = 32,
    ) -> str:
        """Creates a simplified crankshaft mesh.

        Main axis along Y. Main journals on the Y axis; crank pins offset
        radially by stroke/2, evenly distributed around the Y axis.

        Args:
            output_path: Output STL file path.
            throws: Number of crank throws / cylinders (default 4).
            main_journal_radius: Radius of the main bearing journals (default 0.5).
            rod_journal_radius: Radius of crank pins / rod journals (default 0.4).
            journal_width: Axial width of each journal (default 0.4).
            crank_arm_thickness: Axial thickness of crank arm discs (default 0.25).
            stroke: Piston stroke; pin offset = stroke/2 (default 2.0).
            segments: Radial segments (default 32).

        Returns:
            Path to the output file.

        Raises:
            ValueError: If throws < 1.

        Example:
            >>> create_crankshaft("crank.stl", throws=4, stroke=3.0)
            "crank.stl"
        """
        return _core.create_crankshaft(
            output_path,
            throws,
            main_journal_radius,
            rod_journal_radius,
            journal_width,
            crank_arm_thickness,
            stroke,
            segments,
        )

    @mcp.tool()
    def create_valve(
        output_path: str,
        stem_radius: float = 0.15,
        stem_length: float = 3.0,
        head_radius: float = 0.6,
        head_height: float = 0.15,
        segments: int = 32,
    ) -> str:
        """Creates a poppet valve mesh (intake or exhaust valve).

        Stem tip at y = +stem_length/2; head face at y = -(stem_length/2 + head_height).

        Args:
            output_path: Output STL file path.
            stem_radius: Radius of the valve stem (default 0.15).
            stem_length: Length of the cylindrical stem (default 3.0).
            head_radius: Outer radius of the valve head disc (default 0.6).
            head_height: Axial thickness of the valve head (default 0.15).
            segments: Radial segments (default 32).

        Returns:
            Path to the output file.

        Raises:
            ValueError: If stem_radius >= head_radius.

        Example:
            >>> create_valve("valve.stl", stem_radius=0.1, head_radius=0.5)
            "valve.stl"
        """
        return _core.create_valve(
            output_path, stem_radius, stem_length, head_radius, head_height, segments
        )

    @mcp.tool()
    def create_camshaft_lobe(
        output_path: str,
        base_radius: float = 0.8,
        lift: float = 0.4,
        lobe_width: float = 0.8,
        segments: int = 64,
    ) -> str:
        """Creates a cam lobe mesh for camshaft design.

        The lobe is a disc with an eccentric nose on the +X side; axis along Y.

        Args:
            output_path: Output STL file path.
            base_radius: Base-circle radius (default 0.8).
            lift: Maximum lift — nose height above base circle (default 0.4).
            lobe_width: Axial width along Y (default 0.8).
            segments: Circumferential polygon segments (default 64).

        Returns:
            Path to the output file.

        Example:
            >>> create_camshaft_lobe("lobe.stl", base_radius=1.0, lift=0.5)
            "lobe.stl"
        """
        return _core.create_camshaft_lobe(output_path, base_radius, lift, lobe_width, segments)

    @mcp.tool()
    def array_linear(
        path: str, output_path: str, count: int, dx: float, dy: float, dz: float
    ) -> str:
        """Creates a linear array of mesh copies.

        Produces `count` total copies spaced by (dx, dy, dz) per step.
        Useful for placing multiple cylinders in an engine bank.

        Args:
            path: Input STL file path.
            output_path: Output STL file path.
            count: Total copies including the original (minimum 1).
            dx: X offset per step.
            dy: Y offset per step.
            dz: Z offset per step.

        Returns:
            Path to the output file.

        Raises:
            FileNotFoundError: If the input file does not exist.
            ValueError: If count < 1.

        Example:
            >>> array_linear("cylinder.stl", "cylinders.stl", 4, 3.0, 0.0, 0.0)
            "cylinders.stl"
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"STL file not found: {path}")

        return _core.array_linear(path, output_path, count, dx, dy, dz)

    @mcp.tool()
    def array_circular(
        path: str, output_path: str, count: int, axis: str = "y"
    ) -> str:
        """Creates a circular array of mesh copies around a coordinate axis.

        Produces `count` copies at equal angular intervals (360/count degrees)
        around the world-origin axis. Useful for valve arrangements and bolt
        patterns.

        Args:
            path: Input STL file path.
            output_path: Output STL file path.
            count: Total copies (minimum 1).
            axis: Rotation axis — 'x', 'y', or 'z' (default 'y').

        Returns:
            Path to the output file.

        Raises:
            FileNotFoundError: If the input file does not exist.
            ValueError: If count < 1 or axis is invalid.

        Example:
            >>> array_circular("bolt_hole.stl", "bolt_pattern.stl", 6, axis="y")
            "bolt_pattern.stl"
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"STL file not found: {path}")

        return _core.array_circular(path, output_path, count, axis)

    @mcp.tool()
    def create_pyramid(
        output_path: str,
        base_radius: float = 1.0,
        height: float = 2.0,
        segments: int = 4,
    ) -> str:
        """Creates a regular pyramid mesh.

        The pyramid has a regular n-sided polygon base at y=-height/2 and
        a single apex at y=+height/2.

        Args:
            output_path: Output STL file path.
            base_radius: Circumscribed radius of the base polygon (default 1.0).
            height: Distance from base to apex (default 2.0).
            segments: Number of base polygon sides (default 4 → square base).

        Returns:
            Path to the output file.

        Example:
            >>> create_pyramid("pyramid.stl", base_radius=1.0, height=2.0)
            "pyramid.stl"
        """
        return _core.create_pyramid(output_path, base_radius, height, segments)

    @mcp.tool()
    def create_prism(
        output_path: str,
        radius: float = 1.0,
        height: float = 2.0,
        segments: int = 6,
    ) -> str:
        """Creates a regular n-sided prism mesh.

        The prism has a regular n-gon cross-section with the given circumscribed
        radius, extruded along Y from -height/2 to +height/2.

        Args:
            output_path: Output STL file path.
            radius: Circumscribed radius of the cross-section (default 1.0).
            height: Prism height along Y (default 2.0).
            segments: Number of polygon sides (default 6 → hexagonal prism).

        Returns:
            Path to the output file.

        Example:
            >>> create_prism("hexprism.stl", radius=1.0, height=2.0, segments=6)
            "hexprism.stl"
        """
        return _core.create_prism(output_path, radius, height, segments)

    @mcp.tool()
    def create_hemisphere(
        output_path: str,
        radius: float = 1.0,
        segments: int = 32,
    ) -> str:
        """Creates a hemisphere mesh (upper half of a sphere with a flat base).

        The dome covers y ≥ 0.  A flat disc at y=0 closes the solid.

        Args:
            output_path: Output STL file path.
            radius: Hemisphere radius (default 1.0).
            segments: Number of latitude/longitude subdivisions (default 32).

        Returns:
            Path to the output file.

        Example:
            >>> create_hemisphere("dome.stl", radius=1.5)
            "dome.stl"
        """
        return _core.create_hemisphere(output_path, radius, segments)

    @mcp.tool()
    def create_wedge(
        output_path: str,
        width: float = 1.0,
        height: float = 1.0,
        depth: float = 1.0,
    ) -> str:
        """Creates a right-triangular wedge (triangular prism) mesh.

        The right-angle corner sits at the origin; width extends along X,
        height along Y, and the prism is extruded symmetrically along ±Z
        by depth/2.

        Args:
            output_path: Output STL file path.
            width: Extent along X (default 1.0).
            height: Extent along Y (default 1.0).
            depth: Extent along Z (default 1.0).

        Returns:
            Path to the output file.

        Example:
            >>> create_wedge("wedge.stl", width=2.0, height=1.0, depth=3.0)
            "wedge.stl"
        """
        return _core.create_wedge(output_path, width, height, depth)

    @mcp.tool()
    def shear_stl(
        path: str,
        output_path: str,
        xy: float = 0.0,
        xz: float = 0.0,
        yx: float = 0.0,
        yz: float = 0.0,
        zx: float = 0.0,
        zy: float = 0.0,
    ) -> str:
        """Applies a shear transformation to the mesh.

        Each parameter shifts one axis by a given factor per unit along another
        axis (e.g. xy shifts X by xy units for every unit along Y).

        Args:
            path: Input STL file path.
            output_path: Output STL file path.
            xy: X shear along Y (default 0.0).
            xz: X shear along Z (default 0.0).
            yx: Y shear along X (default 0.0).
            yz: Y shear along Z (default 0.0).
            zx: Z shear along X (default 0.0).
            zy: Z shear along Y (default 0.0).

        Returns:
            Path to the output file.

        Raises:
            FileNotFoundError: If the input file does not exist.

        Example:
            >>> shear_stl("cube.stl", "sheared.stl", xy=0.5)
            "sheared.stl"
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"STL file not found: {path}")

        return _core.shear_stl(path, output_path, xy, xz, yx, yz, zx, zy)

    @mcp.tool()
    def twist_stl(
        path: str,
        output_path: str,
        angle: float,
        axis: str = "y",
    ) -> str:
        """Applies a twist (torsion) transformation to the mesh.

        Each vertex is rotated about *axis* by an amount proportional to its
        position along that axis.  The twist is zero at the minimum extent of
        the mesh and reaches *angle* degrees at the maximum extent.

        Useful for adding geometric pitch to propeller blades, turbine blades,
        and swept wing sections created with create_airfoil.

        Args:
            path: Input STL file path.
            output_path: Output STL file path.
            angle: Total twist in degrees over the full extent along *axis*.
            axis: Twist axis ('x', 'y', or 'z'; default 'y').

        Returns:
            Path to the output file.

        Raises:
            FileNotFoundError: If the input file does not exist.
            ValueError: If axis is invalid.

        Example:
            >>> twist_stl("wing.stl", "twisted_wing.stl", 5.0, axis="z")
            "twisted_wing.stl"
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"STL file not found: {path}")

        return _core.twist_stl(path, output_path, angle, axis)

    @mcp.tool()
    def create_airfoil(
        output_path: str,
        chord: float = 1.0,
        span: float = 5.0,
        thickness_ratio: float = 0.12,
        segments: int = 32,
    ) -> str:
        """Creates a symmetric NACA airfoil wing-section mesh.

        The chord runs along X (leading edge at x=0, trailing edge at
        x=chord), thickness is along Y, and the span extrudes along Z from
        z=0 to z=span.  A thickness_ratio of 0.12 gives a NACA 0012 profile.

        Suitable for wings, horizontal stabilisers, and vertical fins of
        airplanes or helicopters.

        Args:
            output_path: Output STL file path.
            chord: Chord length along X (default 1.0).
            span: Wing-section span along Z (default 5.0).
            thickness_ratio: Max thickness as fraction of chord (default 0.12).
            segments: Points per airfoil surface half (default 32).

        Returns:
            Path to the output file.

        Example:
            >>> create_airfoil("wing.stl", chord=1.5, span=6.0)
            "wing.stl"
        """
        return _core.create_airfoil(output_path, chord, span, thickness_ratio, segments)

    @mcp.tool()
    def create_propeller_blade(
        output_path: str,
        length: float = 5.0,
        chord_root: float = 0.5,
        chord_tip: float = 0.15,
        twist_angle: float = 30.0,
        thickness_ratio: float = 0.12,
        segments: int = 16,
        span_segments: int = 20,
    ) -> str:
        """Creates a propeller or helicopter rotor blade mesh.

        The blade spans along Y (root at y=0, tip at y=length).  The
        cross-section is a NACA symmetric airfoil in the X-Z plane, scaled
        by the local chord and progressively rotated about Y by twist_angle
        from root to tip.

        Use array_circular to arrange multiple blades into a full propeller
        or rotor disc.

        Args:
            output_path: Output STL file path.
            length: Blade span along Y (default 5.0).
            chord_root: Chord length at the root (default 0.5).
            chord_tip: Chord length at the tip (default 0.15).
            twist_angle: Total twist from root to tip in degrees (default 30.0).
            thickness_ratio: NACA airfoil thickness ratio (default 0.12).
            segments: Points per airfoil surface half per slice (default 16).
            span_segments: Number of span-wise divisions (default 20).

        Returns:
            Path to the output file.

        Example:
            >>> create_propeller_blade("blade.stl", length=4.0, twist_angle=25.0)
            "blade.stl"
        """
        return _core.create_propeller_blade(
            output_path, length, chord_root, chord_tip, twist_angle,
            thickness_ratio, segments, span_segments,
        )

    @mcp.tool()
    def create_turbine_blade(
        output_path: str,
        span: float = 1.5,
        chord_root: float = 0.4,
        chord_tip: float = 0.25,
        twist_angle: float = 45.0,
        thickness_ratio: float = 0.10,
        segments: int = 16,
        span_segments: int = 16,
    ) -> str:
        """Creates a gas-turbine compressor or fan blade mesh.

        Geometry is identical to a propeller blade (NACA airfoil profile,
        tapered chord, and progressive twist) but with defaults suited to
        turbomachinery: shorter span, more aggressive twist, and thinner
        profile.

        Use array_circular to arrange blades into a full compressor or fan
        stage.

        Args:
            output_path: Output STL file path.
            span: Blade span along Y (default 1.5).
            chord_root: Chord at the root (default 0.4).
            chord_tip: Chord at the tip (default 0.25).
            twist_angle: Total twist from root to tip in degrees (default 45.0).
            thickness_ratio: NACA airfoil thickness ratio (default 0.10).
            segments: Points per airfoil surface half per slice (default 16).
            span_segments: Number of span-wise divisions (default 16).

        Returns:
            Path to the output file.

        Example:
            >>> create_turbine_blade("tblade.stl", span=1.2, twist_angle=50.0)
            "tblade.stl"
        """
        return _core.create_turbine_blade(
            output_path, span, chord_root, chord_tip, twist_angle,
            thickness_ratio, segments, span_segments,
        )

    @mcp.tool()
    def create_piston(
        output_path: str,
        bore: float = 1.0,
        height: float = 1.2,
        wall_thickness: float = 0.1,
        crown_height: float = 0.3,
        segments: int = 32,
    ) -> str:
        """Creates a hollow piston mesh.

        The piston axis is aligned along Y.  The crown (solid top) is at
        y = +height/2 with axial thickness crown_height.  The cylindrical
        skirt extends from y = -height/2 to the crown base; the interior is
        open at the bottom.

        Combine with create_connecting_rod and create_crankshaft to model a
        piston engine.

        Args:
            output_path: Output STL file path.
            bore: Outer piston diameter (default 1.0).
            height: Total piston height along Y (default 1.2).
            wall_thickness: Cylindrical wall thickness (default 0.1).
            crown_height: Axial thickness of the solid crown (default 0.3).
            segments: Number of radial segments (default 32).

        Returns:
            Path to the output file.

        Raises:
            ValueError: If wall_thickness >= bore/2 or crown_height >= height.

        Example:
            >>> create_piston("piston.stl", bore=0.8, height=1.0)
            "piston.stl"
        """
        return _core.create_piston(
            output_path, bore, height, wall_thickness, crown_height, segments
        )

    @mcp.tool()
    def create_turbine_disk(
        output_path: str,
        disk_radius: float = 2.0,
        bore_radius: float = 0.5,
        disk_thickness: float = 0.4,
        web_thickness: float = 0.15,
        hub_radius: float = 0.9,
        segments: int = 64,
    ) -> str:
        """Creates a turbine disk (rotor wheel blank) mesh.

        The disk has a stepped cross-section typical of gas-turbine rotors:
        a hub region (bore to hub_radius) at full disk_thickness, and a web
        region (hub_radius to disk_radius) at the reduced web_thickness.

        The axis is aligned along Y.  Attach turbine blades using
        create_turbine_blade followed by array_circular.

        Args:
            output_path: Output STL file path.
            disk_radius: Outer rim radius (default 2.0).
            bore_radius: Inner bore radius (default 0.5).
            disk_thickness: Full axial thickness of the hub (default 0.4).
            web_thickness: Reduced axial thickness of the web (default 0.15).
            hub_radius: Radial boundary between hub and web (default 0.9).
            segments: Number of radial segments (default 64).

        Returns:
            Path to the output file.

        Raises:
            ValueError: If bore_radius >= hub_radius, hub_radius >=
                disk_radius, or web_thickness >= disk_thickness.

        Example:
            >>> create_turbine_disk("disk.stl", disk_radius=2.0, bore_radius=0.4)
            "disk.stl"
        """
        return _core.create_turbine_disk(
            output_path, disk_radius, bore_radius, disk_thickness,
            web_thickness, hub_radius, segments,
        )

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
