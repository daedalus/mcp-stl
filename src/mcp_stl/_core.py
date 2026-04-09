import struct
from dataclasses import dataclass
from typing import TextIO

import numpy as np
import numpy.typing as npt


@dataclass
class MeshData:
    vertices: npt.NDArray[np.float32]
    normals: npt.NDArray[np.float32]
    face_count: int
    bounding_box: dict[str, tuple[float, float]]
    format: str


def _parse_ascii(file: TextIO) -> MeshData:
    vertices: list[list[float]] = []
    normals: list[list[float]] = []

    for line in file:
        line = line.strip()
        if line.startswith("vertex"):
            parts = line.split()
            if len(parts) == 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith("facet normal"):
            parts = line.split()
            if len(parts) == 5:
                normals.append([float(parts[2]), float(parts[3]), float(parts[4])])

    vertices_arr = (
        np.array(vertices, dtype=np.float32)
        if vertices
        else np.zeros((0, 3), dtype=np.float32)
    )
    normals_arr = (
        np.array(normals, dtype=np.float32)
        if normals
        else np.zeros((0, 3), dtype=np.float32)
    )

    face_count = len(normals_arr)
    bounding_box = _compute_bounding_box(vertices_arr)

    return MeshData(
        vertices=vertices_arr,
        normals=normals_arr,
        face_count=face_count,
        bounding_box=bounding_box,
        format="ascii",
    )


def _parse_binary(data: bytes) -> MeshData:
    if len(data) < 84:
        raise ValueError("Invalid binary STL: file too short")

    face_count = struct.unpack("<I", data[80:84])[0]

    expected_size = 84 + face_count * 50
    if len(data) < expected_size:
        raise ValueError(
            f"Invalid binary STL: expected {expected_size} bytes, got {len(data)}"
        )

    vertices = np.zeros((face_count * 3, 3), dtype=np.float32)
    normals = np.zeros((face_count, 3), dtype=np.float32)

    for i in range(face_count):
        offset = 84 + i * 50
        nx, ny, nz = struct.unpack("<fff", data[offset : offset + 12])
        normals[i] = [nx, ny, nz]

        for j in range(3):
            v_offset = offset + 12 + j * 12
            vx, vy, vz = struct.unpack("<fff", data[v_offset : v_offset + 12])
            vertices[i * 3 + j] = [vx, vy, vz]

    bounding_box = _compute_bounding_box(vertices)

    return MeshData(
        vertices=vertices,
        normals=normals,
        face_count=face_count,
        bounding_box=bounding_box,
        format="binary",
    )


def _compute_bounding_box(
    vertices: npt.NDArray[np.float32],
) -> dict[str, tuple[float, float]]:
    if vertices.size == 0:
        return {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}

    return {
        "x": (float(vertices[:, 0].min()), float(vertices[:, 0].max())),
        "y": (float(vertices[:, 1].min()), float(vertices[:, 1].max())),
        "z": (float(vertices[:, 2].min()), float(vertices[:, 2].max())),
    }


def read_stl_file(path: str) -> MeshData:
    try:
        with open(path, encoding="ascii") as f:
            first_chars = f.read(80)
            f.seek(0)
            if first_chars.strip().startswith("solid"):
                return _parse_ascii(f)
    except UnicodeDecodeError:
        pass

    with open(path, "rb") as f:
        data = f.read()

    return _parse_binary(data)


def get_mesh_info(path: str) -> dict[str, object]:
    try:
        with open(path, encoding="ascii") as f:
            first_chars = f.read(80)
            f.seek(0)
            if first_chars.strip().startswith("solid"):
                data = read_stl_file(path)
                center = _compute_center(data.vertices)
                return {
                    "face_count": data.face_count,
                    "bounding_box": data.bounding_box,
                    "center": center,
                    "format": "ascii",
                }
    except UnicodeDecodeError:
        pass

    with open(path, "rb") as f:
        f.seek(80)
        face_count_bytes = f.read(4)
        if len(face_count_bytes) < 4:
            raise ValueError("Invalid binary STL: cannot read face count")
        face_count = struct.unpack("<I", face_count_bytes)[0]

    data = read_stl_file(path)
    center = _compute_center(data.vertices)

    return {
        "face_count": face_count,
        "bounding_box": data.bounding_box,
        "center": center,
        "format": "binary",
    }


def _compute_center(vertices: npt.NDArray[np.float32]) -> dict[str, float]:
    if vertices.size == 0:
        return {"x": 0.0, "y": 0.0, "z": 0.0}

    return {
        "x": float(vertices[:, 0].mean()),
        "y": float(vertices[:, 1].mean()),
        "z": float(vertices[:, 2].mean()),
    }


def translate_stl(
    path: str,
    output_path: str,
    x: float,
    y: float,
    z: float,
) -> str:
    mesh = read_stl_file(path)

    translation = np.array([x, y, z], dtype=np.float32)
    mesh.vertices += translation

    write_stl_mesh(mesh, output_path, "binary")
    return output_path


def rotate_stl(
    path: str,
    output_path: str,
    axis: str,
    angle: float,
) -> str:
    if axis.lower() not in ("x", "y", "z"):
        raise ValueError(f"Invalid axis: {axis}. Must be 'x', 'y', or 'z'")

    mesh = read_stl_file(path)

    angle_rad = np.radians(angle)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)

    axis_lower = axis.lower()
    if axis_lower == "x":
        rotation_matrix = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
    elif axis_lower == "y":
        rotation_matrix = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    else:
        rotation_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

    center = mesh.vertices.mean(axis=0)
    centered = mesh.vertices - center
    rotated = centered @ rotation_matrix.T
    mesh.vertices = rotated + center

    center_normals = mesh.normals.mean(axis=0)
    if np.linalg.norm(center_normals) > 0:
        centered_normals = mesh.normals - center_normals
        mesh.normals = centered_normals @ rotation_matrix.T + center_normals

    write_stl_mesh(mesh, output_path, "binary")
    return output_path


def scale_stl(
    path: str,
    output_path: str,
    x: float,
    y: float,
    z: float,
) -> str:
    mesh = read_stl_file(path)

    scale = np.array([x, y, z], dtype=np.float32)
    mesh.vertices *= scale

    write_stl_mesh(mesh, output_path, "binary")
    return output_path


def write_stl(
    vertices: list[list[float]],
    normals: list[list[float]],
    output_path: str,
    format: str = "binary",
) -> str:
    vertices_arr = np.array(vertices, dtype=np.float32)
    normals_arr = np.array(normals, dtype=np.float32)

    mesh = MeshData(
        vertices=vertices_arr,
        normals=normals_arr,
        face_count=len(normals_arr),
        bounding_box=_compute_bounding_box(vertices_arr),
        format=format,
    )

    write_stl_mesh(mesh, output_path, format)
    return output_path


def write_stl_mesh(mesh: MeshData, output_path: str, format: str) -> None:
    if format == "ascii":
        _write_ascii(mesh, output_path)
    else:
        _write_binary(mesh, output_path)


def _write_ascii(mesh: MeshData, output_path: str) -> None:
    with open(output_path, "w", encoding="ascii") as f:
        f.write("solid model\n")
        for i in range(mesh.face_count):
            nx, ny, nz = mesh.normals[i]
            f.write(f"facet normal {nx:.6f} {ny:.6f} {nz:.6f}\n")
            f.write("  outer loop\n")
            for j in range(3):
                vx, vy, vz = mesh.vertices[i * 3 + j]
                f.write(f"    vertex {vx:.6f} {vy:.6f} {vz:.6f}\n")
            f.write("  endloop\n")
            f.write("endfacet\n")
        f.write("endsolid model\n")


def _write_binary(mesh: MeshData, output_path: str) -> None:
    header = b"\0" * 80

    with open(output_path, "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", mesh.face_count))

        for i in range(mesh.face_count):
            nx, ny, nz = mesh.normals[i]
            f.write(struct.pack("<fff", nx, ny, nz))

            for j in range(3):
                vx, vy, vz = mesh.vertices[i * 3 + j]
                f.write(struct.pack("<fff", vx, vy, vz))

            f.write(struct.pack("<H", 0))


def create_cube(
    output_path: str, size: float = 1.0, center: list[float] | None = None
) -> str:
    if center is None:
        center = [0.0, 0.0, 0.0]

    s = size / 2
    cx, cy, cz = center

    vertices = [
        [cx - s, cy - s, cz + s],
        [cx + s, cy - s, cz + s],
        [cx + s, cy + s, cz + s],
        [cx - s, cy + s, cz + s],
        [cx - s, cy - s, cz - s],
        [cx + s, cy - s, cz - s],
        [cx + s, cy + s, cz - s],
        [cx - s, cy + s, cz - s],
    ]

    faces = [
        [0, 1, 2, 3],
        [4, 7, 6, 5],
        [0, 4, 5, 1],
        [1, 5, 6, 2],
        [2, 6, 7, 3],
        [4, 0, 3, 7],
    ]

    normals = [
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ]

    verts_list: list[list[float]] = []
    normals_list: list[list[float]] = []

    for i, face in enumerate(faces):
        normal = normals[i]
        for idx in face:
            verts_list.append(vertices[idx])
            normals_list.append(normal)

    mesh = _build_mesh_from_quads(verts_list, normals_list)
    write_stl_mesh(mesh, output_path, "binary")
    return output_path


def create_sphere(output_path: str, radius: float = 1.0, segments: int = 32) -> str:
    verts_list: list[list[float]] = []
    normals_list: list[list[float]] = []

    for i in range(segments):
        phi1 = (i / segments) * np.pi
        phi2 = ((i + 1) / segments) * np.pi

        for j in range(segments):
            theta1 = (j / segments) * 2 * np.pi
            theta2 = ((j + 1) / segments) * 2 * np.pi

            p1 = _spherical_to_cartesian(radius, phi1, theta1)
            p2 = _spherical_to_cartesian(radius, phi1, theta2)
            p3 = _spherical_to_cartesian(radius, phi2, theta2)
            p4 = _spherical_to_cartesian(radius, phi2, theta1)

            n1 = _normalize(p1)
            n2 = _normalize(p2)
            n3 = _normalize(p3)
            n4 = _normalize(p4)

            verts_list.extend([p1, p2, p3, p1, p3, p4])
            normals_list.extend([n1, n2, n3, n1, n3, n4])

    mesh = _build_mesh(verts_list, normals_list)
    write_stl_mesh(mesh, output_path, "binary")
    return output_path


def _spherical_to_cartesian(r: float, phi: float, theta: float) -> list[float]:
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return [float(x), float(y), float(z)]


def _normalize(v: list[float]) -> list[float]:
    arr = np.array(v)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return [0.0, 0.0, 1.0]
    normalized: list[float] = (arr / norm).tolist()
    return normalized


def create_cylinder(
    output_path: str,
    radius: float = 1.0,
    height: float = 2.0,
    segments: int = 32,
) -> str:
    verts_list: list[list[float]] = []
    normals_list: list[list[float]] = []

    half_h = height / 2

    for i in range(segments):
        theta1 = (i / segments) * 2 * np.pi
        theta2 = ((i + 1) / segments) * 2 * np.pi

        x1 = radius * np.cos(theta1)
        z1 = radius * np.sin(theta1)
        x2 = radius * np.cos(theta2)
        z2 = radius * np.sin(theta2)

        n1 = _normalize([x1, 0.0, z1])
        n2 = _normalize([x2, 0.0, z2])

        verts_list.extend(
            [
                [x1, -half_h, z1],
                [x2, -half_h, z2],
                [x1, half_h, z1],
                [x2, -half_h, z2],
                [x2, half_h, z2],
                [x1, half_h, z1],
            ]
        )
        normals_list.extend([n1, n2, n1, n2, n2, n1])

        verts_list.extend(
            [
                [x1, -half_h, z1],
                [x2, -half_h, z2],
                [0.0, -half_h, 0.0],
            ]
        )
        normals_list.extend(
            [
                [0.0, -1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        )

        verts_list.extend(
            [
                [x1, half_h, z1],
                [x2, half_h, z2],
                [0.0, half_h, 0.0],
            ]
        )
        normals_list.extend(
            [
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )

    mesh = _build_mesh(verts_list, normals_list)
    write_stl_mesh(mesh, output_path, "binary")
    return output_path


def create_cone(
    output_path: str,
    radius: float = 1.0,
    height: float = 2.0,
    segments: int = 32,
) -> str:
    verts_list: list[list[float]] = []
    normals_list: list[list[float]] = []

    half_h = height / 2

    for i in range(segments):
        theta1 = (i / segments) * 2 * np.pi
        theta2 = ((i + 1) / segments) * 2 * np.pi

        x1 = radius * np.cos(theta1)
        z1 = radius * np.sin(theta1)
        x2 = radius * np.cos(theta2)
        z2 = radius * np.sin(theta2)

        slope = radius / height
        ny = slope / np.sqrt(1 + slope * slope)
        nxz = 1.0 / np.sqrt(1 + slope * slope)

        n1 = [nxz * np.cos(theta1), ny, nxz * np.sin(theta1)]
        n2 = [nxz * np.cos(theta2), ny, nxz * np.sin(theta2)]

        verts_list.extend(
            [
                [x1, -half_h, z1],
                [x2, -half_h, z2],
                [0.0, half_h, 0.0],
            ]
        )
        normals_list.extend([n1, n2, [0.0, ny, 0.0]])

        verts_list.extend(
            [
                [x1, -half_h, z1],
                [x2, -half_h, z2],
                [0.0, -half_h, 0.0],
            ]
        )
        normals_list.extend(
            [
                [0.0, -1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        )

    mesh = _build_mesh(verts_list, normals_list)
    write_stl_mesh(mesh, output_path, "binary")
    return output_path


def create_torus(
    output_path: str,
    major_radius: float = 1.0,
    minor_radius: float = 0.3,
    major_segments: int = 32,
    minor_segments: int = 16,
) -> str:
    verts_list: list[list[float]] = []
    normals_list: list[list[float]] = []

    for i in range(major_segments):
        u1 = (i / major_segments) * 2 * np.pi
        u2 = ((i + 1) / major_segments) * 2 * np.pi

        for j in range(minor_segments):
            v1 = (j / minor_segments) * 2 * np.pi
            v2 = ((j + 1) / minor_segments) * 2 * np.pi

            p1 = _torus_point(major_radius, minor_radius, u1, v1)
            p2 = _torus_point(major_radius, minor_radius, u1, v2)
            p3 = _torus_point(major_radius, minor_radius, u2, v2)
            p4 = _torus_point(major_radius, minor_radius, u2, v1)

            n1 = _normalize(
                [
                    p1[0] - major_radius * np.cos(u1),
                    0.0,
                    p1[2] - major_radius * np.sin(u1),
                ]
            )
            n2 = _normalize(
                [
                    p2[0] - major_radius * np.cos(u1),
                    0.0,
                    p2[2] - major_radius * np.sin(u1),
                ]
            )
            n3 = _normalize(
                [
                    p3[0] - major_radius * np.cos(u2),
                    0.0,
                    p3[2] - major_radius * np.sin(u2),
                ]
            )
            n4 = _normalize(
                [
                    p4[0] - major_radius * np.cos(u2),
                    0.0,
                    p4[2] - major_radius * np.sin(u2),
                ]
            )

            verts_list.extend([p1, p2, p3, p1, p3, p4])
            normals_list.extend([n1, n2, n3, n1, n3, n4])

    mesh = _build_mesh(verts_list, normals_list)
    write_stl_mesh(mesh, output_path, "binary")
    return output_path


def _torus_point(r_major: float, r_minor: float, u: float, v: float) -> list[float]:
    x = (r_major + r_minor * np.cos(v)) * np.cos(u)
    y = r_minor * np.sin(v)
    z = (r_major + r_minor * np.cos(v)) * np.sin(u)
    return [float(x), float(y), float(z)]


def create_plane(output_path: str, width: float = 1.0, height: float = 1.0) -> str:
    w = width / 2
    h = height / 2

    vertices = [
        [-w, 0.0, -h],
        [w, 0.0, -h],
        [w, 0.0, h],
        [-w, 0.0, h],
    ]

    normals = [[0.0, 1.0, 0.0]]

    verts_list = [
        vertices[0],
        vertices[1],
        vertices[2],
        vertices[0],
        vertices[2],
        vertices[3],
    ]
    normals_list = [normals[0]] * 6

    mesh = _build_mesh(verts_list, normals_list)
    write_stl_mesh(mesh, output_path, "binary")
    return output_path


def _build_mesh(vertices: list[list[float]], normals: list[list[float]]) -> MeshData:
    vertices_arr = np.array(vertices, dtype=np.float32)
    normals_arr = np.array(normals, dtype=np.float32)

    face_count = len(vertices_arr) // 3
    bounding_box = _compute_bounding_box(vertices_arr)

    return MeshData(
        vertices=vertices_arr,
        normals=normals_arr,
        face_count=face_count,
        bounding_box=bounding_box,
        format="binary",
    )


def create_box(
    output_path: str,
    width: float = 1.0,
    height: float = 1.0,
    depth: float = 1.0,
    center: list[float] | None = None,
) -> str:
    if center is None:
        center = [0.0, 0.0, 0.0]

    hw = width / 2
    hh = height / 2
    hd = depth / 2
    cx, cy, cz = center

    vertices = [
        [cx - hw, cy - hh, cz + hd],
        [cx + hw, cy - hh, cz + hd],
        [cx + hw, cy + hh, cz + hd],
        [cx - hw, cy + hh, cz + hd],
        [cx - hw, cy - hh, cz - hd],
        [cx + hw, cy - hh, cz - hd],
        [cx + hw, cy + hh, cz - hd],
        [cx - hw, cy + hh, cz - hd],
    ]

    faces = [
        [0, 1, 2, 3],
        [4, 7, 6, 5],
        [0, 4, 5, 1],
        [1, 5, 6, 2],
        [2, 6, 7, 3],
        [4, 0, 3, 7],
    ]

    normals_list_face = [
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ]

    verts_list: list[list[float]] = []
    normals_flat: list[list[float]] = []

    for i, face in enumerate(faces):
        normal = normals_list_face[i]
        for idx in face:
            verts_list.append(vertices[idx])
            normals_flat.append(normal)

    mesh = _build_mesh_from_quads(verts_list, normals_flat)
    write_stl_mesh(mesh, output_path, "binary")
    return output_path


def create_capsule(
    output_path: str,
    radius: float = 0.5,
    height: float = 2.0,
    segments: int = 32,
) -> str:
    verts_list: list[list[float]] = []
    normals_list: list[list[float]] = []

    half_h = height / 2
    hemi_segs = max(segments // 2, 2)

    # Bottom hemisphere (opens upward towards cylinder)
    for i in range(hemi_segs):
        phi1 = (np.pi / 2) + (i / hemi_segs) * (np.pi / 2)
        phi2 = (np.pi / 2) + ((i + 1) / hemi_segs) * (np.pi / 2)
        for j in range(segments):
            theta1 = (j / segments) * 2 * np.pi
            theta2 = ((j + 1) / segments) * 2 * np.pi

            p1 = _spherical_to_cartesian(radius, phi1, theta1)
            p2 = _spherical_to_cartesian(radius, phi1, theta2)
            p3 = _spherical_to_cartesian(radius, phi2, theta2)
            p4 = _spherical_to_cartesian(radius, phi2, theta1)

            # Normals are the outward unit vectors from the hemisphere centre,
            # which equal the unit-sphere positions BEFORE the vertical offset.
            n1 = _normalize(p1)
            n2 = _normalize(p2)
            n3 = _normalize(p3)
            n4 = _normalize(p4)

            p1[1] -= half_h
            p2[1] -= half_h
            p3[1] -= half_h
            p4[1] -= half_h

            verts_list.extend([p1, p2, p3, p1, p3, p4])
            normals_list.extend([n1, n2, n3, n1, n3, n4])

    # Top hemisphere
    for i in range(hemi_segs):
        phi1 = (i / hemi_segs) * (np.pi / 2)
        phi2 = ((i + 1) / hemi_segs) * (np.pi / 2)
        for j in range(segments):
            theta1 = (j / segments) * 2 * np.pi
            theta2 = ((j + 1) / segments) * 2 * np.pi

            p1 = _spherical_to_cartesian(radius, phi1, theta1)
            p2 = _spherical_to_cartesian(radius, phi1, theta2)
            p3 = _spherical_to_cartesian(radius, phi2, theta2)
            p4 = _spherical_to_cartesian(radius, phi2, theta1)

            # Normals are the outward unit vectors from the hemisphere centre,
            # which equal the unit-sphere positions BEFORE the vertical offset.
            n1 = _normalize(p1)
            n2 = _normalize(p2)
            n3 = _normalize(p3)
            n4 = _normalize(p4)

            p1[1] += half_h
            p2[1] += half_h
            p3[1] += half_h
            p4[1] += half_h

            verts_list.extend([p1, p2, p3, p1, p3, p4])
            normals_list.extend([n1, n2, n3, n1, n3, n4])

    # Cylindrical body
    for i in range(segments):
        theta1 = (i / segments) * 2 * np.pi
        theta2 = ((i + 1) / segments) * 2 * np.pi

        x1 = radius * np.cos(theta1)
        z1 = radius * np.sin(theta1)
        x2 = radius * np.cos(theta2)
        z2 = radius * np.sin(theta2)

        n1 = _normalize([x1, 0.0, z1])
        n2 = _normalize([x2, 0.0, z2])

        verts_list.extend(
            [
                [x1, -half_h, z1],
                [x2, -half_h, z2],
                [x1, half_h, z1],
                [x2, -half_h, z2],
                [x2, half_h, z2],
                [x1, half_h, z1],
            ]
        )
        normals_list.extend([n1, n2, n1, n2, n2, n1])

    mesh = _build_mesh(verts_list, normals_list)
    write_stl_mesh(mesh, output_path, "binary")
    return output_path


def create_ellipsoid(
    output_path: str,
    rx: float = 1.0,
    ry: float = 0.5,
    rz: float = 0.75,
    segments: int = 32,
) -> str:
    verts_list: list[list[float]] = []
    normals_list: list[list[float]] = []

    for i in range(segments):
        phi1 = (i / segments) * np.pi
        phi2 = ((i + 1) / segments) * np.pi

        for j in range(segments):
            theta1 = (j / segments) * 2 * np.pi
            theta2 = ((j + 1) / segments) * 2 * np.pi

            def _pt(phi: float, theta: float) -> list[float]:
                return [
                    float(rx * np.sin(phi) * np.cos(theta)),
                    float(ry * np.cos(phi)),
                    float(rz * np.sin(phi) * np.sin(theta)),
                ]

            p1 = _pt(phi1, theta1)
            p2 = _pt(phi1, theta2)
            p3 = _pt(phi2, theta2)
            p4 = _pt(phi2, theta1)

            # Normals: scale by inverse of squared radii (ellipsoid gradient)
            def _en(p: list[float]) -> list[float]:
                return _normalize(
                    [p[0] / (rx * rx), p[1] / (ry * ry), p[2] / (rz * rz)]
                )

            verts_list.extend([p1, p2, p3, p1, p3, p4])
            normals_list.extend([_en(p1), _en(p2), _en(p3), _en(p1), _en(p3), _en(p4)])

    mesh = _build_mesh(verts_list, normals_list)
    write_stl_mesh(mesh, output_path, "binary")
    return output_path


def create_frustum(
    output_path: str,
    bottom_radius: float = 1.0,
    top_radius: float = 0.5,
    height: float = 2.0,
    segments: int = 32,
) -> str:
    verts_list: list[list[float]] = []
    normals_list: list[list[float]] = []

    half_h = height / 2
    dr = bottom_radius - top_radius
    slant = np.sqrt(dr * dr + height * height)
    ny_norm = dr / slant
    nxz_norm = height / slant

    for i in range(segments):
        theta1 = (i / segments) * 2 * np.pi
        theta2 = ((i + 1) / segments) * 2 * np.pi

        bx1 = bottom_radius * np.cos(theta1)
        bz1 = bottom_radius * np.sin(theta1)
        bx2 = bottom_radius * np.cos(theta2)
        bz2 = bottom_radius * np.sin(theta2)

        tx1 = top_radius * np.cos(theta1)
        tz1 = top_radius * np.sin(theta1)
        tx2 = top_radius * np.cos(theta2)
        tz2 = top_radius * np.sin(theta2)

        n1 = [nxz_norm * np.cos(theta1), ny_norm, nxz_norm * np.sin(theta1)]
        n2 = [nxz_norm * np.cos(theta2), ny_norm, nxz_norm * np.sin(theta2)]

        # Side quad as two triangles
        verts_list.extend(
            [
                [bx1, -half_h, bz1],
                [bx2, -half_h, bz2],
                [tx1, half_h, tz1],
                [bx2, -half_h, bz2],
                [tx2, half_h, tz2],
                [tx1, half_h, tz1],
            ]
        )
        normals_list.extend([n1, n2, n1, n2, n2, n1])

        # Bottom cap
        verts_list.extend(
            [[bx1, -half_h, bz1], [bx2, -half_h, bz2], [0.0, -half_h, 0.0]]
        )
        normals_list.extend([[0.0, -1.0, 0.0]] * 3)

        # Top cap
        verts_list.extend(
            [[tx1, half_h, tz1], [tx2, half_h, tz2], [0.0, half_h, 0.0]]
        )
        normals_list.extend([[0.0, 1.0, 0.0]] * 3)

    mesh = _build_mesh(verts_list, normals_list)
    write_stl_mesh(mesh, output_path, "binary")
    return output_path


def create_tube(
    output_path: str,
    outer_radius: float = 1.0,
    inner_radius: float = 0.7,
    height: float = 2.0,
    segments: int = 32,
) -> str:
    if inner_radius >= outer_radius:
        raise ValueError("inner_radius must be less than outer_radius")

    verts_list: list[list[float]] = []
    normals_list: list[list[float]] = []

    half_h = height / 2

    for i in range(segments):
        theta1 = (i / segments) * 2 * np.pi
        theta2 = ((i + 1) / segments) * 2 * np.pi

        ox1 = outer_radius * np.cos(theta1)
        oz1 = outer_radius * np.sin(theta1)
        ox2 = outer_radius * np.cos(theta2)
        oz2 = outer_radius * np.sin(theta2)

        ix1 = inner_radius * np.cos(theta1)
        iz1 = inner_radius * np.sin(theta1)
        ix2 = inner_radius * np.cos(theta2)
        iz2 = inner_radius * np.sin(theta2)

        no1 = _normalize([ox1, 0.0, oz1])
        no2 = _normalize([ox2, 0.0, oz2])
        ni1 = _normalize([-ix1, 0.0, -iz1])
        ni2 = _normalize([-ix2, 0.0, -iz2])

        # Outer wall
        verts_list.extend(
            [
                [ox1, -half_h, oz1],
                [ox2, -half_h, oz2],
                [ox1, half_h, oz1],
                [ox2, -half_h, oz2],
                [ox2, half_h, oz2],
                [ox1, half_h, oz1],
            ]
        )
        normals_list.extend([no1, no2, no1, no2, no2, no1])

        # Inner wall (reversed winding for inward-facing normals)
        verts_list.extend(
            [
                [ix1, half_h, iz1],
                [ix2, half_h, iz2],
                [ix1, -half_h, iz1],
                [ix2, half_h, iz2],
                [ix2, -half_h, iz2],
                [ix1, -half_h, iz1],
            ]
        )
        normals_list.extend([ni1, ni2, ni1, ni2, ni2, ni1])

        # Bottom annular ring
        verts_list.extend(
            [
                [ox1, -half_h, oz1],
                [ox2, -half_h, oz2],
                [ix2, -half_h, iz2],
                [ox1, -half_h, oz1],
                [ix2, -half_h, iz2],
                [ix1, -half_h, iz1],
            ]
        )
        normals_list.extend([[0.0, -1.0, 0.0]] * 6)

        # Top annular ring
        verts_list.extend(
            [
                [ox1, half_h, oz1],
                [ix2, half_h, iz2],
                [ox2, half_h, oz2],
                [ox1, half_h, oz1],
                [ix1, half_h, iz1],
                [ix2, half_h, iz2],
            ]
        )
        normals_list.extend([[0.0, 1.0, 0.0]] * 6)

    mesh = _build_mesh(verts_list, normals_list)
    write_stl_mesh(mesh, output_path, "binary")
    return output_path


def mirror_stl(path: str, output_path: str, axis: str) -> str:
    if axis.lower() not in ("x", "y", "z"):
        raise ValueError(f"Invalid axis: {axis}. Must be 'x', 'y', or 'z'")

    mesh = read_stl_file(path)

    axis_idx = {"x": 0, "y": 1, "z": 2}[axis.lower()]

    mirrored_vertices = mesh.vertices.copy()
    mirrored_vertices[:, axis_idx] *= -1

    mirrored_normals = mesh.normals.copy()
    mirrored_normals[:, axis_idx] *= -1

    new_mesh = MeshData(
        vertices=mirrored_vertices,
        normals=mirrored_normals,
        face_count=mesh.face_count,
        bounding_box=_compute_bounding_box(mirrored_vertices),
        format="binary",
    )
    write_stl_mesh(new_mesh, output_path, "binary")
    return output_path


def rotate_stl_axis(
    path: str,
    output_path: str,
    ax: float,
    ay: float,
    az: float,
    angle: float,
) -> str:
    axis_vec = np.array([ax, ay, az], dtype=np.float64)
    norm = np.linalg.norm(axis_vec)
    if norm == 0:
        raise ValueError("Rotation axis vector must not be the zero vector")
    axis_vec = axis_vec / norm

    mesh = read_stl_file(path)

    angle_rad = np.radians(angle)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    t = 1.0 - c
    ux, uy, uz = axis_vec

    rotation_matrix = np.array(
        [
            [t * ux * ux + c, t * ux * uy - s * uz, t * ux * uz + s * uy],
            [t * ux * uy + s * uz, t * uy * uy + c, t * uy * uz - s * ux],
            [t * ux * uz - s * uy, t * uy * uz + s * ux, t * uz * uz + c],
        ],
        dtype=np.float32,
    )

    center = mesh.vertices.mean(axis=0)
    centered = mesh.vertices - center
    mesh.vertices = (centered @ rotation_matrix.T) + center

    mesh.normals = mesh.normals @ rotation_matrix.T

    write_stl_mesh(mesh, output_path, "binary")
    return output_path


def combine_stl(paths: list[str], output_path: str) -> str:
    if not paths:
        raise ValueError("At least one input STL path must be provided")

    all_vertices: list[npt.NDArray[np.float32]] = []
    all_normals: list[npt.NDArray[np.float32]] = []
    total_faces = 0

    for p in paths:
        mesh = read_stl_file(p)
        all_vertices.append(mesh.vertices)
        all_normals.append(mesh.normals)
        total_faces += mesh.face_count

    combined_vertices = np.concatenate(all_vertices, axis=0)
    combined_normals = np.concatenate(all_normals, axis=0)

    combined_mesh = MeshData(
        vertices=combined_vertices,
        normals=combined_normals,
        face_count=total_faces,
        bounding_box=_compute_bounding_box(combined_vertices),
        format="binary",
    )

    write_stl_mesh(combined_mesh, output_path, "binary")
    return output_path


# ─────────────────────────── shared geometry helpers ────────────────────────


def _extrude_profile(
    profile: list[tuple[float, float]],
    thickness: float,
) -> tuple[list[list[float]], list[list[float]]]:
    """Extrude a closed 2D polygon in the X-Z plane along the Y axis.

    Args:
        profile: List of (x, z) pairs defining the cross-section in CCW order
                 when viewed from +Y.
        thickness: Extrusion thickness along the Y axis.

    Returns:
        Tuple of (verts_list, normals_list) ready for _build_mesh.
    """
    verts_list: list[list[float]] = []
    normals_list: list[list[float]] = []
    n = len(profile)
    half_h = thickness / 2.0
    cx = float(sum(p[0] for p in profile)) / n
    cz = float(sum(p[1] for p in profile)) / n

    # Top face (y = +half_h, normal [0,1,0])
    for i in range(n):
        x1, z1 = profile[i]
        x2, z2 = profile[(i + 1) % n]
        verts_list.extend([[cx, half_h, cz], [x1, half_h, z1], [x2, half_h, z2]])
        normals_list.extend([[0.0, 1.0, 0.0]] * 3)

    # Bottom face (y = -half_h, normal [0,-1,0]) — reversed winding
    for i in range(n):
        x1, z1 = profile[i]
        x2, z2 = profile[(i + 1) % n]
        verts_list.extend([[cx, -half_h, cz], [x2, -half_h, z2], [x1, -half_h, z1]])
        normals_list.extend([[0.0, -1.0, 0.0]] * 3)

    # Side faces
    for i in range(n):
        x1, z1 = profile[i]
        x2, z2 = profile[(i + 1) % n]
        dx = float(x2 - x1)
        dz = float(z2 - z1)
        # Outward normal: 90° CW rotation of travel direction in XZ
        nx, nz = dz, -dx
        length = float(np.sqrt(nx * nx + nz * nz))
        if length > 1e-10:
            nx /= length
            nz /= length
        n_side: list[float] = [nx, 0.0, nz]
        verts_list.extend(
            [
                [x1, -half_h, z1], [x2, -half_h, z2], [x1, half_h, z1],
                [x2, -half_h, z2], [x2, half_h, z2], [x1, half_h, z1],
            ]
        )
        normals_list.extend([n_side] * 6)

    return verts_list, normals_list


def _add_cylinder_verts(
    verts: list[list[float]],
    norms: list[list[float]],
    cx: float,
    cy_bot: float,
    cy_top: float,
    cz: float,
    radius: float,
    segments: int,
    cap_bot: bool = True,
    cap_top: bool = True,
) -> None:
    """Append solid-cylinder (Y-axis) triangles into existing vertex/normal lists."""
    for i in range(segments):
        theta1 = (i / segments) * 2.0 * np.pi
        theta2 = ((i + 1) / segments) * 2.0 * np.pi
        x1 = cx + radius * float(np.cos(theta1))
        z1 = cz + radius * float(np.sin(theta1))
        x2 = cx + radius * float(np.cos(theta2))
        z2 = cz + radius * float(np.sin(theta2))
        n1 = _normalize([x1 - cx, 0.0, z1 - cz])
        n2 = _normalize([x2 - cx, 0.0, z2 - cz])
        verts.extend(
            [
                [x1, cy_bot, z1], [x2, cy_bot, z2], [x1, cy_top, z1],
                [x2, cy_bot, z2], [x2, cy_top, z2], [x1, cy_top, z1],
            ]
        )
        norms.extend([n1, n2, n1, n2, n2, n1])
        if cap_bot:
            verts.extend([[x1, cy_bot, z1], [x2, cy_bot, z2], [cx, cy_bot, cz]])
            norms.extend([[0.0, -1.0, 0.0]] * 3)
        if cap_top:
            verts.extend([[x2, cy_top, z2], [x1, cy_top, z1], [cx, cy_top, cz]])
            norms.extend([[0.0, 1.0, 0.0]] * 3)


def _add_tube_verts(
    verts: list[list[float]],
    norms: list[list[float]],
    cy_bot: float,
    cy_top: float,
    outer_r: float,
    inner_r: float,
    segments: int,
) -> None:
    """Append hollow-cylinder (tube, Y-axis) triangles into existing lists."""
    for i in range(segments):
        theta1 = (i / segments) * 2.0 * np.pi
        theta2 = ((i + 1) / segments) * 2.0 * np.pi
        ox1, oz1 = outer_r * float(np.cos(theta1)), outer_r * float(np.sin(theta1))
        ox2, oz2 = outer_r * float(np.cos(theta2)), outer_r * float(np.sin(theta2))
        ix1, iz1 = inner_r * float(np.cos(theta1)), inner_r * float(np.sin(theta1))
        ix2, iz2 = inner_r * float(np.cos(theta2)), inner_r * float(np.sin(theta2))
        no1 = _normalize([ox1, 0.0, oz1])
        no2 = _normalize([ox2, 0.0, oz2])
        ni1 = _normalize([-ix1, 0.0, -iz1])
        ni2 = _normalize([-ix2, 0.0, -iz2])
        # Outer wall
        verts.extend(
            [
                [ox1, cy_bot, oz1], [ox2, cy_bot, oz2], [ox1, cy_top, oz1],
                [ox2, cy_bot, oz2], [ox2, cy_top, oz2], [ox1, cy_top, oz1],
            ]
        )
        norms.extend([no1, no2, no1, no2, no2, no1])
        # Inner wall (reversed winding for inward-facing normals)
        verts.extend(
            [
                [ix1, cy_top, iz1], [ix2, cy_top, iz2], [ix1, cy_bot, iz1],
                [ix2, cy_top, iz2], [ix2, cy_bot, iz2], [ix1, cy_bot, iz1],
            ]
        )
        norms.extend([ni1, ni2, ni1, ni2, ni2, ni1])
        # Bottom annular ring
        verts.extend(
            [
                [ox1, cy_bot, oz1], [ox2, cy_bot, oz2], [ix2, cy_bot, iz2],
                [ox1, cy_bot, oz1], [ix2, cy_bot, iz2], [ix1, cy_bot, iz1],
            ]
        )
        norms.extend([[0.0, -1.0, 0.0]] * 6)
        # Top annular ring
        verts.extend(
            [
                [ox1, cy_top, oz1], [ix2, cy_top, iz2], [ox2, cy_top, oz2],
                [ox1, cy_top, oz1], [ix1, cy_top, iz1], [ix2, cy_top, iz2],
            ]
        )
        norms.extend([[0.0, 1.0, 0.0]] * 6)


# ─────────────────────────── airplane / helicopter shapes ────────────────────


def _naca_profile(
    chord: float,
    thickness_ratio: float,
    n_pts: int,
) -> list[tuple[float, float]]:
    """Returns (x, y) points for a NACA 4-digit symmetric airfoil.

    The profile traces a closed loop: top surface from the leading edge
    (x=0) to the trailing edge (x=chord), then the bottom surface back to
    the leading edge.  Cosine spacing is used for better resolution near
    the leading and trailing edges.

    Args:
        chord: Chord length.
        thickness_ratio: Maximum thickness as a fraction of chord (e.g. 0.12
            for a NACA 0012 profile).
        n_pts: Number of sample points per surface half (top + bottom).

    Returns:
        List of (x, y) tuples forming a closed, ordered polygon.
    """
    t = thickness_ratio

    def _yt(xn: float) -> float:
        return (
            5.0
            * t
            * chord
            * (
                0.2969 * float(np.sqrt(max(xn, 0.0)))
                - 0.1260 * xn
                - 0.3516 * xn**2
                + 0.2843 * xn**3
                - 0.1015 * xn**4
            )
        )

    xs = [(1.0 - np.cos(np.pi * i / (n_pts - 1))) / 2.0 for i in range(n_pts)]
    top: list[tuple[float, float]] = [
        (float(x * chord), float(_yt(x))) for x in xs
    ]
    bottom: list[tuple[float, float]] = [
        (float(x * chord), float(-_yt(x))) for x in reversed(xs[1:-1])
    ]
    return top + bottom


def create_airfoil(
    output_path: str,
    chord: float = 1.0,
    span: float = 5.0,
    thickness_ratio: float = 0.12,
    segments: int = 32,
) -> str:
    """Creates a symmetric NACA airfoil wing-section mesh.

    The chord runs along the X axis (leading edge at x=0, trailing edge at
    x=chord), thickness is along Y, and the span extrudes along Z from z=0
    to z=span.  A ``thickness_ratio`` of 0.12 gives a NACA 0012 profile.

    Suitable for wings, horizontal stabilisers, and vertical fins of
    airplanes or helicopters.

    Args:
        output_path: Output STL file path.
        chord: Chord length along X (default 1.0).
        span: Wing-section span along Z (default 5.0).
        thickness_ratio: Max thickness as a fraction of chord (default 0.12).
        segments: Points per airfoil surface half; higher values give a
            smoother leading edge (default 32).

    Returns:
        Path to the output file.

    Example:
        >>> create_airfoil("wing.stl", chord=1.5, span=6.0)
        "wing.stl"
    """
    profile = _naca_profile(chord, thickness_ratio, segments)
    n = len(profile)
    cx = sum(p[0] for p in profile) / n
    cy = sum(p[1] for p in profile) / n

    verts_list: list[list[float]] = []
    normals_list: list[list[float]] = []

    # Side walls – extrude profile along Z
    for i in range(n):
        x1, y1 = profile[i]
        x2, y2 = profile[(i + 1) % n]
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.sqrt(dx * dx + dy * dy))
        if length > 1e-10:
            nx, ny = float(dy / length), float(-dx / length)
        else:
            nx, ny = 0.0, 1.0
        n_side: list[float] = [nx, ny, 0.0]
        verts_list.extend(
            [
                [x1, y1, 0.0], [x2, y2, 0.0], [x1, y1, span],
                [x2, y2, 0.0], [x2, y2, span], [x1, y1, span],
            ]
        )
        normals_list.extend([n_side] * 6)

    # Root cap (z = 0, normal −Z)
    for i in range(n):
        x1, y1 = profile[i]
        x2, y2 = profile[(i + 1) % n]
        verts_list.extend([[x1, y1, 0.0], [cx, cy, 0.0], [x2, y2, 0.0]])
        normals_list.extend([[0.0, 0.0, -1.0]] * 3)

    # Tip cap (z = span, normal +Z)
    for i in range(n):
        x1, y1 = profile[i]
        x2, y2 = profile[(i + 1) % n]
        verts_list.extend([[x1, y1, span], [x2, y2, span], [cx, cy, span]])
        normals_list.extend([[0.0, 0.0, 1.0]] * 3)

    mesh = _build_mesh(verts_list, normals_list)
    write_stl_mesh(mesh, output_path, "binary")
    return output_path


def _create_swept_blade_mesh(
    length: float,
    chord_root: float,
    chord_tip: float,
    twist_angle: float,
    thickness_ratio: float,
    segments: int,
    span_segments: int,
) -> tuple[list[list[float]], list[list[float]]]:
    """Build vertices/normals for a tapered, twisted airfoil blade.

    The blade spans along the Y axis (root at y=0, tip at y=length).
    At each span station the cross-section is a NACA symmetric airfoil in
    the X-Z plane, scaled by the local chord and rotated by the local twist
    about Y.

    Args:
        length: Blade span along Y.
        chord_root: Chord at the root (y=0).
        chord_tip: Chord at the tip (y=length).
        twist_angle: Total twist from root to tip in degrees.
        thickness_ratio: NACA thickness ratio.
        segments: Points per airfoil surface half per slice.
        span_segments: Number of span-wise divisions.

    Returns:
        Tuple (verts_list, normals_list) ready for _build_mesh.
    """
    def _ring(t: float) -> list[tuple[float, float]]:
        chord = chord_root + (chord_tip - chord_root) * t
        twist_rad = float(np.radians(twist_angle * t))
        profile_2d = _naca_profile(chord, thickness_ratio, segments)
        x_offset = chord / 2.0
        c_t = float(np.cos(twist_rad))
        s_t = float(np.sin(twist_rad))
        result: list[tuple[float, float]] = []
        for px, pz in profile_2d:
            px -= x_offset
            rx = px * c_t - pz * s_t
            rz = px * s_t + pz * c_t
            result.append((float(rx), float(rz)))
        return result

    rings = [_ring(i / span_segments) for i in range(span_segments + 1)]
    ys = [(i / span_segments) * length for i in range(span_segments + 1)]
    n = len(rings[0])

    verts_list: list[list[float]] = []
    normals_list: list[list[float]] = []

    # Loft between adjacent rings
    for si in range(span_segments):
        ring0, ring1 = rings[si], rings[si + 1]
        y0, y1 = ys[si], ys[si + 1]
        for i in range(n):
            p00 = [ring0[i][0], y0, ring0[i][1]]
            p01 = [ring0[(i + 1) % n][0], y0, ring0[(i + 1) % n][1]]
            p10 = [ring1[i][0], y1, ring1[i][1]]
            p11 = [ring1[(i + 1) % n][0], y1, ring1[(i + 1) % n][1]]
            # Cross product normals for each triangle
            e1 = [p01[j] - p00[j] for j in range(3)]
            e2 = [p10[j] - p00[j] for j in range(3)]
            nf0 = _normalize([
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ])
            e3 = [p11[j] - p01[j] for j in range(3)]
            e4 = [p10[j] - p01[j] for j in range(3)]
            nf1 = _normalize([
                e3[1] * e4[2] - e3[2] * e4[1],
                e3[2] * e4[0] - e3[0] * e4[2],
                e3[0] * e4[1] - e3[1] * e4[0],
            ])
            verts_list.extend([p00, p01, p10])
            normals_list.extend([nf0, nf0, nf0])
            verts_list.extend([p01, p11, p10])
            normals_list.extend([nf1, nf1, nf1])

    # Root cap (y = 0, normal −Y)
    ring0 = rings[0]
    cx0 = sum(p[0] for p in ring0) / n
    cz0 = sum(p[1] for p in ring0) / n
    for i in range(n):
        x1, z1 = ring0[i]
        x2, z2 = ring0[(i + 1) % n]
        verts_list.extend([[x1, 0.0, z1], [cx0, 0.0, cz0], [x2, 0.0, z2]])
        normals_list.extend([[0.0, -1.0, 0.0]] * 3)

    # Tip cap (y = length, normal +Y)
    ring_tip = rings[-1]
    cx1 = sum(p[0] for p in ring_tip) / n
    cz1 = sum(p[1] for p in ring_tip) / n
    for i in range(n):
        x1, z1 = ring_tip[i]
        x2, z2 = ring_tip[(i + 1) % n]
        verts_list.extend([[x1, length, z1], [x2, length, z2], [cx1, length, cz1]])
        normals_list.extend([[0.0, 1.0, 0.0]] * 3)

    return verts_list, normals_list


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

    The blade spans along the Y axis from y=0 (root/hub face) to y=length
    (tip).  The cross-section is a NACA symmetric airfoil in the X-Z plane,
    scaled by the local chord and progressively rotated about Y by
    *twist_angle* from root to tip.

    Use ``array_circular`` to arrange multiple blades into a full propeller or
    rotor disc.

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
    verts_list, normals_list = _create_swept_blade_mesh(
        length, chord_root, chord_tip, twist_angle, thickness_ratio,
        segments, span_segments,
    )
    mesh = _build_mesh(verts_list, normals_list)
    write_stl_mesh(mesh, output_path, "binary")
    return output_path


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

    Geometry is identical to a propeller blade (NACA airfoil profile, tapered
    chord, and progressive twist) but with defaults suited to turbomachinery:
    shorter span, more aggressive twist, and a thinner profile.

    Use ``array_circular`` to arrange blades into a full compressor or fan stage.

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
    verts_list, normals_list = _create_swept_blade_mesh(
        span, chord_root, chord_tip, twist_angle, thickness_ratio,
        segments, span_segments,
    )
    mesh = _build_mesh(verts_list, normals_list)
    write_stl_mesh(mesh, output_path, "binary")
    return output_path


def create_piston(
    output_path: str,
    bore: float = 1.0,
    height: float = 1.2,
    wall_thickness: float = 0.1,
    crown_height: float = 0.3,
    segments: int = 32,
) -> str:
    """Creates a hollow piston mesh.

    The piston axis is aligned along Y.  The crown (solid top) sits at
    y = +height/2 and has thickness *crown_height*.  The cylindrical skirt
    extends from y = −height/2 up to the base of the crown; the interior is
    open at the bottom, as on a real piston.

    Combine with ``create_connecting_rod`` and ``create_crankshaft`` to model a
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
        ValueError: If wall_thickness >= bore/2, or crown_height >= height.

    Example:
        >>> create_piston("piston.stl", bore=0.8, height=1.0)
        "piston.stl"
    """
    outer_r = bore / 2.0
    inner_r = outer_r - wall_thickness
    if inner_r <= 0.0:
        raise ValueError("wall_thickness must be less than bore/2")
    if crown_height >= height:
        raise ValueError("crown_height must be less than total height")

    verts_list: list[list[float]] = []
    normals_list: list[list[float]] = []

    top = height / 2.0
    bottom = -height / 2.0
    crown_base = top - crown_height

    # Outer cylindrical surface – full height, outward normals
    _add_cylinder_verts(
        verts_list, normals_list,
        0.0, bottom, top, 0.0,
        outer_r, segments,
        cap_bot=False, cap_top=False,
    )

    # Crown top face (solid disc, +Y normal)
    for i in range(segments):
        theta1 = (i / segments) * 2.0 * np.pi
        theta2 = ((i + 1) / segments) * 2.0 * np.pi
        x2, z2 = outer_r * float(np.cos(theta2)), outer_r * float(np.sin(theta2))
        x1, z1 = outer_r * float(np.cos(theta1)), outer_r * float(np.sin(theta1))
        verts_list.extend([[x2, top, z2], [x1, top, z1], [0.0, top, 0.0]])
        normals_list.extend([[0.0, 1.0, 0.0]] * 3)

    # Inner cylindrical surface (hollow skirt, inward normals)
    for i in range(segments):
        theta1 = (i / segments) * 2.0 * np.pi
        theta2 = ((i + 1) / segments) * 2.0 * np.pi
        ix1, iz1 = inner_r * float(np.cos(theta1)), inner_r * float(np.sin(theta1))
        ix2, iz2 = inner_r * float(np.cos(theta2)), inner_r * float(np.sin(theta2))
        ni1 = _normalize([-ix1, 0.0, -iz1])
        ni2 = _normalize([-ix2, 0.0, -iz2])
        # Reversed winding for inward-facing normals
        verts_list.extend(
            [
                [ix1, crown_base, iz1], [ix2, crown_base, iz2], [ix1, bottom, iz1],
                [ix2, crown_base, iz2], [ix2, bottom, iz2], [ix1, bottom, iz1],
            ]
        )
        normals_list.extend([ni1, ni2, ni1, ni2, ni2, ni1])

    # Crown underside (annular ring at crown_base, −Y normal)
    for i in range(segments):
        theta1 = (i / segments) * 2.0 * np.pi
        theta2 = ((i + 1) / segments) * 2.0 * np.pi
        ox1, oz1 = outer_r * float(np.cos(theta1)), outer_r * float(np.sin(theta1))
        ox2, oz2 = outer_r * float(np.cos(theta2)), outer_r * float(np.sin(theta2))
        ix1, iz1 = inner_r * float(np.cos(theta1)), inner_r * float(np.sin(theta1))
        ix2, iz2 = inner_r * float(np.cos(theta2)), inner_r * float(np.sin(theta2))
        verts_list.extend(
            [
                [ox1, crown_base, oz1], [ox2, crown_base, oz2], [ix2, crown_base, iz2],
                [ox1, crown_base, oz1], [ix2, crown_base, iz2], [ix1, crown_base, iz1],
            ]
        )
        normals_list.extend([[0.0, -1.0, 0.0]] * 6)

    mesh = _build_mesh(verts_list, normals_list)
    write_stl_mesh(mesh, output_path, "binary")
    return output_path


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

    * **Hub** region (bore to hub_radius): full *disk_thickness*.
    * **Web** region (hub_radius to disk_radius): reduced *web_thickness*.

    The axis is aligned along Y.  Turbine blades are attached around the
    outer rim using ``create_turbine_blade`` followed by ``array_circular``.

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
        ValueError: If bore_radius >= hub_radius, hub_radius >= disk_radius,
            or web_thickness >= disk_thickness.

    Example:
        >>> create_turbine_disk("disk.stl", disk_radius=2.0, bore_radius=0.4)
        "disk.stl"
    """
    if bore_radius >= hub_radius:
        raise ValueError("bore_radius must be less than hub_radius")
    if hub_radius >= disk_radius:
        raise ValueError("hub_radius must be less than disk_radius")
    if web_thickness >= disk_thickness:
        raise ValueError("web_thickness must be less than disk_thickness")

    verts_list: list[list[float]] = []
    normals_list: list[list[float]] = []

    t_full = disk_thickness / 2.0
    t_web = web_thickness / 2.0

    for i in range(segments):
        theta1 = (i / segments) * 2.0 * np.pi
        theta2 = ((i + 1) / segments) * 2.0 * np.pi
        c1, s1 = float(np.cos(theta1)), float(np.sin(theta1))
        c2, s2 = float(np.cos(theta2)), float(np.sin(theta2))

        bx1, bz1 = bore_radius * c1, bore_radius * s1
        bx2, bz2 = bore_radius * c2, bore_radius * s2
        hx1, hz1 = hub_radius * c1, hub_radius * s1
        hx2, hz2 = hub_radius * c2, hub_radius * s2
        dx1, dz1 = disk_radius * c1, disk_radius * s1
        dx2, dz2 = disk_radius * c2, disk_radius * s2

        ni1 = _normalize([-bx1, 0.0, -bz1])
        ni2 = _normalize([-bx2, 0.0, -bz2])
        nh1 = _normalize([hx1, 0.0, hz1])
        nh2 = _normalize([hx2, 0.0, hz2])
        nd1 = _normalize([dx1, 0.0, dz1])
        nd2 = _normalize([dx2, 0.0, dz2])

        # 1. Bore inner surface (inward normals, full hub thickness)
        verts_list.extend(
            [
                [bx1, t_full, bz1], [bx2, t_full, bz2], [bx1, -t_full, bz1],
                [bx2, t_full, bz2], [bx2, -t_full, bz2], [bx1, -t_full, bz1],
            ]
        )
        normals_list.extend([ni1, ni2, ni1, ni2, ni2, ni1])

        # 2. Hub shoulder outer wall – top (+t_web → +t_full, outward)
        verts_list.extend(
            [
                [hx1, t_web, hz1], [hx2, t_web, hz2], [hx1, t_full, hz1],
                [hx2, t_web, hz2], [hx2, t_full, hz2], [hx1, t_full, hz1],
            ]
        )
        normals_list.extend([nh1, nh2, nh1, nh2, nh2, nh1])

        # 3. Hub shoulder outer wall – bottom (−t_full → −t_web, outward)
        verts_list.extend(
            [
                [hx1, -t_full, hz1], [hx2, -t_full, hz2], [hx1, -t_web, hz1],
                [hx2, -t_full, hz2], [hx2, -t_web, hz2], [hx1, -t_web, hz1],
            ]
        )
        normals_list.extend([nh1, nh2, nh1, nh2, nh2, nh1])

        # 4. Rim outer surface (outward, web thickness)
        verts_list.extend(
            [
                [dx1, -t_web, dz1], [dx2, -t_web, dz2], [dx1, t_web, dz1],
                [dx2, -t_web, dz2], [dx2, t_web, dz2], [dx1, t_web, dz1],
            ]
        )
        normals_list.extend([nd1, nd2, nd1, nd2, nd2, nd1])

        # 5. Hub top annular ring (+Y, bore→hub_r at y=+t_full)
        verts_list.extend(
            [
                [hx1, t_full, hz1], [bx2, t_full, bz2], [hx2, t_full, hz2],
                [hx1, t_full, hz1], [bx1, t_full, bz1], [bx2, t_full, bz2],
            ]
        )
        normals_list.extend([[0.0, 1.0, 0.0]] * 6)

        # 6. Web top annular ring (+Y, hub_r→disk_r at y=+t_web)
        verts_list.extend(
            [
                [dx1, t_web, dz1], [hx2, t_web, hz2], [dx2, t_web, dz2],
                [dx1, t_web, dz1], [hx1, t_web, hz1], [hx2, t_web, hz2],
            ]
        )
        normals_list.extend([[0.0, 1.0, 0.0]] * 6)

        # 7. Hub bottom annular ring (−Y, bore→hub_r at y=−t_full)
        verts_list.extend(
            [
                [hx1, -t_full, hz1], [hx2, -t_full, hz2], [bx2, -t_full, bz2],
                [hx1, -t_full, hz1], [bx2, -t_full, bz2], [bx1, -t_full, bz1],
            ]
        )
        normals_list.extend([[0.0, -1.0, 0.0]] * 6)

        # 8. Web bottom annular ring (−Y, hub_r→disk_r at y=−t_web)
        verts_list.extend(
            [
                [dx1, -t_web, dz1], [dx2, -t_web, dz2], [hx2, -t_web, hz2],
                [dx1, -t_web, dz1], [hx2, -t_web, hz2], [hx1, -t_web, hz1],
            ]
        )
        normals_list.extend([[0.0, -1.0, 0.0]] * 6)

    mesh = _build_mesh(verts_list, normals_list)
    write_stl_mesh(mesh, output_path, "binary")
    return output_path


# ─────────────────────────── engine shapes ───────────────────────────────────


def create_gear(
    output_path: str,
    module: float = 1.0,
    teeth: int = 20,
    thickness: float = 1.0,
    pressure_angle_deg: float = 20.0,
    segments_per_tooth: int = 4,
) -> str:
    """Creates a spur gear mesh.

    The gear axis is aligned along the Y axis. The tooth profile is a
    simplified trapezoidal approximation of an involute gear, scaled by
    the pressure angle.

    Args:
        output_path: Output STL file path.
        module: Gear module — pitch diameter divided by tooth count (default 1.0).
        teeth: Number of teeth (default 20, minimum 4).
        thickness: Face width along Y axis (default 1.0).
        pressure_angle_deg: Pressure angle in degrees; affects tooth tip width
            (default 20.0).
        segments_per_tooth: Polygon segments used per tooth profile (default 4).

    Returns:
        Path to the output file.

    Raises:
        ValueError: If teeth < 4.

    Example:
        >>> create_gear("gear.stl", module=2.0, teeth=16, thickness=1.5)
        "gear.stl"
    """
    if teeth < 4:
        raise ValueError("teeth must be at least 4")

    pitch_radius = module * teeth / 2.0
    addendum = module
    dedendum = 1.25 * module
    outer_radius = pitch_radius + addendum
    root_radius = max(pitch_radius - dedendum, module * 0.1)

    tooth_angle = 2.0 * np.pi / teeth
    tooth_frac = 0.45
    half_root = tooth_angle * tooth_frac / 2.0
    # Pressure angle determines how much narrower the tip is than the root
    pa_rad = float(np.radians(pressure_angle_deg))
    half_tip = half_root * (1.0 - 0.5 * float(np.sin(pa_rad)))

    flank_segs = max(segments_per_tooth // 2, 1)
    tip_segs = max(segments_per_tooth // 4, 1)
    gap_segs = max(segments_per_tooth, 2)

    profile: list[tuple[float, float]] = []

    for i in range(teeth):
        base = float(i) * tooth_angle
        gap_width = tooth_angle * (1.0 - tooth_frac)

        # Gap arc at root_radius (space between teeth)
        gap_start = base - half_root - gap_width
        gap_end = base - half_root
        for k in range(gap_segs):
            t = k / gap_segs
            angle = gap_start + (gap_end - gap_start) * t
            profile.append(
                (float(root_radius * np.cos(angle)), float(root_radius * np.sin(angle)))
            )

        # Left flank: root_radius → outer_radius
        for k in range(flank_segs + 1):
            t = k / flank_segs
            r = root_radius + (outer_radius - root_radius) * t
            angle = (base - half_root) + (half_root - half_tip) * t
            profile.append((float(r * np.cos(angle)), float(r * np.sin(angle))))

        # Tip arc at outer_radius
        for k in range(1, tip_segs + 1):
            t = k / tip_segs
            angle = (base - half_tip) + 2.0 * half_tip * t
            profile.append(
                (float(outer_radius * np.cos(angle)), float(outer_radius * np.sin(angle)))
            )

        # Right flank: outer_radius → root_radius
        for k in range(1, flank_segs + 1):
            t = k / flank_segs
            r = outer_radius + (root_radius - outer_radius) * t
            angle = (base + half_tip) + (half_root - half_tip) * t
            profile.append((float(r * np.cos(angle)), float(r * np.sin(angle))))

    verts_list, normals_list = _extrude_profile(profile, thickness)
    mesh = _build_mesh(verts_list, normals_list)
    write_stl_mesh(mesh, output_path, "binary")
    return output_path


def create_spring(
    output_path: str,
    coil_radius: float = 1.0,
    wire_radius: float = 0.1,
    turns: float = 5.0,
    height: float = 5.0,
    segments: int = 32,
    wire_segments: int = 8,
) -> str:
    """Creates a helical coil spring mesh.

    The spring axis is aligned along the Y axis and the spring is centred at
    the origin. Useful for valve springs and suspension components.

    Args:
        output_path: Output STL file path.
        coil_radius: Radius of the coil helix centre line (default 1.0).
        wire_radius: Radius of the wire cross-section (default 0.1).
        turns: Number of coil turns (default 5.0).
        height: Total spring height along Y axis (default 5.0).
        segments: Number of path segments per full turn (default 32).
        wire_segments: Number of polygon sides on the wire cross-section
            (default 8).

    Returns:
        Path to the output file.

    Example:
        >>> create_spring("spring.stl", coil_radius=0.8, wire_radius=0.08, turns=6)
        "spring.stl"
    """
    verts_list: list[list[float]] = []
    normals_list: list[list[float]] = []

    total_angle = 2.0 * np.pi * turns
    coil_steps = max(int(segments * turns), 1)
    pitch = height / total_angle  # rise per radian

    # Analytic Frenet frame for the helix
    helix_len = float(np.sqrt(coil_radius**2 + pitch**2))

    def helix_center(t: float) -> npt.NDArray[np.float64]:
        return np.array(
            [
                coil_radius * np.cos(t),
                pitch * t - height / 2.0,
                coil_radius * np.sin(t),
            ]
        )

    def frenet_nb(
        t: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        n_vec = np.array([-np.cos(t), 0.0, -np.sin(t)])  # unit inward normal
        b_vec = (
            np.array([-pitch * np.sin(t), -coil_radius, pitch * np.cos(t)])
            / helix_len
        )
        return n_vec, b_vec

    def wire_ring(t: float) -> list[npt.NDArray[np.float64]]:
        c = helix_center(t)
        nv, bv = frenet_nb(t)
        pts = []
        for k in range(wire_segments):
            phi = (k / wire_segments) * 2.0 * np.pi
            pts.append(c + wire_radius * (float(np.cos(phi)) * nv + float(np.sin(phi)) * bv))
        return pts

    prev_ring = wire_ring(0.0)
    for step in range(1, coil_steps + 1):
        t = (step / coil_steps) * total_angle
        curr_ring = wire_ring(t)
        prev_center = helix_center((step - 1) / coil_steps * total_angle)
        curr_center = helix_center(t)

        for k in range(wire_segments):
            k_next = (k + 1) % wire_segments
            p0, p1 = prev_ring[k], prev_ring[k_next]
            p2, p3 = curr_ring[k], curr_ring[k_next]

            n0 = _normalize((p0 - prev_center).tolist())
            n1 = _normalize((p1 - prev_center).tolist())
            n2 = _normalize((p2 - curr_center).tolist())
            n3 = _normalize((p3 - curr_center).tolist())

            verts_list.extend([p0.tolist(), p1.tolist(), p2.tolist()])
            normals_list.extend([n0, n1, n2])
            verts_list.extend([p1.tolist(), p3.tolist(), p2.tolist()])
            normals_list.extend([n1, n3, n2])

        prev_ring = curr_ring

    mesh = _build_mesh(verts_list, normals_list)
    write_stl_mesh(mesh, output_path, "binary")
    return output_path


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

    The rod is aligned along the Y axis. The big end (crankshaft side) is
    centred at y = −length/2 and the small end (piston-pin side) at
    y = +length/2.

    Args:
        output_path: Output STL file path.
        length: Centre-to-centre distance between the two end bores (default 6.0).
        big_end_outer_radius: Outer radius of the big end bore (default 1.0).
        big_end_inner_radius: Inner (bore) radius of the big end (default 0.6).
        small_end_outer_radius: Outer radius of the small end bore (default 0.6).
        small_end_inner_radius: Inner (bore) radius of the small end (default 0.35).
        beam_width: Width of the connecting beam along X (default 0.4).
        beam_height: Depth of the connecting beam along Z (default 0.8).
        segments: Number of radial segments for the end bores (default 32).

    Returns:
        Path to the output file.

    Raises:
        ValueError: If big_end_inner_radius >= big_end_outer_radius or
            small_end_inner_radius >= small_end_outer_radius.

    Example:
        >>> create_connecting_rod("rod.stl", length=8.0)
        "rod.stl"
    """
    if big_end_inner_radius >= big_end_outer_radius:
        raise ValueError("big_end_inner_radius must be less than big_end_outer_radius")
    if small_end_inner_radius >= small_end_outer_radius:
        raise ValueError(
            "small_end_inner_radius must be less than small_end_outer_radius"
        )

    verts_list: list[list[float]] = []
    normals_list: list[list[float]] = []

    big_y = -length / 2.0
    small_y = length / 2.0
    big_tube_h = big_end_outer_radius * 1.0
    small_tube_h = small_end_outer_radius * 0.8

    # Big end tube
    _add_tube_verts(
        verts_list, normals_list,
        big_y - big_tube_h / 2.0, big_y + big_tube_h / 2.0,
        big_end_outer_radius, big_end_inner_radius, segments,
    )
    # Small end tube
    _add_tube_verts(
        verts_list, normals_list,
        small_y - small_tube_h / 2.0, small_y + small_tube_h / 2.0,
        small_end_outer_radius, small_end_inner_radius, segments,
    )

    # Connecting beam (rectangular cross-section along Y)
    hw = beam_width / 2.0
    hd = beam_height / 2.0
    beam_y_bot = big_y + big_tube_h / 2.0
    beam_y_top = small_y - small_tube_h / 2.0

    beam_corners: list[tuple[float, float]] = [
        (-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd),
    ]
    beam_normals: list[list[float]] = [
        [0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0],
    ]

    for k in range(4):
        k_next = (k + 1) % 4
        bx0, bz0 = beam_corners[k]
        bx1, bz1 = beam_corners[k_next]
        n_s = beam_normals[k]
        verts_list.extend(
            [
                [bx0, beam_y_bot, bz0], [bx1, beam_y_bot, bz1], [bx0, beam_y_top, bz0],
                [bx1, beam_y_bot, bz1], [bx1, beam_y_top, bz1], [bx0, beam_y_top, bz0],
            ]
        )
        normals_list.extend([n_s] * 6)

    bc = beam_corners
    # Beam bottom cap
    verts_list.extend(
        [
            [bc[0][0], beam_y_bot, bc[0][1]], [bc[1][0], beam_y_bot, bc[1][1]],
            [bc[2][0], beam_y_bot, bc[2][1]], [bc[0][0], beam_y_bot, bc[0][1]],
            [bc[2][0], beam_y_bot, bc[2][1]], [bc[3][0], beam_y_bot, bc[3][1]],
        ]
    )
    normals_list.extend([[0.0, -1.0, 0.0]] * 6)
    # Beam top cap
    verts_list.extend(
        [
            [bc[0][0], beam_y_top, bc[0][1]], [bc[2][0], beam_y_top, bc[2][1]],
            [bc[1][0], beam_y_top, bc[1][1]], [bc[0][0], beam_y_top, bc[0][1]],
            [bc[3][0], beam_y_top, bc[3][1]], [bc[2][0], beam_y_top, bc[2][1]],
        ]
    )
    normals_list.extend([[0.0, 1.0, 0.0]] * 6)

    mesh = _build_mesh(verts_list, normals_list)
    write_stl_mesh(mesh, output_path, "binary")
    return output_path


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

    The main axis is aligned along Y. Main journals lie on the Y axis; crank
    pins (rod journals) are offset radially by stroke/2 from the main axis and
    distributed at equal angular intervals. Solid-disc crank arms connect each
    main journal to its crank pin.

    Args:
        output_path: Output STL file path.
        throws: Number of crank throws / cylinders (default 4).
        main_journal_radius: Radius of the main bearing journals (default 0.5).
        rod_journal_radius: Radius of the connecting-rod journals / crank pins
            (default 0.4).
        journal_width: Axial width of each journal (default 0.4).
        crank_arm_thickness: Axial thickness of each crank arm disc (default 0.25).
        stroke: Piston stroke; the crank pin offset equals stroke/2 (default 2.0).
        segments: Number of radial segments for the cylinders (default 32).

    Returns:
        Path to the output file.

    Raises:
        ValueError: If throws < 1.

    Example:
        >>> create_crankshaft("crank.stl", throws=4, stroke=3.0)
        "crank.stl"
    """
    if throws < 1:
        raise ValueError("throws must be at least 1")

    verts_list: list[list[float]] = []
    normals_list: list[list[float]] = []

    crank_throw = stroke / 2.0
    # Crank arm disc is large enough to span from main axis to crank pin
    arm_disc_radius = crank_throw + rod_journal_radius + main_journal_radius * 0.3

    # Y-axis layout per throw:
    #   [main_journal] [left_arm] [crank_pin] [right_arm]
    # Final throw is followed by one extra main journal.
    section_w = journal_width + crank_arm_thickness + journal_width + crank_arm_thickness
    total_length = throws * section_w + journal_width
    y_start = -total_length / 2.0

    for i in range(throws):
        throw_angle = (i / throws) * 2.0 * np.pi
        pin_cx = crank_throw * float(np.cos(throw_angle))
        pin_cz = crank_throw * float(np.sin(throw_angle))
        # Crank arm disc centred between main axis and crank pin
        arm_cx = pin_cx / 2.0
        arm_cz = pin_cz / 2.0

        y0 = y_start + i * section_w
        y_main_top = y0 + journal_width
        y_arm_l_top = y_main_top + crank_arm_thickness
        y_pin_top = y_arm_l_top + journal_width
        y_arm_r_top = y_pin_top + crank_arm_thickness

        # Main journal i
        _add_cylinder_verts(
            verts_list, normals_list,
            0.0, y0, y_main_top, 0.0,
            main_journal_radius, segments,
            cap_bot=(i == 0), cap_top=False,
        )
        # Left crank arm (solid disc — no end caps, just the side wall)
        _add_cylinder_verts(
            verts_list, normals_list,
            arm_cx, y_main_top, y_arm_l_top, arm_cz,
            arm_disc_radius, segments,
            cap_bot=False, cap_top=False,
        )
        # Crank pin (rod journal)
        _add_cylinder_verts(
            verts_list, normals_list,
            pin_cx, y_arm_l_top, y_pin_top, pin_cz,
            rod_journal_radius, segments,
            cap_bot=True, cap_top=True,
        )
        # Right crank arm
        _add_cylinder_verts(
            verts_list, normals_list,
            arm_cx, y_pin_top, y_arm_r_top, arm_cz,
            arm_disc_radius, segments,
            cap_bot=False, cap_top=False,
        )

    # Final main journal
    y_last = y_start + throws * section_w
    _add_cylinder_verts(
        verts_list, normals_list,
        0.0, y_last, y_last + journal_width, 0.0,
        main_journal_radius, segments,
        cap_bot=False, cap_top=True,
    )

    mesh = _build_mesh(verts_list, normals_list)
    write_stl_mesh(mesh, output_path, "binary")
    return output_path


def create_valve(
    output_path: str,
    stem_radius: float = 0.15,
    stem_length: float = 3.0,
    head_radius: float = 0.6,
    head_height: float = 0.15,
    segments: int = 32,
) -> str:
    """Creates a poppet valve mesh.

    The valve stem is aligned along the Y axis. The stem tip is at
    y = +stem_length/2 and the head (combustion-chamber face) is at the
    bottom, extending from y = −stem_length/2 down by head_height.

    Args:
        output_path: Output STL file path.
        stem_radius: Radius of the cylindrical valve stem (default 0.15).
        stem_length: Length of the cylindrical stem section (default 3.0).
        head_radius: Outer radius of the valve head disc (default 0.6).
        head_height: Axial thickness of the valve head (default 0.15).
        segments: Number of radial segments (default 32).

    Returns:
        Path to the output file.

    Raises:
        ValueError: If stem_radius >= head_radius.

    Example:
        >>> create_valve("valve.stl", stem_radius=0.1, head_radius=0.5)
        "valve.stl"
    """
    if stem_radius >= head_radius:
        raise ValueError("stem_radius must be less than head_radius")

    verts_list: list[list[float]] = []
    normals_list: list[list[float]] = []

    stem_top = stem_length / 2.0
    stem_bot = -stem_length / 2.0
    head_top = stem_bot
    head_bot = stem_bot - head_height

    # Valve stem (solid cylinder, no bottom cap — connects flush to head)
    _add_cylinder_verts(
        verts_list, normals_list,
        0.0, stem_bot, stem_top, 0.0,
        stem_radius, segments,
        cap_bot=False, cap_top=True,
    )

    # Head outer wall (full cylinder at head_radius)
    for i in range(segments):
        theta1 = (i / segments) * 2.0 * np.pi
        theta2 = ((i + 1) / segments) * 2.0 * np.pi
        ox1, oz1 = head_radius * float(np.cos(theta1)), head_radius * float(np.sin(theta1))
        ox2, oz2 = head_radius * float(np.cos(theta2)), head_radius * float(np.sin(theta2))
        no1 = _normalize([ox1, 0.0, oz1])
        no2 = _normalize([ox2, 0.0, oz2])
        verts_list.extend(
            [
                [ox1, head_bot, oz1], [ox2, head_bot, oz2], [ox1, head_top, oz1],
                [ox2, head_bot, oz2], [ox2, head_top, oz2], [ox1, head_top, oz1],
            ]
        )
        normals_list.extend([no1, no2, no1, no2, no2, no1])
        # Head bottom face (full disc)
        verts_list.extend(
            [[ox1, head_bot, oz1], [ox2, head_bot, oz2], [0.0, head_bot, 0.0]]
        )
        normals_list.extend([[0.0, -1.0, 0.0]] * 3)

    # Head top annular face (annular ring from stem_radius to head_radius)
    for i in range(segments):
        theta1 = (i / segments) * 2.0 * np.pi
        theta2 = ((i + 1) / segments) * 2.0 * np.pi
        ox1 = head_radius * float(np.cos(theta1))
        oz1 = head_radius * float(np.sin(theta1))
        ox2 = head_radius * float(np.cos(theta2))
        oz2 = head_radius * float(np.sin(theta2))
        ix1 = stem_radius * float(np.cos(theta1))
        iz1 = stem_radius * float(np.sin(theta1))
        ix2 = stem_radius * float(np.cos(theta2))
        iz2 = stem_radius * float(np.sin(theta2))
        verts_list.extend(
            [
                [ox2, head_top, oz2], [ox1, head_top, oz1], [ix1, head_top, iz1],
                [ox2, head_top, oz2], [ix1, head_top, iz1], [ix2, head_top, iz2],
            ]
        )
        normals_list.extend([[0.0, 1.0, 0.0]] * 6)

    mesh = _build_mesh(verts_list, normals_list)
    write_stl_mesh(mesh, output_path, "binary")
    return output_path


def create_camshaft_lobe(
    output_path: str,
    base_radius: float = 0.8,
    lift: float = 0.4,
    lobe_width: float = 0.8,
    segments: int = 64,
) -> str:
    """Creates a cam lobe mesh.

    The lobe is a disc-like shape in the X-Z plane with an eccentric raised
    region (the cam nose) on the +X side. The lobe axis is along Y.
    A smooth cosine transition blends the nose into the base circle.

    Args:
        output_path: Output STL file path.
        base_radius: Base-circle radius (default 0.8).
        lift: Maximum lift (height of nose above base circle, default 0.4).
        lobe_width: Axial width along Y (default 0.8).
        segments: Number of circumferential polygon segments (default 64).

    Returns:
        Path to the output file.

    Example:
        >>> create_camshaft_lobe("lobe.stl", base_radius=1.0, lift=0.5)
        "lobe.stl"
    """
    # 108° (0.6π) flank half-angle gives a realistic lift-to-flank ratio that
    # matches common automotive cam profiles (short dwell, gradual ramp).
    flank_half = np.pi * 0.6

    def cam_r(theta: float) -> float:
        a = abs(theta % (2.0 * np.pi))
        if a > np.pi:
            a = 2.0 * np.pi - a
        if a >= flank_half:
            return float(base_radius)
        t = (flank_half - a) / flank_half
        return float(base_radius + lift * (0.5 - 0.5 * np.cos(np.pi * t)))

    profile: list[tuple[float, float]] = []
    for k in range(segments):
        theta = (k / segments) * 2.0 * np.pi
        r = cam_r(theta)
        profile.append((r * float(np.cos(theta)), r * float(np.sin(theta))))

    verts_list, normals_list = _extrude_profile(profile, lobe_width)
    mesh = _build_mesh(verts_list, normals_list)
    write_stl_mesh(mesh, output_path, "binary")
    return output_path


# ──────────────────────────── rocket engine shapes ───────────────────────────


def create_bell_nozzle(
    output_path: str,
    throat_radius: float = 0.15,
    exit_radius: float = 0.75,
    chamber_radius: float = 0.35,
    chamber_length: float = 0.3,
    convergent_length: float = 0.2,
    bell_length: float = 1.0,
    wall_thickness: float = 0.04,
    segments: int = 32,
    profile_points: int = 16,
) -> str:
    """Creates a bell nozzle (convergent-divergent de Laval nozzle) mesh.

    The nozzle axis is aligned along Y. ``y=0`` is the combustion-chamber
    inlet; ``y = chamber_length + convergent_length + bell_length`` is the
    nozzle exit plane.

    Profile sections:

    * **Combustion chamber** (``y ∈ [0, chamber_length]``): cylindrical bore
      at ``chamber_radius``.
    * **Convergent section** (``y ∈ [chamber_length, chamber_length +
      convergent_length]``): cosine-blend taper from ``chamber_radius`` down
      to ``throat_radius``.
    * **Divergent bell** (``y ∈ [chamber_length + convergent_length,
      total_length]``): quarter-sine flare from ``throat_radius`` out to
      ``exit_radius``, giving the characteristic bell contour.

    The mesh is a hollow thin-walled structure (outer surface, inner bore,
    and two annular end caps) parametrised by ``wall_thickness``.

    Args:
        output_path: Output STL file path.
        throat_radius: Nozzle throat (minimum) radius (default 0.15).
        exit_radius: Nozzle exit plane radius (default 0.75).
        chamber_radius: Combustion chamber bore radius (default 0.35).
        chamber_length: Length of cylindrical chamber section (default 0.3).
        convergent_length: Length of the converging section (default 0.2).
        bell_length: Length of the diverging bell section (default 1.0).
        wall_thickness: Nozzle wall thickness (default 0.04).
        segments: Number of circumferential segments (default 32).
        profile_points: Axial profile points per section (default 16).

    Returns:
        Path to the output file.

    Raises:
        ValueError: If ``throat_radius >= chamber_radius``,
            ``throat_radius >= exit_radius``, or
            ``wall_thickness >= throat_radius``.

    Example:
        >>> create_bell_nozzle("nozzle.stl", throat_radius=0.15, exit_radius=0.75)
        "nozzle.stl"
    """
    if throat_radius >= chamber_radius:
        raise ValueError("throat_radius must be less than chamber_radius")
    if throat_radius >= exit_radius:
        raise ValueError("throat_radius must be less than exit_radius")
    if wall_thickness >= throat_radius:
        raise ValueError("wall_thickness must be less than throat_radius")

    # Build inner-bore radial profile: list of (y, r_inner) pairs
    profile: list[tuple[float, float]] = []

    # 1. Combustion-chamber section (cylindrical)
    n_chamber = max(2, profile_points // 4)
    for i in range(n_chamber):
        t = i / (n_chamber - 1)
        profile.append((t * chamber_length, chamber_radius))

    # 2. Convergent section (cosine-blend taper)
    n_conv = max(3, profile_points // 4)
    y_conv_start = chamber_length
    for i in range(1, n_conv + 1):
        t = i / n_conv
        r = (throat_radius + (chamber_radius - throat_radius)
             * 0.5 * (1.0 + float(np.cos(np.pi * t))))
        profile.append((y_conv_start + t * convergent_length, r))

    # 3. Divergent bell section (quarter-sine flare)
    n_bell = max(3, profile_points // 2)
    y_bell_start = chamber_length + convergent_length
    for i in range(1, n_bell + 1):
        t = i / n_bell
        r = (throat_radius
             + (exit_radius - throat_radius) * float(np.sin(t * np.pi / 2.0)))
        profile.append((y_bell_start + t * bell_length, r))

    verts_list: list[list[float]] = []
    normals_list: list[list[float]] = []

    n_p = len(profile)

    # Build hollow-wall segments between adjacent profile rings
    for k in range(n_p - 1):
        y0_seg, r0_in = profile[k]
        y1_seg, r1_in = profile[k + 1]
        r0_out = r0_in + wall_thickness
        r1_out = r1_in + wall_thickness

        dy = y1_seg - y0_seg
        dr = r1_in - r0_in  # same for inner and outer (uniform wall thickness)

        slant_len = float(np.sqrt(dy * dy + dr * dr))
        if slant_len < 1e-12:
            slant_len = 1e-12

        # Outward normal components for outer surface (matches frustum formula)
        nxz_out = dy / slant_len
        ny_out = -dr / slant_len
        # Inward normal components for inner surface
        nxz_in = -dy / slant_len
        ny_in = dr / slant_len

        for j in range(segments):
            theta1 = (j / segments) * 2.0 * np.pi
            theta2 = ((j + 1) / segments) * 2.0 * np.pi
            c1, s1 = float(np.cos(theta1)), float(np.sin(theta1))
            c2, s2 = float(np.cos(theta2)), float(np.sin(theta2))

            # Outer-surface vertices
            ox0c1 = r0_out * c1
            oz0c1 = r0_out * s1
            ox0c2 = r0_out * c2
            oz0c2 = r0_out * s2
            ox1c1 = r1_out * c1
            oz1c1 = r1_out * s1
            ox1c2 = r1_out * c2
            oz1c2 = r1_out * s2

            # Inner-surface vertices
            ix0c1 = r0_in * c1
            iz0c1 = r0_in * s1
            ix0c2 = r0_in * c2
            iz0c2 = r0_in * s2
            ix1c1 = r1_in * c1
            iz1c1 = r1_in * s1
            ix1c2 = r1_in * c2
            iz1c2 = r1_in * s2

            no1: list[float] = [nxz_out * c1, ny_out, nxz_out * s1]
            no2: list[float] = [nxz_out * c2, ny_out, nxz_out * s2]
            ni1: list[float] = [nxz_in * c1, ny_in, nxz_in * s1]
            ni2: list[float] = [nxz_in * c2, ny_in, nxz_in * s2]

            # Outer surface (outward normals)
            verts_list.extend([
                [ox0c1, y0_seg, oz0c1], [ox0c2, y0_seg, oz0c2],
                [ox1c1, y1_seg, oz1c1],
                [ox0c2, y0_seg, oz0c2], [ox1c2, y1_seg, oz1c2],
                [ox1c1, y1_seg, oz1c1],
            ])
            normals_list.extend([no1, no2, no1, no2, no2, no1])

            # Inner surface (reversed winding → inward normals)
            verts_list.extend([
                [ix1c1, y1_seg, iz1c1], [ix1c2, y1_seg, iz1c2],
                [ix0c1, y0_seg, iz0c1],
                [ix1c2, y1_seg, iz1c2], [ix0c2, y0_seg, iz0c2],
                [ix0c1, y0_seg, iz0c1],
            ])
            normals_list.extend([ni1, ni2, ni1, ni2, ni2, ni1])

    # Inlet annular end cap (at y=0, normal −Y, chamber inlet)
    y_inlet = profile[0][0]
    r_inlet_in = profile[0][1]
    r_inlet_out = r_inlet_in + wall_thickness
    for j in range(segments):
        theta1 = (j / segments) * 2.0 * np.pi
        theta2 = ((j + 1) / segments) * 2.0 * np.pi
        ox1 = r_inlet_out * float(np.cos(theta1))
        oz1 = r_inlet_out * float(np.sin(theta1))
        ox2 = r_inlet_out * float(np.cos(theta2))
        oz2 = r_inlet_out * float(np.sin(theta2))
        ix1 = r_inlet_in * float(np.cos(theta1))
        iz1 = r_inlet_in * float(np.sin(theta1))
        ix2 = r_inlet_in * float(np.cos(theta2))
        iz2 = r_inlet_in * float(np.sin(theta2))
        verts_list.extend([
            [ix1, y_inlet, iz1], [ix2, y_inlet, iz2], [ox2, y_inlet, oz2],
            [ix1, y_inlet, iz1], [ox2, y_inlet, oz2], [ox1, y_inlet, oz1],
        ])
        normals_list.extend([[0.0, -1.0, 0.0]] * 6)

    # Exit annular end cap (at y_exit, normal +Y, nozzle exit)
    y_exit = profile[-1][0]
    r_exit_in = profile[-1][1]
    r_exit_out = r_exit_in + wall_thickness
    for j in range(segments):
        theta1 = (j / segments) * 2.0 * np.pi
        theta2 = ((j + 1) / segments) * 2.0 * np.pi
        ox1 = r_exit_out * float(np.cos(theta1))
        oz1 = r_exit_out * float(np.sin(theta1))
        ox2 = r_exit_out * float(np.cos(theta2))
        oz2 = r_exit_out * float(np.sin(theta2))
        ix1 = r_exit_in * float(np.cos(theta1))
        iz1 = r_exit_in * float(np.sin(theta1))
        ix2 = r_exit_in * float(np.cos(theta2))
        iz2 = r_exit_in * float(np.sin(theta2))
        verts_list.extend([
            [ox1, y_exit, oz1], [ox2, y_exit, oz2], [ix2, y_exit, iz2],
            [ox1, y_exit, oz1], [ix2, y_exit, iz2], [ix1, y_exit, iz1],
        ])
        normals_list.extend([[0.0, 1.0, 0.0]] * 6)

    mesh = _build_mesh(verts_list, normals_list)
    write_stl_mesh(mesh, output_path, "binary")
    return output_path


def create_injector_plate(
    output_path: str,
    radius: float = 0.35,
    thickness: float = 0.05,
    num_elements: int = 18,
    element_radius: float = 0.015,
    pattern_radius: float = 0.22,
    segments: int = 32,
) -> str:
    """Creates a propellant injector plate mesh.

    The plate is a solid flat disc with a circular array of small cylindrical
    injector-element stubs protruding from the top face (``+Y`` direction).
    The plate axis is aligned along Y with the bottom face at ``y=0`` and the
    top face at ``y=thickness``.

    Suitable for the combustion-chamber injector head of liquid rocket engines
    such as the RD-180.  Use ``array_circular`` to add a central element, or
    combine multiple rings by calling this function several times with
    different ``pattern_radius`` values.

    Args:
        output_path: Output STL file path.
        radius: Outer radius of the plate (default 0.35).
        thickness: Plate thickness along Y (default 0.05).
        num_elements: Number of injector-element stubs around the ring
            (default 18).
        element_radius: Radius of each cylindrical injector stub (default
            0.015).
        pattern_radius: Radial distance from the plate centre to each stub
            centreline (default 0.22).
        segments: Circumferential segments for the plate and stubs (default
            32).

    Returns:
        Path to the output file.

    Raises:
        ValueError: If ``pattern_radius + element_radius >= radius``.

    Example:
        >>> create_injector_plate("injector.stl", num_elements=24)
        "injector.stl"
    """
    if pattern_radius + element_radius >= radius:
        raise ValueError(
            "pattern_radius + element_radius must be less than plate radius"
        )

    verts_list: list[list[float]] = []
    normals_list: list[list[float]] = []

    # Main plate body: solid disc
    _add_cylinder_verts(
        verts_list, normals_list,
        0.0, 0.0, thickness, 0.0,
        radius, segments,
        cap_bot=True, cap_top=True,
    )

    # Injector-element stubs protruding from the top face (+Y)
    stub_height = thickness  # stubs are as tall as the plate is thick
    stub_segs = max(6, segments // 4)
    for k in range(num_elements):
        angle = (k / num_elements) * 2.0 * np.pi
        cx = pattern_radius * float(np.cos(angle))
        cz = pattern_radius * float(np.sin(angle))
        _add_cylinder_verts(
            verts_list, normals_list,
            cx, thickness, thickness + stub_height, cz,
            element_radius, stub_segs,
            cap_bot=False, cap_top=True,
        )

    mesh = _build_mesh(verts_list, normals_list)
    write_stl_mesh(mesh, output_path, "binary")
    return output_path


def create_pump_housing(
    output_path: str,
    bore_radius: float = 0.25,
    housing_radius: float = 0.6,
    housing_height: float = 0.35,
    outlet_radius: float = 0.12,
    outlet_length: float = 0.25,
    segments: int = 32,
) -> str:
    """Creates a centrifugal pump housing (volute casing) mesh.

    The housing is a hollow cylindrical casing (like a thick annular disc)
    with the pump axis along Y.  A cylindrical outlet pipe extends radially
    from the outer casing wall in the ``+X`` direction, representing the
    volute discharge port.

    The assembly models the outer casing of a turbopump such as those used
    in the RD-180 engine.  Combine with ``create_turbine_disk`` and
    ``create_turbine_blade`` (via ``array_circular``) to build a full
    turbopump stage.

    Args:
        output_path: Output STL file path.
        bore_radius: Inner bore radius for the impeller cavity (default 0.25).
        housing_radius: Outer casing radius (default 0.6).
        housing_height: Axial height of the casing along Y (default 0.35).
        outlet_radius: Radius of the discharge outlet pipe (default 0.12).
        outlet_length: Length of the outlet pipe along X (default 0.25).
        segments: Number of circumferential segments (default 32).

    Returns:
        Path to the output file.

    Raises:
        ValueError: If ``bore_radius >= housing_radius`` or
            ``outlet_radius >= housing_radius``.

    Example:
        >>> create_pump_housing("pump.stl", bore_radius=0.2, housing_radius=0.5)
        "pump.stl"
    """
    if bore_radius >= housing_radius:
        raise ValueError("bore_radius must be less than housing_radius")
    if outlet_radius >= housing_radius:
        raise ValueError("outlet_radius must be less than housing_radius")

    verts_list: list[list[float]] = []
    normals_list: list[list[float]] = []

    half_h = housing_height / 2.0

    # Main hollow cylindrical casing (outer wall, inner wall, top/bottom caps)
    _add_tube_verts(
        verts_list, normals_list,
        -half_h, half_h,
        housing_radius, bore_radius,
        segments,
    )

    # Outlet pipe: cylinder along +X from x=housing_radius to
    # x=housing_radius+outlet_length, centred at y=0, z=0.
    x_start = housing_radius
    x_end = housing_radius + outlet_length
    for j in range(segments):
        theta1 = (j / segments) * 2.0 * np.pi
        theta2 = ((j + 1) / segments) * 2.0 * np.pi
        y1 = outlet_radius * float(np.cos(theta1))
        z1 = outlet_radius * float(np.sin(theta1))
        y2 = outlet_radius * float(np.cos(theta2))
        z2 = outlet_radius * float(np.sin(theta2))

        # Outward radial normal in the Y-Z plane
        n1: list[float] = _normalize([0.0, y1, z1])
        n2: list[float] = _normalize([0.0, y2, z2])

        # Outlet cylinder side wall
        verts_list.extend([
            [x_start, y1, z1], [x_start, y2, z2], [x_end, y1, z1],
            [x_start, y2, z2], [x_end, y2, z2], [x_end, y1, z1],
        ])
        normals_list.extend([n1, n2, n1, n2, n2, n1])

        # Outlet end cap (at x_end, normal +X)
        verts_list.extend([
            [x_end, y1, z1], [x_end, y2, z2], [x_end, 0.0, 0.0],
        ])
        normals_list.extend([[1.0, 0.0, 0.0]] * 3)

    mesh = _build_mesh(verts_list, normals_list)
    write_stl_mesh(mesh, output_path, "binary")
    return output_path


# ─────────────────────────── engine transformations ──────────────────────────


def array_linear(
    path: str,
    output_path: str,
    count: int,
    dx: float,
    dy: float,
    dz: float,
) -> str:
    """Creates a linear array of mesh copies.

    Produces `count` total copies of the mesh, evenly spaced by (dx, dy, dz)
    per step. Copy 0 is at the original position; copy i is at
    (i*dx, i*dy, i*dz).

    Useful for placing multiple cylinders in an engine bank.

    Args:
        path: Input STL file path.
        output_path: Output STL file path.
        count: Total number of copies (including the original; minimum 1).
        dx: Step offset along X axis per copy.
        dy: Step offset along Y axis per copy.
        dz: Step offset along Z axis per copy.

    Returns:
        Path to the output file.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If count < 1.

    Example:
        >>> array_linear("cylinder.stl", "cylinders.stl", 4, 3.0, 0.0, 0.0)
        "cylinders.stl"
    """
    if count < 1:
        raise ValueError("count must be at least 1")

    mesh = read_stl_file(path)
    offset = np.array([dx, dy, dz], dtype=np.float32)

    all_vertices = [mesh.vertices + offset * i for i in range(count)]
    all_normals = [mesh.normals for _ in range(count)]

    combined_vertices = np.concatenate(all_vertices, axis=0)
    combined_normals = np.concatenate(all_normals, axis=0)

    combined_mesh = MeshData(
        vertices=combined_vertices,
        normals=combined_normals,
        face_count=mesh.face_count * count,
        bounding_box=_compute_bounding_box(combined_vertices),
        format="binary",
    )
    write_stl_mesh(combined_mesh, output_path, "binary")
    return output_path


def array_circular(
    path: str,
    output_path: str,
    count: int,
    axis: str = "y",
) -> str:
    """Creates a circular array of mesh copies around a coordinate axis.

    Produces `count` copies of the mesh rotated at equal angular intervals
    (360/count degrees apart) around the specified world-origin axis.

    Useful for valve arrangements, bolt patterns, and gear-wheel teeth.

    Args:
        path: Input STL file path.
        output_path: Output STL file path.
        count: Total number of copies (minimum 1).
        axis: Rotation axis ('x', 'y', or 'z'; default 'y').

    Returns:
        Path to the output file.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If count < 1 or axis is invalid.

    Example:
        >>> array_circular("bolt_hole.stl", "bolt_pattern.stl", 6, axis="y")
        "bolt_pattern.stl"
    """
    if count < 1:
        raise ValueError("count must be at least 1")
    if axis.lower() not in ("x", "y", "z"):
        raise ValueError(f"Invalid axis: {axis}. Must be 'x', 'y', or 'z'")

    mesh = read_stl_file(path)
    axis_idx = {"x": 0, "y": 1, "z": 2}[axis.lower()]

    all_vertices: list[npt.NDArray[np.float32]] = []
    all_normals: list[npt.NDArray[np.float32]] = []

    for i in range(count):
        angle = (i / count) * 2.0 * np.pi
        c = float(np.cos(angle))
        s = float(np.sin(angle))
        if axis_idx == 0:
            rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
        elif axis_idx == 1:
            rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        else:
            rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        all_vertices.append(mesh.vertices @ rot.T)
        all_normals.append(mesh.normals @ rot.T)

    combined_vertices = np.concatenate(all_vertices, axis=0)
    combined_normals = np.concatenate(all_normals, axis=0)

    combined_mesh = MeshData(
        vertices=combined_vertices,
        normals=combined_normals,
        face_count=mesh.face_count * count,
        bounding_box=_compute_bounding_box(combined_vertices),
        format="binary",
    )
    write_stl_mesh(combined_mesh, output_path, "binary")
    return output_path


def create_pyramid(
    output_path: str,
    base_radius: float = 1.0,
    height: float = 2.0,
    segments: int = 4,
) -> str:
    """Creates a regular pyramid mesh.

    The pyramid has a regular n-sided polygon base lying in the XZ plane centred
    at y=-height/2 and a single apex at y=+height/2.

    Args:
        output_path: Output STL file path.
        base_radius: Circumscribed radius of the base polygon (default 1.0).
        height: Distance from base to apex (default 2.0).
        segments: Number of base polygon vertices / sides (default 4 → square base).

    Returns:
        Path to the output file.
    """
    verts_list: list[list[float]] = []
    normals_list: list[list[float]] = []

    half_h = height / 2
    apex = [0.0, half_h, 0.0]

    for i in range(segments):
        theta1 = (i / segments) * 2 * np.pi
        theta2 = ((i + 1) / segments) * 2 * np.pi

        x1 = base_radius * np.cos(theta1)
        z1 = base_radius * np.sin(theta1)
        x2 = base_radius * np.cos(theta2)
        z2 = base_radius * np.sin(theta2)

        base1 = [x1, -half_h, z1]
        base2 = [x2, -half_h, z2]

        # Side face: base1, base2, apex
        edge1 = [x2 - x1, 0.0, z2 - z1]
        edge2 = [-x1, height, -z1]
        side_normal = _normalize(
            [
                edge1[1] * edge2[2] - edge1[2] * edge2[1],
                edge1[2] * edge2[0] - edge1[0] * edge2[2],
                edge1[0] * edge2[1] - edge1[1] * edge2[0],
            ]
        )
        verts_list.extend([base1, base2, apex])
        normals_list.extend([side_normal, side_normal, side_normal])

        # Bottom face: base2, base1, center (winding for outward-pointing downward normal)
        verts_list.extend([base2, base1, [0.0, -half_h, 0.0]])
        normals_list.extend(
            [[0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 0.0]]
        )

    mesh = _build_mesh(verts_list, normals_list)
    write_stl_mesh(mesh, output_path, "binary")
    return output_path


def create_prism(
    output_path: str,
    radius: float = 1.0,
    height: float = 2.0,
    segments: int = 6,
) -> str:
    """Creates a regular n-sided prism mesh.

    The prism has a regular n-gon cross-section with circumscribed radius
    *radius*, extruded along the Y axis from -height/2 to +height/2.

    Args:
        output_path: Output STL file path.
        radius: Circumscribed radius of the cross-section polygon (default 1.0).
        height: Prism height along Y axis (default 2.0).
        segments: Number of sides / vertices of the polygon cross-section
            (default 6 → hexagonal prism).

    Returns:
        Path to the output file.
    """
    verts_list: list[list[float]] = []
    normals_list: list[list[float]] = []

    half_h = height / 2

    for i in range(segments):
        theta1 = (i / segments) * 2 * np.pi
        theta2 = ((i + 1) / segments) * 2 * np.pi

        x1 = radius * np.cos(theta1)
        z1 = radius * np.sin(theta1)
        x2 = radius * np.cos(theta2)
        z2 = radius * np.sin(theta2)

        # Outward normal for this rectangular side face (perpendicular to the edge)
        mid_theta = (theta1 + theta2) / 2
        n = [float(np.cos(mid_theta)), 0.0, float(np.sin(mid_theta))]

        # Side rectangle (two triangles)
        verts_list.extend(
            [
                [x1, -half_h, z1],
                [x2, -half_h, z2],
                [x1, half_h, z1],
                [x2, -half_h, z2],
                [x2, half_h, z2],
                [x1, half_h, z1],
            ]
        )
        normals_list.extend([n, n, n, n, n, n])

        # Bottom cap triangles (fan from origin)
        verts_list.extend([[x2, -half_h, z2], [x1, -half_h, z1], [0.0, -half_h, 0.0]])
        normals_list.extend(
            [[0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 0.0]]
        )

        # Top cap triangles (fan from origin)
        verts_list.extend([[x1, half_h, z1], [x2, half_h, z2], [0.0, half_h, 0.0]])
        normals_list.extend([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])

    mesh = _build_mesh(verts_list, normals_list)
    write_stl_mesh(mesh, output_path, "binary")
    return output_path


def create_hemisphere(
    output_path: str,
    radius: float = 1.0,
    segments: int = 32,
) -> str:
    """Creates a hemisphere mesh (upper half of a sphere plus a flat circular base).

    The dome extends from y=0 (flat base) to y=radius (apex).  The flat
    disc at y=0 closes the shape to form a watertight solid.  This matches
    the Y-up axis convention used by the other primitive generators.

    Args:
        output_path: Output STL file path.
        radius: Hemisphere radius (default 1.0).
        segments: Number of latitude and longitude subdivisions (default 32).

    Returns:
        Path to the output file.
    """
    verts_list: list[list[float]] = []
    normals_list: list[list[float]] = []

    hemi_segs = max(segments // 2, 2)

    # Dome surface: phi from 0 (top, y=radius) to π/2 (equator, y=0).
    # Y-up parametrisation:  x = r*sin(φ)*cos(θ),  y = r*cos(φ),  z = r*sin(φ)*sin(θ)
    for i in range(hemi_segs):
        phi1 = (i / hemi_segs) * (np.pi / 2)
        phi2 = ((i + 1) / hemi_segs) * (np.pi / 2)

        for j in range(segments):
            theta1 = (j / segments) * 2 * np.pi
            theta2 = ((j + 1) / segments) * 2 * np.pi

            def _pt(phi: float, theta: float) -> list[float]:
                return [
                    float(radius * np.sin(phi) * np.cos(theta)),
                    float(radius * np.cos(phi)),
                    float(radius * np.sin(phi) * np.sin(theta)),
                ]

            p1 = _pt(phi1, theta1)
            p2 = _pt(phi1, theta2)
            p3 = _pt(phi2, theta2)
            p4 = _pt(phi2, theta1)

            # Normals equal the normalised position vector (sphere surface)
            n1 = _normalize(p1)
            n2 = _normalize(p2)
            n3 = _normalize(p3)
            n4 = _normalize(p4)

            verts_list.extend([p1, p2, p3, p1, p3, p4])
            normals_list.extend([n1, n2, n3, n1, n3, n4])

    # Flat circular base at y=0 (the equator), normal points in −Y direction.
    for j in range(segments):
        theta1 = (j / segments) * 2 * np.pi
        theta2 = ((j + 1) / segments) * 2 * np.pi

        x1 = float(radius * np.cos(theta1))
        z1 = float(radius * np.sin(theta1))
        x2 = float(radius * np.cos(theta2))
        z2 = float(radius * np.sin(theta2))

        # Winding: outer1 → outer2 → centre so normal points in −Y direction
        verts_list.extend([[x1, 0.0, z1], [x2, 0.0, z2], [0.0, 0.0, 0.0]])
        normals_list.extend(
            [[0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 0.0]]
        )

    mesh = _build_mesh(verts_list, normals_list)
    write_stl_mesh(mesh, output_path, "binary")
    return output_path


def create_wedge(
    output_path: str,
    width: float = 1.0,
    height: float = 1.0,
    depth: float = 1.0,
) -> str:
    """Creates a right-triangular wedge (triangular prism) mesh.

    The wedge has a right-triangle cross-section in the XY plane:
    the right-angle corner is at the origin, width extends along +X and
    height along +Y.  The prism is extruded symmetrically along ±Z by
    depth/2.

    Args:
        output_path: Output STL file path.
        width: Extent along the X axis (default 1.0).
        height: Extent along the Y axis (default 1.0).
        depth: Extent along the Z axis (default 1.0).

    Returns:
        Path to the output file.
    """
    hw = width / 2
    hh = height / 2
    hd = depth / 2

    # 6 vertices
    v = [
        [-hw, -hh, -hd],  # 0: bottom-left front
        [hw, -hh, -hd],   # 1: bottom-right front
        [-hw, hh, -hd],   # 2: top-left front
        [-hw, -hh, hd],   # 3: bottom-left back
        [hw, -hh, hd],    # 4: bottom-right back
        [-hw, hh, hd],    # 5: top-left back
    ]

    verts_list: list[list[float]] = []
    normals_list: list[list[float]] = []

    # Bottom face (y = -hh): 0,1,4,3 – normal [0,-1,0]
    for tri in [[v[0], v[1], v[4]], [v[0], v[4], v[3]]]:
        verts_list.extend(tri)
        normals_list.extend([[0.0, -1.0, 0.0]] * 3)

    # Left face (x = -hw): 0,3,5,2 – normal [-1,0,0]
    for tri in [[v[0], v[3], v[5]], [v[0], v[5], v[2]]]:
        verts_list.extend(tri)
        normals_list.extend([[-1.0, 0.0, 0.0]] * 3)

    # Hypotenuse face (slanted, from top-left to bottom-right):
    # vertices 1,2,5,4; compute outward normal
    # Edge in XY: from (hw,-hh) to (-hw,hh), direction (-width, height, 0)
    # Normal perpendicular to that in XY and perpendicular to Z: (height, width, 0) normalised
    slope_n = _normalize([float(height), float(width), 0.0])
    for tri in [[v[1], v[2], v[5]], [v[1], v[5], v[4]]]:
        verts_list.extend(tri)
        normals_list.extend([slope_n] * 3)

    # Front face (z = -hd): 0,2,1 – normal [0,0,-1]
    verts_list.extend([v[0], v[2], v[1]])
    normals_list.extend([[0.0, 0.0, -1.0]] * 3)

    # Back face (z = +hd): 3,4,5 – normal [0,0,+1]
    verts_list.extend([v[3], v[4], v[5]])
    normals_list.extend([[0.0, 0.0, 1.0]] * 3)

    mesh = _build_mesh(verts_list, normals_list)
    write_stl_mesh(mesh, output_path, "binary")
    return output_path


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

    The shear matrix is::

        M = [[1,  xy, xz],
             [yx,  1, yz],
             [zx, zy,  1]]

    Each parameter names which axis is shifted by how many units per unit
    along which other axis (e.g. *xy* shifts X by *xy* units for every
    unit along Y).

    Normals are transformed by the inverse-transpose of M to preserve
    geometric correctness.

    Args:
        path: Input STL file path.
        output_path: Output STL file path.
        xy: X shear factor along Y (default 0.0).
        xz: X shear factor along Z (default 0.0).
        yx: Y shear factor along X (default 0.0).
        yz: Y shear factor along Z (default 0.0).
        zx: Z shear factor along X (default 0.0).
        zy: Z shear factor along Y (default 0.0).

    Returns:
        Path to the output file.
    """
    mesh = read_stl_file(path)

    shear_matrix = np.array(
        [[1.0, xy, xz], [yx, 1.0, yz], [zx, zy, 1.0]], dtype=np.float64
    )

    mesh.vertices = (mesh.vertices.astype(np.float64) @ shear_matrix.T).astype(
        np.float32
    )

    # Transform normals by the inverse-transpose so they remain perpendicular
    inv_t = np.linalg.inv(shear_matrix).T
    normals_f64 = mesh.normals.astype(np.float64) @ inv_t.T
    # Re-normalise each normal row
    norms = np.linalg.norm(normals_f64, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    mesh.normals = (normals_f64 / norms).astype(np.float32)

    mesh.bounding_box = _compute_bounding_box(mesh.vertices)
    write_stl_mesh(mesh, output_path, "binary")
    return output_path


def twist_stl(
    path: str,
    output_path: str,
    angle: float,
    axis: str = "y",
) -> str:
    """Applies a twist (torsion) transformation to the mesh.

    Each vertex is rotated about the given axis by an angle proportional to
    its coordinate along that axis.  The twist is zero at the minimum extent
    of the mesh along *axis* and reaches *angle* degrees at the maximum
    extent.

    Useful for adding geometric twist to propeller blades, turbine blades,
    and swept wing sections.

    Args:
        path: Input STL file path.
        output_path: Output STL file path.
        angle: Total twist angle in degrees applied over the full extent of
            the mesh along *axis*.
        axis: Twist axis ('x', 'y', or 'z'; default 'y').

    Returns:
        Path to the output file.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If axis is invalid.

    Example:
        >>> twist_stl("blade.stl", "twisted.stl", 30.0, axis="y")
        "twisted.stl"
    """
    if axis.lower() not in ("x", "y", "z"):
        raise ValueError(f"Invalid axis: {axis!r}. Must be 'x', 'y', or 'z'")

    mesh = read_stl_file(path)
    axis_idx = {"x": 0, "y": 1, "z": 2}[axis.lower()]

    verts = mesh.vertices.astype(np.float64)
    norms = mesh.normals.astype(np.float64)

    coords = verts[:, axis_idx]
    coord_min = float(coords.min())
    coord_max = float(coords.max())
    coord_range = coord_max - coord_min

    if coord_range < 1e-10:
        # Mesh has no extent along the axis; write unchanged
        write_stl_mesh(mesh, output_path, "binary")
        return output_path

    # Per-vertex twist angles
    t_vert = (coords - coord_min) / coord_range
    vert_angles = float(np.radians(angle)) * t_vert
    c_v = np.cos(vert_angles)
    s_v = np.sin(vert_angles)

    if axis_idx == 0:  # X axis: rotate Y-Z
        v1, v2 = verts[:, 1].copy(), verts[:, 2].copy()
        verts[:, 1] = v1 * c_v - v2 * s_v
        verts[:, 2] = v1 * s_v + v2 * c_v
    elif axis_idx == 1:  # Y axis: rotate X-Z (right-hand rule)
        v1, v2 = verts[:, 0].copy(), verts[:, 2].copy()
        verts[:, 0] = v1 * c_v + v2 * s_v
        verts[:, 2] = -v1 * s_v + v2 * c_v
    else:  # Z axis: rotate X-Y
        v1, v2 = verts[:, 0].copy(), verts[:, 1].copy()
        verts[:, 0] = v1 * c_v - v2 * s_v
        verts[:, 1] = v1 * s_v + v2 * c_v

    # Per-face twist angles (use average vertex position for each face)
    face_coords = (coords[0::3] + coords[1::3] + coords[2::3]) / 3.0
    t_face = (face_coords - coord_min) / coord_range
    face_angles = float(np.radians(angle)) * t_face
    c_f = np.cos(face_angles)
    s_f = np.sin(face_angles)

    if axis_idx == 0:
        n1, n2 = norms[:, 1].copy(), norms[:, 2].copy()
        norms[:, 1] = n1 * c_f - n2 * s_f
        norms[:, 2] = n1 * s_f + n2 * c_f
    elif axis_idx == 1:
        n1, n2 = norms[:, 0].copy(), norms[:, 2].copy()
        norms[:, 0] = n1 * c_f + n2 * s_f
        norms[:, 2] = -n1 * s_f + n2 * c_f
    else:
        n1, n2 = norms[:, 0].copy(), norms[:, 1].copy()
        norms[:, 0] = n1 * c_f - n2 * s_f
        norms[:, 1] = n1 * s_f + n2 * c_f

    mesh.vertices = verts.astype(np.float32)
    mesh.normals = norms.astype(np.float32)
    mesh.bounding_box = _compute_bounding_box(mesh.vertices)
    write_stl_mesh(mesh, output_path, "binary")
    return output_path


def _build_mesh_from_quads(
    vertices: list[list[float]], normals: list[list[float]]
) -> MeshData:
    verts_arr = np.array(vertices, dtype=np.float32)
    norms_arr = np.array(normals, dtype=np.float32)

    n_quads = len(verts_arr) // 4
    new_verts: list[list[float]] = []
    new_norms: list[list[float]] = []

    for i in range(n_quads):
        base = i * 4
        new_verts.extend(
            [
                verts_arr[base],
                verts_arr[base + 1],
                verts_arr[base + 2],
                verts_arr[base],
                verts_arr[base + 2],
                verts_arr[base + 3],
            ]
        )
        new_norms.extend([norms_arr[base]] * 6)

    return _build_mesh(new_verts, new_norms)
