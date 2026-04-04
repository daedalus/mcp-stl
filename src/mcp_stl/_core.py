import struct
from collections.abc import Sequence
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
