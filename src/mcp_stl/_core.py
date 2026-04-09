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
