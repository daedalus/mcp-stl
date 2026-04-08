import struct
from pathlib import Path

import numpy as np
import pytest

from mcp_stl._core import (
    _compute_bounding_box,
    _compute_center,
    _parse_ascii,
    _parse_binary,
    combine_stl,
    create_box,
    create_capsule,
    create_cone,
    create_cube,
    create_cylinder,
    create_ellipsoid,
    create_frustum,
    create_plane,
    create_sphere,
    create_torus,
    create_tube,
    get_mesh_info,
    mirror_stl,
    read_stl_file,
    rotate_stl,
    rotate_stl_axis,
    scale_stl,
    translate_stl,
    write_stl,
)


@pytest.fixture
def sample_binary_stl(tmp_path: Path) -> Path:
    vertices = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    normal = [0.0, 0.0, 1.0]

    header = b"\0" * 80
    face_count = 1

    with open(tmp_path / "sample_binary.stl", "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", face_count))

        nx, ny, nz = normal
        f.write(struct.pack("<fff", nx, ny, nz))

        for v in vertices:
            f.write(struct.pack("<fff", *v))

        f.write(struct.pack("<H", 0))

    return tmp_path / "sample_binary.stl"


@pytest.fixture
def sample_ascii_stl(tmp_path: Path) -> Path:
    content = """solid model
facet normal 0.000000 0.000000 1.000000
  outer loop
    vertex 0.000000 0.000000 0.000000
    vertex 1.000000 0.000000 0.000000
    vertex 0.000000 1.000000 0.000000
  endloop
endfacet
endsolid model
"""
    path = tmp_path / "sample_ascii.stl"
    path.write_text(content)
    return path


def test_parse_binary_single_triangle() -> None:
    normal = [0.0, 0.0, 1.0]
    vertices = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    data = b"\0" * 80
    data += struct.pack("<I", 1)
    data += struct.pack("<fff", *normal)
    for v in vertices:
        data += struct.pack("<fff", *v)
    data += struct.pack("<H", 0)

    mesh = _parse_binary(data)

    assert mesh.face_count == 1
    assert mesh.format == "binary"
    np.testing.assert_array_almost_equal(mesh.normals[0], normal)
    np.testing.assert_array_almost_equal(mesh.vertices[0], vertices[0])


def test_parse_binary_truncated_raises() -> None:
    data = b"\0" * 80
    data += struct.pack("<I", 1)

    with pytest.raises(ValueError, match="expected"):
        _parse_binary(data)


def test_parse_binary_wrong_face_count_raises() -> None:
    vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    normal = [0.0, 0.0, 1.0]

    data = b"\0" * 80
    data += struct.pack("<I", 100)
    data += struct.pack("<fff", *normal)
    for v in vertices:
        data += struct.pack("<fff", *v)
    data += struct.pack("<H", 0)

    with pytest.raises(ValueError, match="expected"):
        _parse_binary(data)


def test_compute_bounding_box_empty() -> None:
    vertices = np.zeros((0, 3), dtype=np.float32)
    bb = _compute_bounding_box(vertices)

    assert bb == {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}


def test_compute_bounding_box() -> None:
    vertices = np.array([[0.0, 0.0, 0.0], [2.0, 3.0, 4.0]], dtype=np.float32)
    bb = _compute_bounding_box(vertices)

    assert bb["x"] == (0.0, 2.0)
    assert bb["y"] == (0.0, 3.0)
    assert bb["z"] == (0.0, 4.0)


def test_read_binary_stl(sample_binary_stl: Path) -> None:
    mesh = read_stl_file(str(sample_binary_stl))

    assert mesh.face_count == 1
    assert mesh.format == "binary"
    assert len(mesh.vertices) == 3


def test_read_ascii_stl(sample_ascii_stl: Path) -> None:
    mesh = read_stl_file(str(sample_ascii_stl))

    assert mesh.face_count == 1
    assert mesh.format == "ascii"
    assert len(mesh.vertices) == 3


def test_get_mesh_info(sample_binary_stl: Path) -> None:
    info = get_mesh_info(str(sample_binary_stl))

    assert info["face_count"] == 1
    assert info["format"] == "binary"
    assert "bounding_box" in info
    assert "center" in info


def test_get_mesh_info_nonexistent_raises() -> None:
    with pytest.raises(FileNotFoundError):
        get_mesh_info("/nonexistent/file.stl")


def test_read_stl_file_nonexistent_raises() -> None:
    with pytest.raises(FileNotFoundError):
        read_stl_file("/nonexistent/file.stl")


def test_translate_stl(sample_binary_stl: Path, tmp_path: Path) -> None:
    output_path = tmp_path / "translated.stl"

    result = translate_stl(str(sample_binary_stl), str(output_path), 10.0, 20.0, 30.0)

    assert result == str(output_path)
    assert output_path.exists()

    mesh = read_stl_file(str(output_path))
    np.testing.assert_array_almost_equal(mesh.vertices[0], [10.0, 20.0, 30.0])


def test_rotate_stl_x_axis(sample_binary_stl: Path, tmp_path: Path) -> None:
    output_path = tmp_path / "rotated.stl"

    result = rotate_stl(str(sample_binary_stl), str(output_path), "x", 90.0)

    assert result == str(output_path)
    assert output_path.exists()


def test_rotate_stl_invalid_axis_raises(
    sample_binary_stl: Path, tmp_path: Path
) -> None:
    output_path = tmp_path / "rotated.stl"

    with pytest.raises(ValueError, match="Invalid axis"):
        rotate_stl(str(sample_binary_stl), str(output_path), "invalid", 90.0)


def test_rotate_stl_zero_angle_no_op(sample_binary_stl: Path, tmp_path: Path) -> None:
    output_path = tmp_path / "rotated.stl"

    rotate_stl(str(sample_binary_stl), str(output_path), "y", 0.0)

    mesh = read_stl_file(str(output_path))
    original = read_stl_file(str(sample_binary_stl))
    np.testing.assert_array_almost_equal(mesh.vertices, original.vertices)


def test_scale_stl(sample_binary_stl: Path, tmp_path: Path) -> None:
    output_path = tmp_path / "scaled.stl"

    result = scale_stl(str(sample_binary_stl), str(output_path), 2.0, 2.0, 2.0)

    assert result == str(output_path)
    assert output_path.exists()

    mesh = read_stl_file(str(output_path))
    np.testing.assert_array_almost_equal(mesh.vertices[1], [2.0, 0.0, 0.0])


def test_write_stl_binary(tmp_path: Path) -> None:
    vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    normals = [[0.0, 0.0, 1.0]]
    output_path = tmp_path / "written.stl"

    result = write_stl(vertices, normals, str(output_path))

    assert result == str(output_path)
    assert output_path.exists()

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count == 1


def test_write_stl_ascii(tmp_path: Path) -> None:
    vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    normals = [[0.0, 0.0, 1.0]]
    output_path = tmp_path / "written_ascii.stl"

    result = write_stl(vertices, normals, str(output_path), format="ascii")

    assert result == str(output_path)
    assert output_path.exists()

    content = output_path.read_text()
    assert content.startswith("solid")


def test_edge_case_negative_coordinates(tmp_path: Path) -> None:
    vertices = [[-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0]]
    normals = [[0.0, 0.0, -1.0]]
    output_path = tmp_path / "negative.stl"

    write_stl(vertices, normals, str(output_path))

    mesh = read_stl_file(str(output_path))
    assert mesh.vertices[0][0] == -1.0


def test_edge_case_zero_scale(tmp_path: Path) -> None:
    vertices = [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [1.0, 2.0, 1.0]]
    normals = [[0.0, 0.0, 1.0]]
    input_path = tmp_path / "input.stl"
    output_path = tmp_path / "zero_scale.stl"

    write_stl(vertices, normals, str(input_path))
    scale_stl(str(input_path), str(output_path), 0.0, 1.0, 1.0)

    mesh = read_stl_file(str(output_path))
    np.testing.assert_array_almost_equal(mesh.vertices[:, 0], [0.0, 0.0, 0.0])


def test_parse_ascii_basic(sample_ascii_stl: Path) -> None:
    with open(sample_ascii_stl) as f:
        mesh = _parse_ascii(f)

    assert mesh.face_count == 1
    assert mesh.format == "ascii"
    assert len(mesh.vertices) == 3


def test_parse_ascii_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.stl"
    path.write_text("solid model\nendsolid model\n")

    with open(path) as f:
        mesh = _parse_ascii(f)

    assert mesh.face_count == 0
    assert mesh.format == "ascii"


def test_compute_center_empty() -> None:
    vertices = np.zeros((0, 3), dtype=np.float32)
    center = _compute_center(vertices)

    assert center == {"x": 0.0, "y": 0.0, "z": 0.0}


def test_compute_center() -> None:
    vertices = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]], dtype=np.float32)
    center = _compute_center(vertices)

    assert center == {"x": 1.0, "y": 1.0, "z": 1.0}


def test_rotate_stl_y_axis(sample_binary_stl: Path, tmp_path: Path) -> None:
    output_path = tmp_path / "rotated_y.stl"

    result = rotate_stl(str(sample_binary_stl), str(output_path), "y", 90.0)

    assert result == str(output_path)
    assert output_path.exists()


def test_rotate_stl_z_axis(sample_binary_stl: Path, tmp_path: Path) -> None:
    output_path = tmp_path / "rotated_z.stl"

    result = rotate_stl(str(sample_binary_stl), str(output_path), "z", 90.0)

    assert result == str(output_path)
    assert output_path.exists()


def test_rotate_stl_360_degrees(sample_binary_stl: Path, tmp_path: Path) -> None:
    output_path = tmp_path / "rotated_360.stl"

    rotate_stl(str(sample_binary_stl), str(output_path), "x", 360.0)

    mesh = read_stl_file(str(output_path))
    original = read_stl_file(str(sample_binary_stl))
    np.testing.assert_array_almost_equal(mesh.vertices, original.vertices)


def test_create_cube(tmp_path: Path) -> None:
    output_path = tmp_path / "cube.stl"

    result = create_cube(str(output_path))

    assert result == str(output_path)
    assert output_path.exists()

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count == 12


def test_create_cube_custom_size(tmp_path: Path) -> None:
    output_path = tmp_path / "cube_large.stl"

    create_cube(str(output_path), size=5.0)

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count == 12
    assert mesh.bounding_box["x"][1] - mesh.bounding_box["x"][0] == 5.0


def test_create_cube_custom_center(tmp_path: Path) -> None:
    output_path = tmp_path / "cube_centered.stl"

    create_cube(str(output_path), center=[10.0, 20.0, 30.0])

    mesh = read_stl_file(str(output_path))
    assert mesh.bounding_box["x"][0] == 9.5
    assert mesh.bounding_box["y"][0] == 19.5
    assert mesh.bounding_box["z"][0] == 29.5


def test_create_sphere(tmp_path: Path) -> None:
    output_path = tmp_path / "sphere.stl"

    result = create_sphere(str(output_path))

    assert result == str(output_path)
    assert output_path.exists()

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count > 0


def test_create_sphere_custom_params(tmp_path: Path) -> None:
    output_path = tmp_path / "sphere_custom.stl"

    create_sphere(str(output_path), radius=2.0, segments=8)

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count > 0


def test_create_cylinder(tmp_path: Path) -> None:
    output_path = tmp_path / "cylinder.stl"

    result = create_cylinder(str(output_path))

    assert result == str(output_path)
    assert output_path.exists()

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count > 0


def test_create_cylinder_custom_params(tmp_path: Path) -> None:
    output_path = tmp_path / "cylinder_custom.stl"

    create_cylinder(str(output_path), radius=0.5, height=5.0, segments=16)

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count > 0


def test_create_cone(tmp_path: Path) -> None:
    output_path = tmp_path / "cone.stl"

    result = create_cone(str(output_path))

    assert result == str(output_path)
    assert output_path.exists()

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count > 0


def test_create_cone_custom_params(tmp_path: Path) -> None:
    output_path = tmp_path / "cone_custom.stl"

    create_cone(str(output_path), radius=2.0, height=4.0, segments=8)

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count > 0


def test_create_torus(tmp_path: Path) -> None:
    output_path = tmp_path / "torus.stl"

    result = create_torus(str(output_path))

    assert result == str(output_path)
    assert output_path.exists()

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count > 0


def test_create_torus_custom_params(tmp_path: Path) -> None:
    output_path = tmp_path / "torus_custom.stl"

    create_torus(
        str(output_path),
        major_radius=3.0,
        minor_radius=0.5,
        major_segments=8,
        minor_segments=4,
    )

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count > 0


def test_create_plane(tmp_path: Path) -> None:
    output_path = tmp_path / "plane.stl"

    result = create_plane(str(output_path))

    assert result == str(output_path)
    assert output_path.exists()

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count == 2


def test_create_plane_custom_size(tmp_path: Path) -> None:
    output_path = tmp_path / "plane_custom.stl"

    create_plane(str(output_path), width=10.0, height=5.0)

    mesh = read_stl_file(str(output_path))
    assert mesh.bounding_box["x"][1] - mesh.bounding_box["x"][0] == 10.0
    assert mesh.bounding_box["z"][1] - mesh.bounding_box["z"][0] == 5.0


def test_get_mesh_info_ascii(sample_ascii_stl: Path) -> None:
    info = get_mesh_info(str(sample_ascii_stl))

    assert info["face_count"] == 1
    assert info["format"] == "ascii"
    assert "bounding_box" in info
    assert "center" in info


def test_translate_stl_negative(sample_binary_stl: Path, tmp_path: Path) -> None:
    output_path = tmp_path / "translated_neg.stl"

    result = translate_stl(str(sample_binary_stl), str(output_path), -5.0, -10.0, -15.0)

    assert result == str(output_path)
    assert output_path.exists()

    mesh = read_stl_file(str(output_path))
    np.testing.assert_array_almost_equal(mesh.vertices[0], [-5.0, -10.0, -15.0])


def test_scale_stl_non_uniform(tmp_path: Path) -> None:
    vertices = [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [1.0, 2.0, 1.0]]
    normals = [[0.0, 0.0, 1.0]]
    input_path = tmp_path / "input.stl"
    output_path = tmp_path / "scaled.stl"

    write_stl(vertices, normals, str(input_path))
    scale_stl(str(input_path), str(output_path), 2.0, 3.0, 4.0)

    mesh = read_stl_file(str(output_path))
    np.testing.assert_array_almost_equal(mesh.vertices[1], [4.0, 3.0, 4.0])


# ---------------------------------------------------------------------------
# New shapes
# ---------------------------------------------------------------------------


def test_create_box(tmp_path: Path) -> None:
    output_path = tmp_path / "box.stl"

    result = create_box(str(output_path))

    assert result == str(output_path)
    assert output_path.exists()

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count == 12


def test_create_box_custom_dimensions(tmp_path: Path) -> None:
    output_path = tmp_path / "box_custom.stl"

    create_box(str(output_path), width=4.0, height=2.0, depth=1.0)

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count == 12
    w = mesh.bounding_box["x"][1] - mesh.bounding_box["x"][0]
    h = mesh.bounding_box["y"][1] - mesh.bounding_box["y"][0]
    d = mesh.bounding_box["z"][1] - mesh.bounding_box["z"][0]
    assert abs(w - 4.0) < 1e-4
    assert abs(h - 2.0) < 1e-4
    assert abs(d - 1.0) < 1e-4


def test_create_box_custom_center(tmp_path: Path) -> None:
    output_path = tmp_path / "box_center.stl"

    create_box(str(output_path), width=1.0, height=1.0, depth=1.0, center=[5.0, 5.0, 5.0])

    mesh = read_stl_file(str(output_path))
    assert abs(mesh.bounding_box["x"][0] - 4.5) < 1e-4
    assert abs(mesh.bounding_box["y"][0] - 4.5) < 1e-4
    assert abs(mesh.bounding_box["z"][0] - 4.5) < 1e-4


def test_create_capsule(tmp_path: Path) -> None:
    output_path = tmp_path / "capsule.stl"

    result = create_capsule(str(output_path))

    assert result == str(output_path)
    assert output_path.exists()

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count > 0


def test_create_capsule_custom_params(tmp_path: Path) -> None:
    output_path = tmp_path / "capsule_custom.stl"

    create_capsule(str(output_path), radius=0.3, height=1.5, segments=16)

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count > 0
    # Total height should be cylinder height + 2 * radius (two hemispheres)
    total_height = mesh.bounding_box["y"][1] - mesh.bounding_box["y"][0]
    assert abs(total_height - (1.5 + 2 * 0.3)) < 0.05


def test_create_ellipsoid(tmp_path: Path) -> None:
    output_path = tmp_path / "ellipsoid.stl"

    result = create_ellipsoid(str(output_path))

    assert result == str(output_path)
    assert output_path.exists()

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count > 0


def test_create_ellipsoid_custom_radii(tmp_path: Path) -> None:
    output_path = tmp_path / "ellipsoid_custom.stl"

    create_ellipsoid(str(output_path), rx=2.0, ry=1.0, rz=0.5, segments=16)

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count > 0
    ex = (mesh.bounding_box["x"][1] - mesh.bounding_box["x"][0]) / 2
    ey = (mesh.bounding_box["y"][1] - mesh.bounding_box["y"][0]) / 2
    assert abs(ex - 2.0) < 0.1
    assert abs(ey - 1.0) < 0.1


def test_create_frustum(tmp_path: Path) -> None:
    output_path = tmp_path / "frustum.stl"

    result = create_frustum(str(output_path))

    assert result == str(output_path)
    assert output_path.exists()

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count > 0


def test_create_frustum_custom_params(tmp_path: Path) -> None:
    output_path = tmp_path / "frustum_custom.stl"

    create_frustum(str(output_path), bottom_radius=1.5, top_radius=0.3, height=3.0, segments=16)

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count > 0
    # Height along y should equal 3.0
    height = mesh.bounding_box["y"][1] - mesh.bounding_box["y"][0]
    assert abs(height - 3.0) < 0.01


def test_create_tube(tmp_path: Path) -> None:
    output_path = tmp_path / "tube.stl"

    result = create_tube(str(output_path))

    assert result == str(output_path)
    assert output_path.exists()

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count > 0


def test_create_tube_custom_params(tmp_path: Path) -> None:
    output_path = tmp_path / "tube_custom.stl"

    create_tube(str(output_path), outer_radius=1.0, inner_radius=0.5, height=3.0, segments=16)

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count > 0
    height = mesh.bounding_box["y"][1] - mesh.bounding_box["y"][0]
    assert abs(height - 3.0) < 0.01


def test_create_tube_invalid_radii_raises(tmp_path: Path) -> None:
    output_path = tmp_path / "tube_bad.stl"

    with pytest.raises(ValueError, match="inner_radius"):
        create_tube(str(output_path), outer_radius=0.5, inner_radius=1.0)


# ---------------------------------------------------------------------------
# New transformations
# ---------------------------------------------------------------------------


def test_mirror_stl_x(sample_binary_stl: Path, tmp_path: Path) -> None:
    output_path = tmp_path / "mirrored_x.stl"

    result = mirror_stl(str(sample_binary_stl), str(output_path), "x")

    assert result == str(output_path)
    assert output_path.exists()

    original = read_stl_file(str(sample_binary_stl))
    mirrored = read_stl_file(str(output_path))
    # X coordinates should be negated
    np.testing.assert_array_almost_equal(
        mirrored.vertices[:, 0], -original.vertices[:, 0]
    )
    # Y and Z should be unchanged
    np.testing.assert_array_almost_equal(mirrored.vertices[:, 1], original.vertices[:, 1])
    np.testing.assert_array_almost_equal(mirrored.vertices[:, 2], original.vertices[:, 2])


def test_mirror_stl_y(sample_binary_stl: Path, tmp_path: Path) -> None:
    output_path = tmp_path / "mirrored_y.stl"

    mirror_stl(str(sample_binary_stl), str(output_path), "y")

    original = read_stl_file(str(sample_binary_stl))
    mirrored = read_stl_file(str(output_path))
    np.testing.assert_array_almost_equal(
        mirrored.vertices[:, 1], -original.vertices[:, 1]
    )


def test_mirror_stl_z(sample_binary_stl: Path, tmp_path: Path) -> None:
    output_path = tmp_path / "mirrored_z.stl"

    mirror_stl(str(sample_binary_stl), str(output_path), "z")

    original = read_stl_file(str(sample_binary_stl))
    mirrored = read_stl_file(str(output_path))
    np.testing.assert_array_almost_equal(
        mirrored.vertices[:, 2], -original.vertices[:, 2]
    )


def test_mirror_stl_invalid_axis_raises(sample_binary_stl: Path, tmp_path: Path) -> None:
    output_path = tmp_path / "mirrored_bad.stl"

    with pytest.raises(ValueError, match="Invalid axis"):
        mirror_stl(str(sample_binary_stl), str(output_path), "w")


def test_rotate_stl_axis_cardinal_z_matches_rotate_stl(
    sample_binary_stl: Path, tmp_path: Path
) -> None:
    out1 = tmp_path / "rz_cardinal.stl"
    out2 = tmp_path / "rz_axis.stl"

    rotate_stl(str(sample_binary_stl), str(out1), "z", 45.0)
    rotate_stl_axis(str(sample_binary_stl), str(out2), 0.0, 0.0, 1.0, 45.0)

    m1 = read_stl_file(str(out1))
    m2 = read_stl_file(str(out2))
    np.testing.assert_array_almost_equal(m1.vertices, m2.vertices, decimal=4)


def test_rotate_stl_axis_zero_vector_raises(
    sample_binary_stl: Path, tmp_path: Path
) -> None:
    output_path = tmp_path / "rot_bad.stl"

    with pytest.raises(ValueError, match="zero vector"):
        rotate_stl_axis(str(sample_binary_stl), str(output_path), 0.0, 0.0, 0.0, 45.0)


def test_rotate_stl_axis_arbitrary(sample_binary_stl: Path, tmp_path: Path) -> None:
    output_path = tmp_path / "rot_arb.stl"

    result = rotate_stl_axis(str(sample_binary_stl), str(output_path), 1.0, 1.0, 0.0, 90.0)

    assert result == str(output_path)
    assert output_path.exists()

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count == 1


def test_combine_stl(tmp_path: Path) -> None:
    cube_path = tmp_path / "cube.stl"
    sphere_path = tmp_path / "sphere.stl"
    combined_path = tmp_path / "combined.stl"

    create_cube(str(cube_path))
    create_sphere(str(sphere_path), segments=8)

    result = combine_stl([str(cube_path), str(sphere_path)], str(combined_path))

    assert result == str(combined_path)
    assert combined_path.exists()

    cube_mesh = read_stl_file(str(cube_path))
    sphere_mesh = read_stl_file(str(sphere_path))
    combined_mesh = read_stl_file(str(combined_path))

    assert combined_mesh.face_count == cube_mesh.face_count + sphere_mesh.face_count


def test_combine_stl_single_file(tmp_path: Path) -> None:
    cube_path = tmp_path / "cube.stl"
    combined_path = tmp_path / "combined.stl"

    create_cube(str(cube_path))

    combine_stl([str(cube_path)], str(combined_path))

    original = read_stl_file(str(cube_path))
    combined = read_stl_file(str(combined_path))
    assert combined.face_count == original.face_count


def test_combine_stl_empty_list_raises(tmp_path: Path) -> None:
    combined_path = tmp_path / "combined.stl"

    with pytest.raises(ValueError, match="At least one"):
        combine_stl([], str(combined_path))

