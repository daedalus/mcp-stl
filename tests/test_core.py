import struct
from pathlib import Path

import numpy as np
import pytest

from mcp_stl._core import (
    _compute_bounding_box,
    _compute_center,
    _parse_ascii,
    _parse_binary,
    array_circular,
    array_linear,
    combine_stl,
    create_airfoil,
    create_box,
    create_camshaft_lobe,
    create_capsule,
    create_cone,
    create_connecting_rod,
    create_crankshaft,
    create_cube,
    create_cylinder,
    create_ellipsoid,
    create_frustum,
    create_gear,
    create_hemisphere,
    create_piston,
    create_plane,
    create_prism,
    create_propeller_blade,
    create_pyramid,
    create_sphere,
    create_spring,
    create_torus,
    create_tube,
    create_turbine_blade,
    create_turbine_disk,
    create_valve,
    create_wedge,
    get_mesh_info,
    mirror_stl,
    read_stl_file,
    rotate_stl,
    rotate_stl_axis,
    scale_stl,
    shear_stl,
    translate_stl,
    twist_stl,
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


# ---------------------------------------------------------------------------
# Engine shapes
# ---------------------------------------------------------------------------


def test_create_gear_default(tmp_path: Path) -> None:
    output_path = tmp_path / "gear.stl"

    result = create_gear(str(output_path))

    assert result == str(output_path)
    assert output_path.exists()

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count > 0


def test_create_gear_custom_params(tmp_path: Path) -> None:
    output_path = tmp_path / "gear_custom.stl"

    create_gear(str(output_path), module=2.0, teeth=16, thickness=1.5, segments_per_tooth=6)

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count > 0
    # Outer radius = module * teeth/2 + module = 2*(16/2 + 1) = 18; diameter ≈ 36
    span_x = mesh.bounding_box["x"][1] - mesh.bounding_box["x"][0]
    assert 35.0 < span_x < 37.0


def test_create_gear_pressure_angle_effect(tmp_path: Path) -> None:
    out1 = tmp_path / "gear_pa10.stl"
    out2 = tmp_path / "gear_pa30.stl"

    # Different pressure angles should produce different face counts
    create_gear(str(out1), teeth=12, pressure_angle_deg=10.0)
    create_gear(str(out2), teeth=12, pressure_angle_deg=30.0)

    m1 = read_stl_file(str(out1))
    m2 = read_stl_file(str(out2))
    assert m1.face_count > 0
    assert m2.face_count > 0


def test_create_gear_too_few_teeth_raises(tmp_path: Path) -> None:
    output_path = tmp_path / "bad_gear.stl"

    with pytest.raises(ValueError, match="teeth must be at least 4"):
        create_gear(str(output_path), teeth=3)


def test_create_spring_default(tmp_path: Path) -> None:
    output_path = tmp_path / "spring.stl"

    result = create_spring(str(output_path))

    assert result == str(output_path)
    assert output_path.exists()

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count > 0


def test_create_spring_height(tmp_path: Path) -> None:
    output_path = tmp_path / "spring_h.stl"

    create_spring(str(output_path), coil_radius=1.0, wire_radius=0.05, turns=3, height=4.0)

    mesh = read_stl_file(str(output_path))
    total_height = mesh.bounding_box["y"][1] - mesh.bounding_box["y"][0]
    # Spring body = requested height; wire ends extend by ~wire_radius on each side.
    # For turns=3, height=4.0, wire_radius=0.05 the total should be ≈ 4.1.
    assert abs(total_height - 4.0) < 0.3


def test_create_spring_custom_params(tmp_path: Path) -> None:
    output_path = tmp_path / "spring_custom.stl"

    create_spring(
        str(output_path),
        coil_radius=0.5,
        wire_radius=0.05,
        turns=4.0,
        height=3.0,
        segments=16,
        wire_segments=6,
    )

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count > 0


def test_create_connecting_rod_default(tmp_path: Path) -> None:
    output_path = tmp_path / "rod.stl"

    result = create_connecting_rod(str(output_path))

    assert result == str(output_path)
    assert output_path.exists()

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count > 0


def test_create_connecting_rod_dimensions(tmp_path: Path) -> None:
    output_path = tmp_path / "rod_dim.stl"

    create_connecting_rod(str(output_path), length=8.0, segments=16)

    mesh = read_stl_file(str(output_path))
    height = mesh.bounding_box["y"][1] - mesh.bounding_box["y"][0]
    # Total extent along Y must be at least the rod length
    assert height >= 8.0


def test_create_connecting_rod_invalid_big_end_raises(tmp_path: Path) -> None:
    output_path = tmp_path / "rod_bad.stl"

    with pytest.raises(ValueError, match="big_end_inner_radius"):
        create_connecting_rod(str(output_path), big_end_outer_radius=0.5, big_end_inner_radius=0.8)


def test_create_connecting_rod_invalid_small_end_raises(tmp_path: Path) -> None:
    output_path = tmp_path / "rod_bad2.stl"

    with pytest.raises(ValueError, match="small_end_inner_radius"):
        create_connecting_rod(
            str(output_path), small_end_outer_radius=0.3, small_end_inner_radius=0.5
        )


def test_create_crankshaft_default(tmp_path: Path) -> None:
    output_path = tmp_path / "crank.stl"

    result = create_crankshaft(str(output_path))

    assert result == str(output_path)
    assert output_path.exists()

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count > 0


def test_create_crankshaft_single_throw(tmp_path: Path) -> None:
    output_path = tmp_path / "crank1.stl"

    create_crankshaft(str(output_path), throws=1, segments=16)

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count > 0


def test_create_crankshaft_custom_params(tmp_path: Path) -> None:
    output_path = tmp_path / "crank_custom.stl"

    create_crankshaft(
        str(output_path),
        throws=6,
        main_journal_radius=0.6,
        rod_journal_radius=0.5,
        journal_width=0.5,
        crank_arm_thickness=0.3,
        stroke=2.5,
        segments=16,
    )

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count > 0


def test_create_crankshaft_invalid_throws_raises(tmp_path: Path) -> None:
    output_path = tmp_path / "crank_bad.stl"

    with pytest.raises(ValueError, match="throws must be at least 1"):
        create_crankshaft(str(output_path), throws=0)


def test_create_valve_default(tmp_path: Path) -> None:
    output_path = tmp_path / "valve.stl"

    result = create_valve(str(output_path))

    assert result == str(output_path)
    assert output_path.exists()

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count > 0


def test_create_valve_dimensions(tmp_path: Path) -> None:
    output_path = tmp_path / "valve_dim.stl"

    create_valve(
        str(output_path),
        stem_radius=0.1,
        stem_length=4.0,
        head_radius=0.5,
        head_height=0.2,
        segments=16,
    )

    mesh = read_stl_file(str(output_path))
    total_height = mesh.bounding_box["y"][1] - mesh.bounding_box["y"][0]
    # stem_length + head_height = 4.2; allow a small tolerance
    assert abs(total_height - 4.2) < 0.05
    # Head radius sets the max X/Z extent
    span_x = mesh.bounding_box["x"][1] - mesh.bounding_box["x"][0]
    assert abs(span_x - 2 * 0.5) < 0.05


def test_create_valve_invalid_radii_raises(tmp_path: Path) -> None:
    output_path = tmp_path / "valve_bad.stl"

    with pytest.raises(ValueError, match="stem_radius must be less than head_radius"):
        create_valve(str(output_path), stem_radius=0.8, head_radius=0.5)


def test_create_camshaft_lobe_default(tmp_path: Path) -> None:
    output_path = tmp_path / "lobe.stl"

    result = create_camshaft_lobe(str(output_path))

    assert result == str(output_path)
    assert output_path.exists()

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count > 0


def test_create_camshaft_lobe_lift(tmp_path: Path) -> None:
    output_path = tmp_path / "lobe_lift.stl"

    create_camshaft_lobe(str(output_path), base_radius=1.0, lift=0.5, lobe_width=0.6)

    mesh = read_stl_file(str(output_path))
    # Maximum radius should be approximately base_radius + lift = 1.5
    max_x = mesh.bounding_box["x"][1]
    assert abs(max_x - 1.5) < 0.05
    # Minimum radius (base) should be approximately -base_radius = -1.0
    min_x = mesh.bounding_box["x"][0]
    assert abs(min_x - (-1.0)) < 0.05


def test_create_camshaft_lobe_width(tmp_path: Path) -> None:
    output_path = tmp_path / "lobe_width.stl"

    create_camshaft_lobe(str(output_path), lobe_width=1.2, segments=32)

    mesh = read_stl_file(str(output_path))
    height = mesh.bounding_box["y"][1] - mesh.bounding_box["y"][0]
    assert abs(height - 1.2) < 0.01


# ---------------------------------------------------------------------------
# Engine transformations
# ---------------------------------------------------------------------------


def test_array_linear_default(tmp_path: Path) -> None:
    src = tmp_path / "cylinder.stl"
    dst = tmp_path / "linear.stl"

    create_cylinder(str(src), radius=0.5, height=1.0, segments=8)

    result = array_linear(str(src), str(dst), 3, 2.0, 0.0, 0.0)

    assert result == str(dst)
    assert dst.exists()

    original = read_stl_file(str(src))
    arrayed = read_stl_file(str(dst))
    assert arrayed.face_count == original.face_count * 3


def test_array_linear_spacing(tmp_path: Path) -> None:
    src = tmp_path / "cube.stl"
    dst = tmp_path / "cubes.stl"

    create_cube(str(src), size=1.0)
    array_linear(str(src), str(dst), 4, 5.0, 0.0, 0.0)

    mesh = read_stl_file(str(dst))
    span = mesh.bounding_box["x"][1] - mesh.bounding_box["x"][0]
    # 4 copies at x = 0, 5, 10, 15 each centred; half-size of cube = 0.5.
    # Total span = (15 + 0.5) - (0 - 0.5) = 16.0
    assert abs(span - 16.0) < 0.01


def test_array_linear_count_one(tmp_path: Path) -> None:
    src = tmp_path / "cube.stl"
    dst = tmp_path / "single.stl"

    create_cube(str(src))
    array_linear(str(src), str(dst), 1, 10.0, 0.0, 0.0)

    original = read_stl_file(str(src))
    result = read_stl_file(str(dst))
    assert result.face_count == original.face_count
    np.testing.assert_array_almost_equal(result.vertices, original.vertices)


def test_array_linear_invalid_count_raises(tmp_path: Path) -> None:
    src = tmp_path / "cube.stl"
    dst = tmp_path / "bad.stl"

    create_cube(str(src))

    with pytest.raises(ValueError, match="count must be at least 1"):
        array_linear(str(src), str(dst), 0, 1.0, 0.0, 0.0)


def test_array_circular_default(tmp_path: Path) -> None:
    src = tmp_path / "cylinder.stl"
    dst = tmp_path / "circular.stl"

    create_cylinder(str(src), radius=0.2, height=1.0, segments=8)
    result = array_circular(str(src), str(dst), 6)

    assert result == str(dst)
    assert dst.exists()

    original = read_stl_file(str(src))
    arrayed = read_stl_file(str(dst))
    assert arrayed.face_count == original.face_count * 6


def test_array_circular_single_copy(tmp_path: Path) -> None:
    src = tmp_path / "cube.stl"
    dst = tmp_path / "single_circ.stl"

    create_cube(str(src))
    array_circular(str(src), str(dst), 1, axis="y")

    original = read_stl_file(str(src))
    result = read_stl_file(str(dst))
    assert result.face_count == original.face_count
    np.testing.assert_array_almost_equal(result.vertices, original.vertices, decimal=5)


def test_array_circular_symmetry(tmp_path: Path) -> None:
    src = tmp_path / "cyl.stl"
    dst = tmp_path / "sym.stl"

    create_cylinder(str(src), radius=0.3, height=0.5, segments=8)
    array_circular(str(src), str(dst), 4, axis="y")

    mesh = read_stl_file(str(dst))
    # Circular array of 4 copies symmetric around Y; x and z spans should be equal
    span_x = mesh.bounding_box["x"][1] - mesh.bounding_box["x"][0]
    span_z = mesh.bounding_box["z"][1] - mesh.bounding_box["z"][0]
    assert abs(span_x - span_z) < 0.05


def test_array_circular_x_axis(tmp_path: Path) -> None:
    src = tmp_path / "cube.stl"
    dst = tmp_path / "circ_x.stl"

    create_cube(str(src))
    array_circular(str(src), str(dst), 3, axis="x")

    mesh = read_stl_file(str(dst))
    assert mesh.face_count > 0


def test_array_circular_invalid_axis_raises(tmp_path: Path) -> None:
    src = tmp_path / "cube.stl"
    dst = tmp_path / "bad_circ.stl"

    create_cube(str(src))

    with pytest.raises(ValueError, match="Invalid axis"):
        array_circular(str(src), str(dst), 4, axis="w")


def test_array_circular_invalid_count_raises(tmp_path: Path) -> None:
    src = tmp_path / "cube.stl"
    dst = tmp_path / "bad_circ2.stl"

    create_cube(str(src))

    with pytest.raises(ValueError, match="count must be at least 1"):
        array_circular(str(src), str(dst), 0)


# ---------------------------------------------------------------------------
# New basic geometric shapes
# ---------------------------------------------------------------------------


def test_create_pyramid_default(tmp_path: Path) -> None:
    output_path = tmp_path / "pyramid.stl"

    result = create_pyramid(str(output_path))

    assert result == str(output_path)
    assert output_path.exists()

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count > 0


def test_create_pyramid_dimensions(tmp_path: Path) -> None:
    output_path = tmp_path / "pyramid_dim.stl"

    create_pyramid(str(output_path), base_radius=2.0, height=4.0, segments=4)

    mesh = read_stl_file(str(output_path))
    # Height should span from -height/2 to +height/2 along Y
    height_span = mesh.bounding_box["y"][1] - mesh.bounding_box["y"][0]
    assert abs(height_span - 4.0) < 0.01
    # Base radius sets the X/Z extent
    x_extent = (mesh.bounding_box["x"][1] - mesh.bounding_box["x"][0]) / 2
    assert abs(x_extent - 2.0) < 0.01


def test_create_pyramid_triangular_base(tmp_path: Path) -> None:
    output_path = tmp_path / "tetrahedron.stl"

    create_pyramid(str(output_path), base_radius=1.0, height=2.0, segments=3)

    mesh = read_stl_file(str(output_path))
    # 3-sided pyramid: 3 side faces + 3 bottom triangles = 6 faces
    assert mesh.face_count == 6


def test_create_pyramid_square_base_face_count(tmp_path: Path) -> None:
    output_path = tmp_path / "square_pyramid.stl"

    create_pyramid(str(output_path), segments=4)

    mesh = read_stl_file(str(output_path))
    # 4-sided pyramid: 4 side faces + 4 bottom triangles = 8 faces
    assert mesh.face_count == 8


def test_create_prism_default(tmp_path: Path) -> None:
    output_path = tmp_path / "prism.stl"

    result = create_prism(str(output_path))

    assert result == str(output_path)
    assert output_path.exists()

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count > 0


def test_create_prism_dimensions(tmp_path: Path) -> None:
    output_path = tmp_path / "prism_dim.stl"

    create_prism(str(output_path), radius=1.0, height=3.0, segments=6)

    mesh = read_stl_file(str(output_path))
    # Height along Y should equal 3.0
    height_span = mesh.bounding_box["y"][1] - mesh.bounding_box["y"][0]
    assert abs(height_span - 3.0) < 0.01


def test_create_prism_triangular(tmp_path: Path) -> None:
    output_path = tmp_path / "tri_prism.stl"

    create_prism(str(output_path), radius=1.0, height=2.0, segments=3)

    mesh = read_stl_file(str(output_path))
    # Triangular prism: 3 side rects (×2 tris each) + top + bottom (3 tris each) = 12 faces
    assert mesh.face_count == 12


def test_create_prism_hexagonal_face_count(tmp_path: Path) -> None:
    output_path = tmp_path / "hex_prism.stl"

    create_prism(str(output_path), segments=6)

    mesh = read_stl_file(str(output_path))
    # Hexagonal prism: 6 sides (×2) + top (6 tris) + bottom (6 tris) = 24 faces
    assert mesh.face_count == 24


def test_create_hemisphere_default(tmp_path: Path) -> None:
    output_path = tmp_path / "hemisphere.stl"

    result = create_hemisphere(str(output_path))

    assert result == str(output_path)
    assert output_path.exists()

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count > 0


def test_create_hemisphere_dimensions(tmp_path: Path) -> None:
    output_path = tmp_path / "hemisphere_dim.stl"

    create_hemisphere(str(output_path), radius=2.0, segments=16)

    mesh = read_stl_file(str(output_path))
    # Flat base at y=0, dome apex at y=radius
    assert abs(mesh.bounding_box["y"][0]) < 0.01
    assert abs(mesh.bounding_box["y"][1] - 2.0) < 0.05
    # X and Z should span ±radius
    x_half = (mesh.bounding_box["x"][1] - mesh.bounding_box["x"][0]) / 2
    assert abs(x_half - 2.0) < 0.05


def test_create_hemisphere_custom_params(tmp_path: Path) -> None:
    output_path = tmp_path / "hemisphere_custom.stl"

    create_hemisphere(str(output_path), radius=0.5, segments=8)

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count > 0
    assert abs(mesh.bounding_box["y"][1] - 0.5) < 0.05


def test_create_wedge_default(tmp_path: Path) -> None:
    output_path = tmp_path / "wedge.stl"

    result = create_wedge(str(output_path))

    assert result == str(output_path)
    assert output_path.exists()

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count > 0


def test_create_wedge_dimensions(tmp_path: Path) -> None:
    output_path = tmp_path / "wedge_dim.stl"

    create_wedge(str(output_path), width=4.0, height=2.0, depth=6.0)

    mesh = read_stl_file(str(output_path))
    x_span = mesh.bounding_box["x"][1] - mesh.bounding_box["x"][0]
    y_span = mesh.bounding_box["y"][1] - mesh.bounding_box["y"][0]
    z_span = mesh.bounding_box["z"][1] - mesh.bounding_box["z"][0]
    assert abs(x_span - 4.0) < 0.01
    assert abs(y_span - 2.0) < 0.01
    assert abs(z_span - 6.0) < 0.01


def test_create_wedge_face_count(tmp_path: Path) -> None:
    output_path = tmp_path / "wedge_faces.stl"

    create_wedge(str(output_path))

    mesh = read_stl_file(str(output_path))
    # 5 faces: bottom rect (2 tris) + left rect (2 tris) + hypotenuse rect (2 tris)
    # + front tri (1) + back tri (1) = 8 triangles
    assert mesh.face_count == 8


# ---------------------------------------------------------------------------
# New transformation: shear_stl
# ---------------------------------------------------------------------------


def test_shear_stl_default_no_op(sample_binary_stl: Path, tmp_path: Path) -> None:
    output_path = tmp_path / "shear_noop.stl"

    shear_stl(str(sample_binary_stl), str(output_path))

    original = read_stl_file(str(sample_binary_stl))
    result = read_stl_file(str(output_path))
    np.testing.assert_array_almost_equal(result.vertices, original.vertices, decimal=5)


def test_shear_stl_xy(tmp_path: Path) -> None:
    src = tmp_path / "cube.stl"
    dst = tmp_path / "sheared_xy.stl"

    create_cube(str(src), size=1.0)
    shear_stl(str(src), str(dst), xy=0.5)

    original = read_stl_file(str(src))
    result = read_stl_file(str(dst))

    # X range should be wider after shear
    orig_x_span = original.bounding_box["x"][1] - original.bounding_box["x"][0]
    new_x_span = result.bounding_box["x"][1] - result.bounding_box["x"][0]
    assert new_x_span > orig_x_span

    # Y and Z dimensions should be unchanged
    assert abs(
        (result.bounding_box["y"][1] - result.bounding_box["y"][0])
        - (original.bounding_box["y"][1] - original.bounding_box["y"][0])
    ) < 0.01


def test_shear_stl_xz(tmp_path: Path) -> None:
    src = tmp_path / "cube.stl"
    dst = tmp_path / "sheared_xz.stl"

    create_cube(str(src), size=1.0)
    shear_stl(str(src), str(dst), xz=1.0)

    original = read_stl_file(str(src))
    result = read_stl_file(str(dst))

    # X range should be wider after shearing X by Z
    orig_x_span = original.bounding_box["x"][1] - original.bounding_box["x"][0]
    new_x_span = result.bounding_box["x"][1] - result.bounding_box["x"][0]
    assert new_x_span > orig_x_span


def test_shear_stl_face_count_preserved(sample_binary_stl: Path, tmp_path: Path) -> None:
    output_path = tmp_path / "shear_faces.stl"

    original = read_stl_file(str(sample_binary_stl))
    shear_stl(str(sample_binary_stl), str(output_path), yx=0.3, zy=0.2)
    result = read_stl_file(str(output_path))

    assert result.face_count == original.face_count


def test_shear_stl_output_path_returned(sample_binary_stl: Path, tmp_path: Path) -> None:
    output_path = tmp_path / "shear_return.stl"

    result = shear_stl(str(sample_binary_stl), str(output_path), zx=0.1)

    assert result == str(output_path)
    assert output_path.exists()


# ---------------------------------------------------------------------------
# Airplane / helicopter shapes – create_airfoil
# ---------------------------------------------------------------------------


def test_create_airfoil_default(tmp_path: Path) -> None:
    output_path = tmp_path / "airfoil.stl"

    result = create_airfoil(str(output_path))

    assert result == str(output_path)
    assert output_path.exists()

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count > 0


def test_create_airfoil_dimensions(tmp_path: Path) -> None:
    output_path = tmp_path / "airfoil_dim.stl"

    create_airfoil(str(output_path), chord=2.0, span=8.0, thickness_ratio=0.12)

    mesh = read_stl_file(str(output_path))
    # Chord along X: 0 to 2.0
    x_span = mesh.bounding_box["x"][1] - mesh.bounding_box["x"][0]
    assert abs(x_span - 2.0) < 0.05
    # Span along Z: 0 to 8.0
    z_span = mesh.bounding_box["z"][1] - mesh.bounding_box["z"][0]
    assert abs(z_span - 8.0) < 0.01


def test_create_airfoil_thickness(tmp_path: Path) -> None:
    output_path = tmp_path / "airfoil_thick.stl"

    # NACA 0012: thickness = 0.12 * chord = 0.12
    create_airfoil(str(output_path), chord=1.0, span=1.0, thickness_ratio=0.12, segments=64)

    mesh = read_stl_file(str(output_path))
    y_span = mesh.bounding_box["y"][1] - mesh.bounding_box["y"][0]
    # Total thickness is approximately 2 * 0.12 * 0.15 * chord ≈ 0.12 (NACA max ~12% chord)
    assert y_span > 0.0
    assert y_span < 0.16  # must be less than 16% chord


def test_create_airfoil_thin_profile(tmp_path: Path) -> None:
    p6 = tmp_path / "naca6.stl"
    p12 = tmp_path / "naca12.stl"

    create_airfoil(str(p6), chord=1.0, span=1.0, thickness_ratio=0.06)
    create_airfoil(str(p12), chord=1.0, span=1.0, thickness_ratio=0.12)

    m6 = read_stl_file(str(p6))
    m12 = read_stl_file(str(p12))

    y6 = m6.bounding_box["y"][1] - m6.bounding_box["y"][0]
    y12 = m12.bounding_box["y"][1] - m12.bounding_box["y"][0]
    assert y6 < y12


# ---------------------------------------------------------------------------
# create_propeller_blade
# ---------------------------------------------------------------------------


def test_create_propeller_blade_default(tmp_path: Path) -> None:
    output_path = tmp_path / "blade.stl"

    result = create_propeller_blade(str(output_path))

    assert result == str(output_path)
    assert output_path.exists()

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count > 0


def test_create_propeller_blade_length(tmp_path: Path) -> None:
    output_path = tmp_path / "blade_len.stl"

    create_propeller_blade(str(output_path), length=4.0, span_segments=8)

    mesh = read_stl_file(str(output_path))
    y_span = mesh.bounding_box["y"][1] - mesh.bounding_box["y"][0]
    assert abs(y_span - 4.0) < 0.05


def test_create_propeller_blade_tapered(tmp_path: Path) -> None:
    """Blade with chord_tip < chord_root should be narrower at the tip."""
    output_path = tmp_path / "tapered.stl"

    create_propeller_blade(
        str(output_path),
        length=3.0,
        chord_root=0.6,
        chord_tip=0.1,
        twist_angle=20.0,
        segments=8,
        span_segments=4,
    )

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count > 0


def test_create_propeller_blade_twist_changes_geometry(tmp_path: Path) -> None:
    """Twisted and untwisted blades should differ in bounding-box Z extent."""
    p0 = tmp_path / "no_twist.stl"
    p45 = tmp_path / "twisted.stl"

    create_propeller_blade(str(p0), length=2.0, twist_angle=0.0, segments=8, span_segments=4)
    create_propeller_blade(str(p45), length=2.0, twist_angle=45.0, segments=8, span_segments=4)

    m0 = read_stl_file(str(p0))
    m45 = read_stl_file(str(p45))

    assert m0.face_count == m45.face_count
    # A 45° twist rotates chord into Z, so the Z extent grows noticeably
    z0 = m0.bounding_box["z"][1] - m0.bounding_box["z"][0]
    z45 = m45.bounding_box["z"][1] - m45.bounding_box["z"][0]
    assert z45 > z0 + 1e-3


# ---------------------------------------------------------------------------
# create_turbine_blade
# ---------------------------------------------------------------------------


def test_create_turbine_blade_default(tmp_path: Path) -> None:
    output_path = tmp_path / "tblade.stl"

    result = create_turbine_blade(str(output_path))

    assert result == str(output_path)
    assert output_path.exists()

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count > 0


def test_create_turbine_blade_span(tmp_path: Path) -> None:
    output_path = tmp_path / "tblade_span.stl"

    create_turbine_blade(str(output_path), span=2.0, span_segments=8)

    mesh = read_stl_file(str(output_path))
    y_span = mesh.bounding_box["y"][1] - mesh.bounding_box["y"][0]
    assert abs(y_span - 2.0) < 0.05


def test_create_turbine_blade_custom_params(tmp_path: Path) -> None:
    output_path = tmp_path / "tblade_custom.stl"

    create_turbine_blade(
        str(output_path),
        span=1.0,
        chord_root=0.3,
        chord_tip=0.2,
        twist_angle=60.0,
        thickness_ratio=0.08,
        segments=8,
        span_segments=8,
    )

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count > 0


# ---------------------------------------------------------------------------
# create_piston
# ---------------------------------------------------------------------------


def test_create_piston_default(tmp_path: Path) -> None:
    output_path = tmp_path / "piston.stl"

    result = create_piston(str(output_path))

    assert result == str(output_path)
    assert output_path.exists()

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count > 0


def test_create_piston_dimensions(tmp_path: Path) -> None:
    output_path = tmp_path / "piston_dim.stl"

    create_piston(str(output_path), bore=2.0, height=2.4, wall_thickness=0.2,
                  crown_height=0.6, segments=16)

    mesh = read_stl_file(str(output_path))
    # Height along Y = 2.4
    y_span = mesh.bounding_box["y"][1] - mesh.bounding_box["y"][0]
    assert abs(y_span - 2.4) < 0.01
    # Outer diameter = bore
    x_span = mesh.bounding_box["x"][1] - mesh.bounding_box["x"][0]
    assert abs(x_span - 2.0) < 0.05


def test_create_piston_invalid_wall_raises(tmp_path: Path) -> None:
    output_path = tmp_path / "piston_bad.stl"

    with pytest.raises(ValueError, match="wall_thickness must be less than bore/2"):
        create_piston(str(output_path), bore=1.0, wall_thickness=0.6)


def test_create_piston_invalid_crown_raises(tmp_path: Path) -> None:
    output_path = tmp_path / "piston_bad2.stl"

    with pytest.raises(ValueError, match="crown_height must be less than total height"):
        create_piston(str(output_path), height=1.0, crown_height=1.5)


# ---------------------------------------------------------------------------
# create_turbine_disk
# ---------------------------------------------------------------------------


def test_create_turbine_disk_default(tmp_path: Path) -> None:
    output_path = tmp_path / "disk.stl"

    result = create_turbine_disk(str(output_path))

    assert result == str(output_path)
    assert output_path.exists()

    mesh = read_stl_file(str(output_path))
    assert mesh.face_count > 0


def test_create_turbine_disk_dimensions(tmp_path: Path) -> None:
    output_path = tmp_path / "disk_dim.stl"

    create_turbine_disk(
        str(output_path),
        disk_radius=3.0,
        bore_radius=0.6,
        disk_thickness=0.6,
        web_thickness=0.2,
        hub_radius=1.2,
        segments=32,
    )

    mesh = read_stl_file(str(output_path))
    # Outer diameter should span ≈ 2 * disk_radius
    x_span = mesh.bounding_box["x"][1] - mesh.bounding_box["x"][0]
    assert abs(x_span - 6.0) < 0.05
    # Thickness along Y (full hub thickness = 0.6)
    y_span = mesh.bounding_box["y"][1] - mesh.bounding_box["y"][0]
    assert abs(y_span - 0.6) < 0.01


def test_create_turbine_disk_invalid_bore_raises(tmp_path: Path) -> None:
    output_path = tmp_path / "disk_bad.stl"

    with pytest.raises(ValueError, match="bore_radius must be less than hub_radius"):
        create_turbine_disk(str(output_path), bore_radius=1.5, hub_radius=1.0)


def test_create_turbine_disk_invalid_hub_raises(tmp_path: Path) -> None:
    output_path = tmp_path / "disk_bad2.stl"

    with pytest.raises(ValueError, match="hub_radius must be less than disk_radius"):
        create_turbine_disk(str(output_path), hub_radius=3.0, disk_radius=2.0)


def test_create_turbine_disk_invalid_web_raises(tmp_path: Path) -> None:
    output_path = tmp_path / "disk_bad3.stl"

    with pytest.raises(ValueError, match="web_thickness must be less than disk_thickness"):
        create_turbine_disk(str(output_path), disk_thickness=0.3, web_thickness=0.5)


# ---------------------------------------------------------------------------
# twist_stl
# ---------------------------------------------------------------------------


def test_twist_stl_zero_angle_no_op(tmp_path: Path) -> None:
    src = tmp_path / "cube.stl"
    dst = tmp_path / "twist0.stl"

    create_cube(str(src))
    twist_stl(str(src), str(dst), angle=0.0)

    original = read_stl_file(str(src))
    result = read_stl_file(str(dst))
    np.testing.assert_array_almost_equal(result.vertices, original.vertices, decimal=5)


def test_twist_stl_output_path_returned(tmp_path: Path) -> None:
    src = tmp_path / "cube.stl"
    dst = tmp_path / "twist_out.stl"

    create_cube(str(src))
    result = twist_stl(str(src), str(dst), angle=45.0, axis="y")

    assert result == str(dst)
    assert dst.exists()


def test_twist_stl_face_count_preserved(tmp_path: Path) -> None:
    src = tmp_path / "cube.stl"
    dst = tmp_path / "twist_faces.stl"

    create_cube(str(src))
    original = read_stl_file(str(src))
    twist_stl(str(src), str(dst), angle=90.0)
    result = read_stl_file(str(dst))

    assert result.face_count == original.face_count


def test_twist_stl_changes_geometry(tmp_path: Path) -> None:
    """A 90° Y-twist exchanges the wide X extent into the Z direction."""
    src = tmp_path / "box.stl"
    dst = tmp_path / "twisted_box.stl"

    # Flat box: wide in X (2.0), thin in Z (0.5), tall in Y (4.0)
    create_box(str(src), width=2.0, height=4.0, depth=0.5)
    original = read_stl_file(str(src))
    twist_stl(str(src), str(dst), angle=90.0, axis="y")
    result = read_stl_file(str(dst))

    orig_z = original.bounding_box["z"][1] - original.bounding_box["z"][0]
    new_z = result.bounding_box["z"][1] - result.bounding_box["z"][0]
    # At the twisted tip, the wide X (2.0) is rotated into Z → Z span ≫ 0.5
    assert new_z > orig_z + 0.5


def test_twist_stl_x_axis(tmp_path: Path) -> None:
    src = tmp_path / "cyl.stl"
    dst = tmp_path / "twisted_x.stl"

    create_cylinder(str(src), radius=0.5, height=2.0, segments=8)
    result_path = twist_stl(str(src), str(dst), angle=180.0, axis="x")

    assert result_path == str(dst)
    mesh = read_stl_file(str(dst))
    assert mesh.face_count > 0


def test_twist_stl_z_axis(tmp_path: Path) -> None:
    src = tmp_path / "cyl_z.stl"
    dst = tmp_path / "twisted_z.stl"

    create_cylinder(str(src), radius=0.5, height=2.0, segments=8)
    result_path = twist_stl(str(src), str(dst), angle=45.0, axis="z")

    assert result_path == str(dst)
    mesh = read_stl_file(str(dst))
    assert mesh.face_count > 0


def test_twist_stl_invalid_axis_raises(tmp_path: Path) -> None:
    src = tmp_path / "cube.stl"
    dst = tmp_path / "bad_twist.stl"

    create_cube(str(src))

    with pytest.raises(ValueError, match="Invalid axis"):
        twist_stl(str(src), str(dst), angle=30.0, axis="w")
