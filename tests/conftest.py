import pytest


@pytest.fixture
def sample_vertices() -> list[list[float]]:
    return [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]


@pytest.fixture
def sample_normals() -> list[list[float]]:
    return [[0.0, 0.0, 1.0]]
