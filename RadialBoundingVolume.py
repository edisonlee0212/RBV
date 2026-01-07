from __future__ import annotations
import csv
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import math
import random

try:
    import numpy as np  # optional
except Exception:  # pragma: no cover
    np = None  # type: ignore


Vec3 = Tuple[float, float, float]
Vec2 = Tuple[float, float]


@dataclass
class _RBVSector:
    max_distance: float = 0.0


@dataclass
class Vertex:
    position: Vec3
    normal: Vec3
    tex_coord: Vec2


@dataclass
class MeshData:
    """Engine-agnostic mesh container."""

    vertices: List[Vertex]
    indices: List[int]


class RadialBoundingVolume:
    """
    RBV core + a direct port of C++ GenerateMesh() that produces engine-agnostic MeshData.

    Coordinate convention:
      - y is height
      - radius in xz-plane
      - sector selection uses atan2(x, z) (note order), consistent with glm::atan(x, z).
    """

    def __init__(
        self, layer_amount: int = 16, sector_amount: int = 16, offset: float = 0.0
    ) -> None:
        if layer_amount <= 0 or sector_amount <= 0:
            raise ValueError("layer_amount and sector_amount must be positive.")
        self.layer_amount = int(layer_amount)
        self.sector_amount = int(sector_amount)
        self.offset = float(offset)

        self.max_height: float = 0.0
        self.max_radius: float = 0.0

        self.layers: List[List[_RBVSector]] = []
        self.meshes: List[MeshData] = []  # per-layer meshes, as in the C++ code
        self._mesh_generated: bool = False

        # sampling weights (r^2) used by get_random_point()
        self._layer_weights: List[float] = []
        self._sector_weights: List[List[float]] = []
        self._total_weight: float = 0.0

        self._resize_volumes()

    # -------------------------
    # Input handling (now with NumPy)
    # -------------------------

    @staticmethod
    def _is_numpy_vec3(p: Any) -> bool:
        if np is None:
            return False
        return (
            isinstance(p, np.ndarray)
            and p.shape == (3,)
            and p.dtype.kind in ("f", "i", "u")
        )

    @staticmethod
    def _as_vec3(p: Any) -> Vec3:
        """
        Accepts:
          - (x, y, z) tuple/list
          - numpy.ndarray shape (3,)
          - object with attributes .x .y .z
          - object/dict with .position / 'position'
        """
        if p is None:
            raise ValueError("Point is None.")

        # numpy array (explicitly requested)
        if RadialBoundingVolume._is_numpy_vec3(p):
            return (float(p[0]), float(p[1]), float(p[2]))

        # dict
        if isinstance(p, dict):
            if "position" in p:
                return RadialBoundingVolume._as_vec3(p["position"])
            if all(k in p for k in ("x", "y", "z")):
                return (float(p["x"]), float(p["y"]), float(p["z"]))

        # tuple/list
        if isinstance(p, (tuple, list)) and len(p) == 3:
            return (float(p[0]), float(p[1]), float(p[2]))

        # attribute forms
        if hasattr(p, "position"):
            return RadialBoundingVolume._as_vec3(getattr(p, "position"))
        if all(hasattr(p, k) for k in ("x", "y", "z")):
            return (
                float(getattr(p, "x")),
                float(getattr(p, "y")),
                float(getattr(p, "z")),
            )

        raise TypeError(f"Unsupported point/node format: {type(p)}")

    def _height_level(self) -> float:
        return self.max_height / self.layer_amount if self.layer_amount > 0 else 0.0

    def _slice_angle_deg(self) -> float:
        return 360.0 / self.sector_amount if self.sector_amount > 0 else 360.0

    # -------------------------
    # Slice selection / query
    # -------------------------

    def select_slice(self, position: Union[Vec3, Any]) -> Tuple[int, int]:
        """Port of C++ SelectSlice(). Accepts Vec3 or numpy array (3,)."""
        x, y, z = self._as_vec3(position)
        height_level = self._height_level()
        slice_angle = self._slice_angle_deg()

        # layer
        if height_level <= 0.0:
            layer_idx = 0
        else:
            layer_idx = int(y / height_level)
            if layer_idx < 0:
                layer_idx = 0
            if layer_idx >= self.layer_amount:
                layer_idx = self.layer_amount - 1

        # sector
        if x == 0.0 and z == 0.0:
            sector_idx = 0
        else:
            ang_deg = math.degrees(math.atan2(x, z)) + 180.0  # [0, 360)
            sector_idx = int(ang_deg / slice_angle) if slice_angle > 0.0 else 0
            if sector_idx >= self.sector_amount:
                sector_idx = self.sector_amount - 1

        return (layer_idx, sector_idx)

    @staticmethod
    def _radius_xz(position: Vec3) -> float:
        x, _, z = position
        return math.hypot(x, z)

    # -------------------------
    # Build RBV from points/graph
    # -------------------------

    @classmethod
    def from_skeletal_graph(
        cls,
        nodes: Iterable[Any],
        *,
        layer_amount: int = 16,
        sector_amount: int = 16,
        offset: float = 0.0,
        origin: Optional[Union[Vec3, Any]] = None,
        drop_negative_y: bool = False,
    ) -> "RadialBoundingVolume":
        rbv = cls(layer_amount=layer_amount, sector_amount=sector_amount, offset=offset)
        ox, oy, oz = cls._as_vec3(origin) if origin is not None else (0.0, 0.0, 0.0)

        pts: List[Vec3] = []
        for n in nodes:
            x, y, z = cls._as_vec3(n)
            x -= ox
            y -= oy
            z -= oz
            if drop_negative_y and y < 0.0:
                continue
            pts.append((x, y, z))

        rbv.calculate_volume(pts)
        return rbv

    def _resize_volumes(self) -> None:
        self.layers = [
            [_RBVSector(0.0) for _ in range(self.sector_amount)]
            for _ in range(self.layer_amount)
        ]
        self.meshes = []
        self._mesh_generated = False

    def calculate_volume(self, points: Sequence[Union[Vec3, Any]]) -> None:
        """Port of C++ CalculateVolume(). Accepts Vec3 or numpy arrays."""
        self._resize_volumes()
        self.max_height = 0.0
        self.max_radius = 0.0

        pts: List[Vec3] = [self._as_vec3(p) for p in points]

        # bounds
        for x, y, z in pts:
            if y > self.max_height:
                self.max_height = y
            r = math.hypot(x, z)
            if r > self.max_radius:
                self.max_radius = r

        # fill
        for p in pts:
            layer_idx, sector_idx = self.select_slice(p)
            r = self._radius_xz(p)

            if r <= self.offset:
                target = r + self.offset
                for s in self.layers[layer_idx]:
                    if s.max_distance < target:
                        s.max_distance = target
            else:
                target = r + self.offset
                cell = self.layers[layer_idx][sector_idx]
                if cell.max_distance < target:
                    cell.max_distance = target

        # GenerateMesh(); CalculateSizes();
        self.generate_mesh()
        self._calculate_sizes()

    def _calculate_sizes(self) -> None:
        self._layer_weights = [0.0 for _ in range(self.layer_amount)]
        self._sector_weights = [
            [0.0 for _ in range(self.sector_amount)] for _ in range(self.layer_amount)
        ]
        self._total_weight = 0.0

        for i in range(self.layer_amount):
            layer_sum = 0.0
            for j in range(self.sector_amount):
                r = self.layers[i][j].max_distance
                w = r * r
                self._sector_weights[i][j] = w
                layer_sum += w
            self._layer_weights[i] = layer_sum
            self._total_weight += layer_sum

    # -------------------------
    # Mesh generation (direct port of C++ GenerateMesh)
    # -------------------------

    @staticmethod
    def _normalize(v: Vec3) -> Vec3:
        x, y, z = v
        n = math.sqrt(x * x + y * y + z * z)
        if n <= 1e-12:
            return (0.0, 0.0, 0.0)
        return (x / n, y / n, z / n)

    def generate_mesh(self) -> List[MeshData]:
        """
        Port of C++ GenerateMesh().

        Output:
          - self.meshes: list of MeshData, one per layer (tierIndex), same structure as m_boundMeshes.

        Important fidelity notes:
          - This reproduces the vertex count/layout and indexing pattern from the C++ code,
            including the two vertex sets (side surface + caps) and two center vertices.
          - The somewhat unusual direction construction using abs(tan(angle)) is preserved.
        """
        self.meshes = []
        if not self.layers:
            self._mesh_generated = False
            return self.meshes

        slice_angle = 360.0 / self.sector_amount

        # C++: const int totalAngleStep = 360.0f / m_sectorAmount; (integer)
        # This becomes exactly slice_angle when 360 divisible by sector_amount.
        # We mimic the C++ integer truncation.
        total_angle_step = int(360.0 / self.sector_amount)
        if total_angle_step < 2:
            # prevent division by zero in stepAngle; still produce a degenerate mesh
            total_angle_step = 2

        total_level_step = 2

        step_angle = slice_angle / (total_angle_step - 1)
        height_level = (
            self.max_height / self.layer_amount if self.layer_amount > 0 else 0.0
        )
        step_level = (
            height_level / (total_level_step - 1) if total_level_step > 1 else 0.0
        )

        # vertex layout (exactly as C++):
        # vertices.resize(totalLevelStep * m_sectorAmount * totalAngleStep * 2 + totalLevelStep);
        ring_count = total_level_step * self.sector_amount * total_angle_step
        vertex_count = ring_count * 2 + total_level_step
        # indices.resize((12 * (totalLevelStep - 1) * totalAngleStep) * m_sectorAmount);
        index_count = (
            12 * (total_level_step - 1) * total_angle_step
        ) * self.sector_amount

        for tier_index in range(self.layer_amount):
            vertices: List[Vertex] = [
                Vertex((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0))
                for _ in range(vertex_count)
            ]
            indices: List[int] = [0 for _ in range(index_count)]

            # Build vertices
            for level_step in range(total_level_step):
                current_height = height_level * tier_index + step_level * level_step

                for slice_index in range(self.sector_amount):
                    rmax = self.layers[tier_index][slice_index].max_distance

                    for angle_step in range(total_angle_step):
                        actual_angle_step = slice_index * total_angle_step + angle_step

                        current_angle = (
                            slice_angle * slice_index + step_angle * angle_step
                        )
                        if current_angle >= 360.0:
                            current_angle = 0.0

                        # C++ logic:
                        # float x = abs(tan(radians(angle)));
                        # float z = 1;
                        # then quadrant-based signs
                        x = abs(math.tan(math.radians(current_angle)))
                        z = 1.0

                        if 0.0 <= current_angle <= 90.0:
                            z *= -1.0
                            x *= -1.0
                        elif 90.0 < current_angle <= 180.0:
                            x *= -1.0
                        elif 270.0 < current_angle <= 360.0:
                            z *= -1.0

                        dir_xz = self._normalize((x, 0.0, z))
                        pos = (dir_xz[0] * rmax, current_height, dir_xz[2] * rmax)

                        # side surface vertex
                        side_idx = (
                            level_step * total_angle_step * self.sector_amount
                            + actual_angle_step
                        )
                        uv = (
                            float(level_step) / (total_level_step - 1),
                            float(angle_step) / (total_angle_step - 1),
                        )
                        vertices[side_idx] = Vertex(
                            position=pos,
                            tex_coord=uv,
                            normal=self._normalize(pos),
                        )

                        # cap vertex (duplicated position, different normal)
                        cap_base = ring_count
                        cap_idx = (
                            cap_base
                            + level_step * total_angle_step * self.sector_amount
                            + actual_angle_step
                        )
                        cap_normal = (0.0, -1.0 if level_step == 0 else 1.0, 0.0)
                        vertices[cap_idx] = Vertex(
                            position=pos,
                            tex_coord=uv,
                            normal=cap_normal,
                        )

                # center vertices (2 of them: bottom and top)
                center_idx = vertex_count - total_level_step + level_step
                center_normal = (0.0, -1.0 if level_step == 0 else 1.0, 0.0)
                vertices[center_idx] = Vertex(
                    position=(0.0, current_height, 0.0),
                    tex_coord=(0.0, 0.0),
                    normal=center_normal,
                )

            # Build indices (ported verbatim)
            for level_step in range(total_level_step - 1):
                for slice_index in range(self.sector_amount):
                    for angle_step in range(total_angle_step):
                        actual_angle_step = slice_index * total_angle_step + angle_step
                        base = 12 * (
                            level_step * total_angle_step * self.sector_amount
                            + actual_angle_step
                        )

                        # side rings (first half)
                        side0 = (
                            level_step * total_angle_step * self.sector_amount
                            + actual_angle_step
                        )
                        side1 = (
                            (level_step + 1) * total_angle_step * self.sector_amount
                            + actual_angle_step
                        )

                        cap_base = ring_count
                        cap0 = (
                            cap_base
                            + level_step * total_angle_step * self.sector_amount
                            + actual_angle_step
                        )
                        cap1 = (
                            cap_base
                            + (level_step + 1) * total_angle_step * self.sector_amount
                            + actual_angle_step
                        )

                        center0 = vertex_count - total_level_step + level_step
                        center1 = vertex_count - total_level_step + (level_step + 1)

                        if (
                            actual_angle_step
                            < self.sector_amount * total_angle_step - 1
                        ):
                            # next within unwrapped ring
                            indices[base + 0] = side0
                            indices[base + 1] = side0 + 1
                            indices[base + 2] = side1

                            indices[base + 3] = side1 + 1
                            indices[base + 4] = side1
                            indices[base + 5] = side0 + 1

                            indices[base + 6] = cap0
                            indices[base + 7] = center0
                            indices[base + 8] = cap0 + 1

                            indices[base + 9] = cap1 + 1
                            indices[base + 10] = center1
                            indices[base + 11] = cap1
                        else:
                            # wrap to beginning of ring
                            indices[base + 0] = side0
                            indices[base + 1] = (
                                level_step * total_angle_step * self.sector_amount
                            )
                            indices[base + 2] = side1

                            indices[base + 3] = (
                                (level_step + 1) * total_angle_step * self.sector_amount
                            )
                            indices[base + 4] = side1
                            indices[base + 5] = (
                                level_step * total_angle_step * self.sector_amount
                            )

                            indices[base + 6] = cap0
                            indices[base + 7] = center0
                            indices[base + 8] = (
                                cap_base
                                + level_step * total_angle_step * self.sector_amount
                            )

                            indices[base + 9] = (
                                cap_base
                                + (level_step + 1)
                                * total_angle_step
                                * self.sector_amount
                            )
                            indices[base + 10] = center1
                            indices[base + 11] = cap1

            self.meshes.append(MeshData(vertices=vertices, indices=indices))

        self._mesh_generated = True
        return self.meshes

    # -------------------------
    # Sampling / queries (unchanged)
    # -------------------------

    def get_random_point(self) -> Vec3:
        if not self._mesh_generated or self._total_weight <= 0.0:
            return (0.0, 0.0, 0.0)

        size_point = random.uniform(0.0, self._total_weight)

        layer_idx = self.layer_amount - 1
        sector_idx = self.sector_amount - 1
        found = False

        for i in range(self.layer_amount):
            lw = self._layer_weights[i]
            if size_point > lw:
                size_point -= lw
                continue
            for j in range(self.sector_amount):
                sw = self._sector_weights[i][j]
                if size_point > sw:
                    size_point -= sw
                else:
                    layer_idx, sector_idx = i, j
                    found = True
                    break
            break

        if not found:
            layer_idx = self.layer_amount - 1
            sector_idx = self.sector_amount - 1

        height_level = self._height_level()
        slice_angle = self._slice_angle_deg()

        h = height_level * layer_idx + random.uniform(0.0, height_level)
        ang_deg = slice_angle * sector_idx + random.uniform(0.0, slice_angle)

        rmax = self.layers[layer_idx][sector_idx].max_distance
        r = rmax * math.sqrt(random.random())  # disk-uniform

        x = r * math.sin(math.radians(ang_deg))
        z = r * math.cos(math.radians(ang_deg))
        return (x, h, z)

    def in_volume(self, position: Union[Vec3, Any]) -> bool:
        x, y, z = self._as_vec3(position)
        if any(math.isnan(v) for v in (x, y, z)):
            return True
        if not self._mesh_generated:
            return True
        layer_idx, sector_idx = self.select_slice((x, y, z))
        r = math.hypot(x, z)
        allowed = max(1.0, self.layers[layer_idx][sector_idx].max_distance)
        return (allowed >= r) and (y <= self.max_height)

    def export_as_obj(self, filepath: str) -> None:
        """
        Port of C++ ExportAsObj(const std::string& filename).

        Behavior:
        - Ensures meshes are generated (calls generate_mesh()).
        - Writes a single OBJ with:
            o RBV
            v ... for all vertices across all layer meshes
            f ... for all triangles across all layer meshes
        - Indices are written 1-based (OBJ convention), with a running vertex offset per mesh.

        Notes:
        - Matches the C++ exporter: positions only, no normals/uvs in the OBJ.
        - Faces are emitted in the same winding as stored in `indices`.
        """
        # Ensure we have up-to-date mesh data
        self.generate_mesh()
        meshes = self.meshes

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("o RBV\n")

            # Write vertices for all meshes
            for mesh in meshes:
                for v in mesh.vertices:
                    x, y, z = v.position
                    f.write(f"v {x} {y} {z}\n")

            # Write faces, applying 1-based indexing and per-mesh offsets
            offset = 1  # OBJ is 1-based
            for mesh in meshes:
                idx = mesh.indices
                if len(idx) % 3 != 0:
                    raise ValueError(
                        "Mesh indices length must be a multiple of 3 (triangles)."
                    )

                for i in range(0, len(idx), 3):
                    a = idx[i] + offset
                    b = idx[i + 1] + offset
                    c = idx[i + 2] + offset
                    f.write(f"f {a} {b} {c}\n")

                offset += len(mesh.vertices)

    import csv

    def export_rbv_to_csv(self, filepath: str, *, float_format: str = ".6f") -> None:
        """
        Export RBV to CSV with layer top heights.

        Format:
        Row 0:
            ["layer_top_height", angle_0, angle_1, ..., angle_{S-1}]
        Row i+1:
            [layer_top_height_i, d_i0, d_i1, ..., d_i{S-1}]

        where:
        - angle_j = j * (360 / sector_amount)
        - layer_top_height_i = (i + 1) * (max_height / layer_amount)
        """
        if not self.layers or self.layer_amount <= 0 or self.sector_amount <= 0:
            raise ValueError("RBV is not initialized.")

        slice_angle = 360.0 / self.sector_amount
        height_step = self.max_height / self.layer_amount

        header = ["layer_top_height"] + [
            format(j * slice_angle, float_format) for j in range(self.sector_amount)
        ]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for layer_idx in range(self.layer_amount):
                layer_top = (layer_idx + 1) * height_step
                row = [format(layer_top, float_format)]
                for sector_idx in range(self.sector_amount):
                    row.append(
                        format(
                            self.layers[layer_idx][sector_idx].max_distance,
                            float_format,
                        )
                    )
                writer.writerow(row)

    def import_from_csv(self, filepath: str) -> None:
        """
        Reconstruct (overwrite) this RBV from a CSV exported by export_rbv_to_csv().

        Restores:
          - layer_amount
          - sector_amount
          - max_height (exact, taken from last layer_top_height)
          - max_radius (max of all distances)
          - layers[*][*].max_distance
          - meshes + sampling weights (generate_mesh + _calculate_sizes)

        Notes:
          - This method overwrites the RBV's existing dimensions and contents.
          - Keeps self.offset unchanged (CSV format does not store it). If you want offset stored too,
            add a metadata row/comment line and I can extend the parser.
        """
        with open(filepath, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))

        if len(rows) < 2:
            raise ValueError("CSV must contain at least a header row and one data row.")

        header = rows[0]
        if not header or header[0].strip() != "layer_top_height":
            raise ValueError("CSV header must start with 'layer_top_height'.")

        sector_amount = len(header) - 1
        if sector_amount <= 0:
            raise ValueError("CSV must contain at least one sector column.")

        data_rows = rows[1:]
        layer_amount = len(data_rows)

        # Overwrite dimensions
        self.layer_amount = layer_amount
        self.sector_amount = sector_amount

        # Re-init layers with new dimensions
        self._resize_volumes()

        layer_tops: List[float] = []
        max_radius = 0.0

        for layer_idx, row in enumerate(data_rows):
            if len(row) != 1 + sector_amount:
                raise ValueError(
                    f"Row {layer_idx + 2} has {len(row)} columns; expected {1 + sector_amount}."
                )

            try:
                layer_top = float(row[0])
            except Exception as e:
                raise ValueError(
                    f"Invalid layer_top_height at CSV row {layer_idx + 2}: {row[0]}"
                ) from e

            layer_tops.append(layer_top)

            for sector_idx in range(sector_amount):
                cell = row[1 + sector_idx]
                try:
                    d = float(cell)
                except Exception as e:
                    raise ValueError(
                        f"Invalid max_distance at layer {layer_idx}, sector {sector_idx}: {cell}"
                    ) from e

                self.layers[layer_idx][sector_idx].max_distance = d
                if d > max_radius:
                    max_radius = d

        # Recover exact max_height from last layer top.
        # (Assumes CSV rows are ordered from bottom layer to top layer.)
        self.max_height = layer_tops[-1]
        self.max_radius = max_radius

        self.generate_mesh()
        self._calculate_sizes()


def generate_pine_tree_points(
    n_points: int,
    *,
    trunk_height: float = 2.0,
    trunk_radius: float = 0.08,
    crown_height: float = 6.0,
    crown_radius: float = 2.0,
    crown_base_height: Optional[float] = None,
    trunk_fraction: float = 0.25,
    jitter: float = 0.0,
    seed: Optional[int] = None,
    as_numpy: bool = False,
    crown_mode: Literal["volume", "surface"] = "surface",
    surface_thickness: float = 0.08,
    surface_profile: Literal["shell", "gaussian"] = "gaussian",
) -> List[Vec3]:
    """
    Pine-like synthetic points:
      - Trunk: thin cylinder (volume-uniform)
      - Crown: cone, either volume-uniform ("volume") or surface-biased ("surface")

    Surface-biased crown:
      - For each crown point, we choose a height, compute the cone's local radius R(y),
        then sample a radius r that lies close to R(y).
      - Two profiles:
          * "shell": uniform within [R*(1-thickness), R]
          * "gaussian": r = clamp(R - |N(0, sigma)|, 0, R), where sigma=thickness*R (relative)

    Parameters (new)
    ----------------
    crown_mode:
        "volume"  -> uniform in cone volume (previous behavior).
        "surface" -> biased towards the lateral cone surface.
    surface_thickness:
        Controls how close points are to the cone surface.
        Interpreted as a fraction of local radius in "shell" mode, and as sigma fraction in "gaussian".
        Reasonable range: 0.02 .. 0.20
    surface_profile:
        "shell" or "gaussian" distribution for the radial offset from the surface.

    Returns list of (x,y,z) tuples, or list of np.ndarray shape (3,) float32 if as_numpy=True.
    """
    if n_points <= 0:
        return []
    if trunk_height <= 0 or trunk_radius <= 0 or crown_height <= 0 or crown_radius <= 0:
        raise ValueError("All dimensions must be positive.")
    if not (0.0 <= trunk_fraction <= 1.0):
        raise ValueError("trunk_fraction must be in [0, 1].")
    if crown_mode not in ("volume", "surface"):
        raise ValueError("crown_mode must be 'volume' or 'surface'.")
    if surface_profile not in ("shell", "gaussian"):
        raise ValueError("surface_profile must be 'shell' or 'gaussian'.")
    if surface_thickness <= 0.0:
        raise ValueError("surface_thickness must be > 0.")
    if as_numpy and np is None:
        raise RuntimeError("as_numpy=True requires numpy to be installed.")

    rng = random.Random(seed)
    crown_base = trunk_height if crown_base_height is None else float(crown_base_height)

    n_trunk = int(round(n_points * trunk_fraction))
    n_crown = n_points - n_trunk

    out: List[Vec3] = []

    # ----------------
    # Trunk: uniform volume in cylinder
    # ----------------
    for _ in range(n_trunk):
        y = rng.uniform(0.0, trunk_height)
        r = trunk_radius * math.sqrt(rng.random())
        theta = rng.uniform(0.0, 2.0 * math.pi)
        x = r * math.cos(theta)
        z = r * math.sin(theta)
        out.append((x, y, z))

    # ----------------
    # Crown
    # ----------------
    for _ in range(n_crown):
        # Choose a height distribution.
        # For a visually pine-like crown, we still want more mass lower in the crown.
        # Using t ~ U^(1/3) (volume-correct) tends to prefer larger cross-sections (lower region),
        # which generally looks good even for surface-biased sampling.
        t = rng.random() ** (1.0 / 3.0)
        h = crown_height * t
        y = crown_base + h

        # Local max radius on cone at this height.
        R = crown_radius * (1.0 - (h / crown_height))
        if R <= 1e-12:
            # Very near apex; put it essentially on-axis.
            out.append((0.0, y, 0.0))
            continue

        theta = rng.uniform(0.0, 2.0 * math.pi)

        if crown_mode == "volume":
            # Uniform in disk area => r = R*sqrt(U)
            r = R * math.sqrt(rng.random())

        else:
            # Surface-biased (close to R)
            if surface_profile == "shell":
                # Uniform shell near the boundary:
                # r in [R*(1 - thickness), R]
                inner = max(0.0, 1.0 - surface_thickness)
                r = R * (inner + (1.0 - inner) * rng.random())

            else:
                # Gaussian falloff from surface:
                # r = clamp(R - |N(0, sigma)|, 0, R)
                # sigma is relative to local radius
                sigma = surface_thickness * R
                # rng.gauss(mu, sigma) is available in random.Random
                dr = abs(rng.gauss(0.0, sigma))
                r = max(0.0, R - dr)

        x = r * math.cos(theta)
        z = r * math.sin(theta)
        out.append((x, y, z))

    # Optional jitter
    if jitter > 0.0:
        j = float(jitter)
        out = [
            (x + rng.uniform(-j, j), y + rng.uniform(-j, j), z + rng.uniform(-j, j))
            for (x, y, z) in out
        ]

    if as_numpy:
        return [np.array(p, dtype=np.float32) for p in out]  # type: ignore[return-value]
    return out


# -------------------------
# Example
# -------------------------
if __name__ == "__main__":
    rbv = RadialBoundingVolume(layer_amount=8, sector_amount=12, offset=0.05)

    pts = generate_pine_tree_points(
        50_000,
        trunk_height=2.0,
        trunk_radius=0.06,
        crown_height=6.0,
        crown_radius=2.2,
        trunk_fraction=0.15,
        crown_mode="surface",
        surface_profile="gaussian",
        surface_thickness=0.06,  # smaller -> closer to surface
        jitter=0.01,
        seed=7,
    )

    # NumPy input also works (if numpy is installed)
    if np is not None:
        pts.append(np.array([0.4, 1.0, -0.2], dtype=np.float32))

    rbv.calculate_volume(pts)
    meshes = rbv.generate_mesh()
    print("Num layer meshes:", len(meshes))
    print(
        "First mesh: vertices =",
        len(meshes[0].vertices),
        "indices =",
        len(meshes[0].indices),
    )
    rbv.export_as_obj("pine_tree_rbv.obj")
    rbv.export_rbv_to_csv("rbv_grid.csv")
    rbv.import_from_csv("rbv_grid.csv")
    rbv.export_as_obj("imported_rbv.obj")