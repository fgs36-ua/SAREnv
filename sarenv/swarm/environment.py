# sarenv/swarm/environment.py
"""
Wrapper del grid sobre SARDatasetItem para el simulador de enjambre.

Expone el mapa de probabilidad y funciones de conversión de coordenadas
mundo <-> grid que necesitan los agentes. En Fase 1 el terreno es uniforme
(mismo coste de movimiento en todas las celdas). En Fase 2 se añadirán
detection_modifier_map y traversability_map para terreno heterogéneo.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sarenv.core.loading import SARDatasetItem


@dataclass(frozen=True)
class GridInfo:
    """Geometría precomputada del grid para conversiones rápidas mundo <-> grid."""

    rows: int
    cols: int
    minx: float
    miny: float
    maxx: float
    maxy: float
    dx: float  # ancho de celda en metros
    dy: float  # alto de celda en metros


class SwarmEnvironment:
    """Wrapper sobre SARDatasetItem que da acceso grid al mapa de probabilidad.

    Convierte el dataset en una representación discreta con la que los agentes
    del enjambre pueden operar directamente (coordenadas grid, vecinos, costes).
    """

    def __init__(self, dataset_item: SARDatasetItem) -> None:
        self.dataset_item = dataset_item
        self.probability_map: np.ndarray = dataset_item.heatmap.astype(np.float32)
        self.bounds: tuple[float, float, float, float] = dataset_item.bounds

        rows, cols = self.probability_map.shape
        minx, miny, maxx, maxy = self.bounds
        dx = (maxx - minx) / cols
        dy = (maxy - miny) / rows

        self.grid = GridInfo(
            rows=rows, cols=cols,
            minx=minx, miny=miny, maxx=maxx, maxy=maxy,
            dx=dx, dy=dy,
        )

        # Centro del entorno en coordenadas mundo (para despliegue inicial)
        self.center_x: float = (minx + maxx) / 2.0
        self.center_y: float = (miny + maxy) / 2.0

    # -- Conversión de coordenadas --

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        """Coordenadas mundo (x, y) -> índices grid (row, col)."""
        col = int((x - self.grid.minx) / self.grid.dx)
        row = int((y - self.grid.miny) / self.grid.dy)
        col = max(0, min(col, self.grid.cols - 1))
        row = max(0, min(row, self.grid.rows - 1))
        return row, col

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        """Indices grid (row, col) -> coordenadas mundo del centro de la celda."""
        x = self.grid.minx + (col + 0.5) * self.grid.dx
        y = self.grid.miny + (row + 0.5) * self.grid.dy
        return x, y

    def in_bounds(self, row: int, col: int) -> bool:
        """True si la celda está dentro de los límites del mapa."""
        return 0 <= row < self.grid.rows and 0 <= col < self.grid.cols

    # -- Detección (Fase 1: circular uniforme, como el greedy de SAREnv) --

    def get_visible_cells(
        self,
        row: int,
        col: int,
        detection_radius: float,
    ) -> set[tuple[int, int]]:
        """Celdas visibles desde (row, col) con footprint circular.

        Optimización: calcula distancias en espacio grid escalado en vez de
        convertir cada celda a coordenadas mundo (evita N² llamadas a grid_to_world).
        """
        # Radio en número de celdas por cada eje
        radius_cells_x = int(np.ceil(detection_radius / self.grid.dx))
        radius_cells_y = int(np.ceil(detection_radius / self.grid.dy))
        r2 = detection_radius * detection_radius

        visible: set[tuple[int, int]] = set()

        r_min = max(0, row - radius_cells_y)
        r_max = min(self.grid.rows, row + radius_cells_y + 1)
        c_min = max(0, col - radius_cells_x)
        c_max = min(self.grid.cols, col + radius_cells_x + 1)

        for r in range(r_min, r_max):
            dy_m = (r - row) * self.grid.dy
            dy2 = dy_m * dy_m
            # Si solo la componente Y ya excede el radio, saltar fila
            if dy2 > r2:
                continue
            for c in range(c_min, c_max):
                dx_m = (c - col) * self.grid.dx
                if dy2 + dx_m * dx_m <= r2:
                    visible.add((r, c))

        return visible

    # -- Movimiento (Fase 1: coste uniforme, 8-conectado) --

    # Offsets de los 8 vecinos
    NEIGHBOR_OFFSETS = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    def get_reachable_neighbors(
        self, row: int, col: int, _agent_type: str = "drone"
    ) -> list[tuple[int, int]]:
        """Vecinos alcanzables desde (row, col) en un tick.

        Fase 1: todos los 8 vecinos si están dentro del mapa.
        Fase 2: usará traversability_map según tipo de agente.
        """
        neighbors: list[tuple[int, int]] = []
        for dr, dc in self.NEIGHBOR_OFFSETS:
            nr, nc = row + dr, col + dc
            if self.in_bounds(nr, nc):
                neighbors.append((nr, nc))
        return neighbors

    def movement_cost(
        self, from_rc: tuple[int, int], to_rc: tuple[int, int], _agent_type: str = "drone"
    ) -> float:
        """Coste en metros de moverse entre dos celdas adyacentes.

        Fase 1: distancia euclídea pura (terreno uniforme).
        Fase 2: multiplicará por coste del terreno.
        """
        dr = to_rc[0] - from_rc[0]
        dc = to_rc[1] - from_rc[1]
        return np.sqrt((dr * self.grid.dy) ** 2 + (dc * self.grid.dx) ** 2)
