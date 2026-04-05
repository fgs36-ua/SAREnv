# sarenv/swarm/agents.py
"""
Jerarquía de agentes del enjambre.

Fase 1: BaseSwarmAgent con heurística greedy adaptada al paradigma de
feromonas.  La regla de decisión es sencilla a propósito, porque el
comportamiento coordinado emerge de la dinámica de feromonas y gossip.

    score = probability * (1 - exploration_pheromone) - repulsion

Fase 2 añadirá RobotDogAgent con detección/movimiento dependiente del terreno.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import AgentConfig, DroneConfig, RobotDogConfig
    from .environment import SwarmEnvironment
    from .knowledge import LocalKnowledgeMap


@dataclass
class Perception:
    """Snapshot de lo que un agente percibe en un tick."""

    visible_cells: set[tuple[int, int]]
    local_probability: dict[tuple[int, int], float]
    local_exploration: dict[tuple[int, int], float]
    local_alert: dict[tuple[int, int], float]
    neighbors: list[BaseSwarmAgent]
    budget_remaining: float
    position: tuple[int, int]


class BaseSwarmAgent:
    """Agente genérico del enjambre que opera sobre un grid discreto.

    Cadena de prioridades en decide():
      1. Volver a base si queda poco budget
      2. Investigar feromona de alerta
      3. Greedy: prob * novedad - repulsión
      4. Paseo aleatorio como fallback
    """

    agent_type: str = "drone"

    def __init__(
        self,
        agent_id: str,
        config: AgentConfig,
        environment: SwarmEnvironment,
        knowledge: LocalKnowledgeMap,
        start_position: tuple[int, int],
        rng: np.random.Generator | None = None,
    ) -> None:
        self.id = agent_id
        self.config = config
        self.env = environment
        self.knowledge = knowledge

        self.position: tuple[int, int] = start_position
        self.base_position: tuple[int, int] = start_position  # for return-to-base
        self.budget_remaining: float = config.budget
        self.path: list[tuple[int, int]] = [start_position]
        self.active: bool = True  # False cuando se agota budget o vuelve a base

        self._rng = rng or np.random.default_rng()

        # Radio de detección cacheado desde config
        self._detection_radius: float = self._compute_detection_radius()

    # -- Hooks que las subclases pueden sobreescribir --

    def _compute_detection_radius(self) -> float:
        """Radio de detección en metros (sobreescrito por subclases)."""
        return getattr(self.config, "detection_radius", 80.0)

    def _get_visible_cells(self) -> set[tuple[int, int]]:
        """Cells visible from the current position."""
        return self.env.get_visible_cells(
            self.position[0], self.position[1], self._detection_radius
        )

    def _get_reachable_neighbors(self) -> list[tuple[int, int]]:
        """Grid cells the agent can move to in one tick."""
        return self.env.get_reachable_neighbors(
            self.position[0], self.position[1], self.agent_type
        )

    def _movement_cost(self, target: tuple[int, int]) -> float:
        """Cost in metres of moving from current position to *target*."""
        return self.env.movement_cost(self.position, target, self.agent_type)

    # ── perception ────────────────────────────────────────────────────

    def perceive(self, nearby_agents: list[BaseSwarmAgent]) -> Perception:
        """Build a local perception snapshot."""
        visible = self._get_visible_cells()
        return Perception(
            visible_cells=visible,
            local_probability={
                c: self.knowledge.probability_map[c[0], c[1]] for c in visible
            },
            local_exploration={
                c: self.knowledge.exploration_map[c[0], c[1]] for c in visible
            },
            local_alert={
                c: self.knowledge.alert_map[c[0], c[1]]
                for c in visible
                if self.knowledge.alert_map[c[0], c[1]] > 0
            },
            neighbors=nearby_agents,
            budget_remaining=self.budget_remaining,
            position=self.position,
        )

    # -- Decisión --

    def decide(self, perception: Perception | None = None) -> tuple[int, int] | None:
        """Elige la siguiente celda. Cadena de 4 prioridades (ver docstring de clase)."""
        if not self.active:
            return None

        if perception is None:
            perception = self.perceive([])

        # Prioridad 1 -- conservar budget para volver a base
        dist_to_base = self._grid_distance(self.position, self.base_position)
        if perception.budget_remaining < dist_to_base * self.config.return_safety_factor:
            return self._step_toward(self.base_position)

        # Prioridad 2 -- investigar la alerta más fuerte cercana
        if perception.local_alert:
            best_alert_cell = max(perception.local_alert, key=perception.local_alert.get)
            if perception.local_alert[best_alert_cell] > self.config.alert_threshold:
                return self._step_toward(best_alert_cell)

        # Prioridad 3 -- exploración greedy con repulsión
        reachable = self._get_reachable_neighbors()
        if not reachable:
            return None

        best_score = -np.inf
        best_cell = None

        # Pre-calcular posiciones de vecinos para no recalcular cada vez
        neighbor_positions = [n.position for n in perception.neighbors]

        for cell in reachable:
            prob = self.knowledge.probability_map[cell[0], cell[1]]
            novelty = 1.0 - self.knowledge.exploration_map[cell[0], cell[1]]

            # Repulsión: penalizar celdas cerca de otros agentes
            repulsion = 0.0
            for npos in neighbor_positions:
                d = self._grid_distance(cell, npos)
                if d > 0:
                    repulsion += 1.0 / d

            score = prob * novelty - self.config.repulsion_weight * repulsion
            if score > best_score:
                best_score = score
                best_cell = cell

        if best_cell is not None and best_score > 0:
            return best_cell

        # Fallback -- paseo aleatorio
        return self._random_walk(reachable)

    # -- Ejecución --

    def execute_move(self, target: tuple[int, int] | None) -> None:
        """Mueve al agente a target consumiendo budget. Desactiva si se agota."""
        if target is None or not self.active:
            return
        cost = self._movement_cost(target)
        if cost > self.budget_remaining:
            self.active = False
            return
        self.position = target
        self.budget_remaining -= cost
        self.path.append(target)

    # -- Helpers --

    def _grid_distance(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        """Distancia euclídea en metros entre dos celdas."""
        dr = (a[0] - b[0]) * self.env.grid.dy
        dc = (a[1] - b[1]) * self.env.grid.dx
        return np.sqrt(dr * dr + dc * dc)

    def _step_toward(self, target: tuple[int, int]) -> tuple[int, int]:
        """Celda adyacente que más acerca al objetivo."""
        neighbors = self._get_reachable_neighbors()
        if not neighbors:
            return self.position
        return min(neighbors, key=lambda c: self._grid_distance(c, target))

    def _random_walk(self, reachable: list[tuple[int, int]]) -> tuple[int, int]:
        """Elige un vecino alcanzable al azar."""
        idx = self._rng.integers(len(reachable))
        return reachable[idx]

    # -- Conversión a LineString (compatibilidad con PathEvaluator) --

    def get_path_linestring(self):
        """Devuelve el path como Shapely LineString en coordenadas mundo."""
        from shapely.geometry import LineString

        if len(self.path) < 2:
            return LineString()
        coords = [self.env.grid_to_world(r, c) for r, c in self.path]
        return LineString(coords)


# -- Tipos concretos de agente --


class DroneAgent(BaseSwarmAgent):
    """Dron aéreo -- ve desde arriba, se mueve uniformemente sobre todo el terreno."""

    agent_type = "drone"

    def _compute_detection_radius(self) -> float:
        from .config import DroneConfig
        if isinstance(self.config, DroneConfig):
            return self.config.detection_radius
        # fallback genérico
        return 80.0 * np.tan(np.radians(45.0 / 2))


class RobotDogAgent(BaseSwarmAgent):
    """Robot terrestre -- en Fase 1 se comporta igual que un dron.

    En Fase 2 sobreescribirá _get_visible_cells, _get_reachable_neighbors
    y _movement_cost para tener en cuenta el tipo de terreno.
    """

    agent_type = "robot_dog"

    def _compute_detection_radius(self) -> float:
        from .config import RobotDogConfig
        if isinstance(self.config, RobotDogConfig):
            return self.config.detection_radius
        return 20.0
