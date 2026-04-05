# sarenv/swarm/metrics.py
"""
Adaptador que evalúa los resultados de la simulación de enjambre usando
el PathEvaluator de sarenv.analytics.metrics, más métricas específicas
del enjambre (cobertura, solapamiento, contribución por agente).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from shapely.geometry import LineString

if TYPE_CHECKING:
    import geopandas as gpd
    from .simulator import SwarmSimulator


class SwarmMetrics:
    """Evaluación de una simulación de enjambre completada.

    Envuelve PathEvaluator y añade métricas propias del enjambre:
    solapamiento de exploración, contribución por agente, etc.
    """

    def __init__(
        self,
        simulator: SwarmSimulator,
        victims: gpd.GeoDataFrame | None = None,
        discount_factor: float = 0.999,
    ) -> None:
        self.sim = simulator
        self.victims = victims
        self.discount_factor = discount_factor

    # -- Puente con PathEvaluator --

    def evaluate_with_path_evaluator(
        self,
        fov_deg: float | None = None,
        altitude: float | None = None,
        meters_per_bin: int | None = None,
    ) -> dict:
        """Ejecuta PathEvaluator.calculate_all_metrics sobre los paths del enjambre.

        Los parámetros que falten se derivan de la config del dron.
        """
        from sarenv.analytics.metrics import PathEvaluator

        env = self.sim.env
        cfg = self.sim.config.drone_config

        fov_deg = fov_deg or cfg.fov_deg
        altitude = altitude or cfg.altitude
        meters_per_bin = meters_per_bin or int(np.ceil(env.grid.dx))

        evaluator = PathEvaluator(
            heatmap=env.probability_map,
            extent=env.bounds,
            victims=self._ensure_victims_gdf(),
            fov_deg=fov_deg,
            altitude=altitude,
            meters_per_bin=meters_per_bin,
        )

        paths = self.sim.get_paths()
        return evaluator.calculate_all_metrics(paths, self.discount_factor)

    # -- Métricas específicas del enjambre --

    def coverage_summary(self) -> dict:
        """Resumen rápido de cobertura usando los mapas de conocimiento
        de los agentes (sin necesidad de PathEvaluator).
        """
        env = self.sim.env
        total_cells = env.grid.rows * env.grid.cols

        # Usar cells_ever_explored (acumulativo, inmune a evaporación)
        # en vez del exploration_map que decae cada tick.
        per_agent_explored: dict[str, int] = {}
        all_explored: set[tuple[int, int]] = set()

        for agent in self.sim.agents:
            per_agent_explored[agent.id] = len(agent.cells_ever_explored)
            all_explored.update(agent.cells_ever_explored)

        total_explored = len(all_explored)

        # Solapamiento: celdas exploradas por más de un agente
        sum_individual = sum(per_agent_explored.values())
        overlap = (sum_individual - total_explored) if sum_individual > 0 else 0

        # Masa de probabilidad cubierta (sobre el heatmap original, sin normalizar)
        prob_covered = 0.0
        for r, c in all_explored:
            prob_covered += float(env.probability_map[r, c])
        prob_total = float(np.sum(env.probability_map))

        return {
            "total_cells": total_cells,
            "explored_cells": total_explored,
            "coverage_ratio": total_explored / total_cells if total_cells else 0,
            "overlap_cells": overlap,
            "overlap_ratio": overlap / sum_individual if sum_individual else 0,
            "probability_covered": prob_covered,
            "probability_total": prob_total,
            "probability_coverage_ratio": (
                prob_covered / prob_total if prob_total > 0 else 0
            ),
            "per_agent_explored": per_agent_explored,
            "total_timesteps": self.sim.timestep,
            "paths_lengths_m": {
                a.id: a.get_path_linestring().length for a in self.sim.agents
            },
        }

    # -- Helpers privados --

    def _ensure_victims_gdf(self):
        """Devuelve un GeoDataFrame válido (posiblemente vacío) para PathEvaluator."""
        import geopandas as gpd
        if self.victims is None:
            return gpd.GeoDataFrame(geometry=[], crs=None)
        if isinstance(self.victims, gpd.GeoDataFrame):
            if not self.victims.empty:
                return self.victims
            return gpd.GeoDataFrame(geometry=[], crs=None)
        # Handle list of Points
        if isinstance(self.victims, list) and len(self.victims) > 0:
            return gpd.GeoDataFrame(geometry=self.victims, crs=None)
        return gpd.GeoDataFrame(geometry=[], crs=None)
