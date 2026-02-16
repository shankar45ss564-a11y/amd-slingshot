from typing import Any, Dict, List
from app.services.simulation_service import SimulationService
from app.services.task_service import TaskService

class ContextAgent:
	"""Maintains shared project state and historical performance."""
	def __init__(self, simulation: SimulationService, task_service: TaskService):
		self.simulation = simulation
		self.task_service = task_service
		self.history: List[Dict[str, Any]] = []

	def get_state(self) -> Dict[str, Any]:
		return self.simulation.get_state()

	def get_backlog(self) -> List[Dict[str, Any]]:
		state = self.get_state()
		return [t for t in state.get("tasks", []) if t.get("status") == "TODO"]

	def record_event(self, event: Dict[str, Any]) -> None:
		self.history.append(event)

	def get_history(self) -> List[Dict[str, Any]]:
		return list(self.history)
