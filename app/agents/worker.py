from typing import Dict, Any
from app.db.mock_db import global_state

class WorkerAgent:
	"""Represents a simulated team member. Lightweight wrapper around global worker state."""
	def __init__(self, worker_id: str):
		self.worker_id = worker_id

	def _worker(self) -> Dict[str, Any]:
		return next((w for w in global_state.workers if w.id == self.worker_id), None)

	def profile(self) -> Dict[str, Any]:
		w = self._worker()
		if not w:
			return {}
		return {
			"id": w.id,
			"name": w.name,
			"skill_level": w.skill_level,
			"efficiency": w.efficiency,
			"fatigue": w.fatigue,
			"current_task_id": w.current_task_id,
		}

	def is_busy(self) -> bool:
		w = self._worker()
		return bool(w and w.current_task_id)
