from typing import Dict, Any, Optional
from app.services.task_service import TaskService
from app.services.simulation_service import SimulationService
from app.agents.context import ContextAgent

class PlannerAgent:
	"""Manager agent that decides who gets which task using a policy."""
	def __init__(self, policy, task_service: TaskService, simulation: SimulationService, context: ContextAgent):
		self.policy = policy
		self.task_service = task_service
		self.simulation = simulation
		self.context = context

	def observe(self) -> Dict[str, Any]:
		return self.context.get_state()

	def decide(self) -> Dict[str, Any]:
		obs = self.observe()
		action = self.policy.select_action(obs)
		return action

	def act(self, action: Dict[str, Any]) -> Dict[str, Any]:
		if action.get("type") == "assign":
			return self.task_service.assign_task(action.get("worker_id"), action.get("task_id"), automated=True)
		elif action.get("type") == "wait":
			hours = action.get("hours", 1.0)
			return self.simulation.step(hours)
		else:
			return {"error": "unknown_action"}
