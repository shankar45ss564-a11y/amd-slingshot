import random
from typing import Dict, Any

class RandomPolicy:
	"""Simple policy that picks a random valid assignment or waits."""
	def select_action(self, obs: Dict[str, Any]) -> Dict[str, Any]:
		workers = [w for w in obs.get("workers", []) if w.get("is_busy") == 0]
		tasks = obs.get("tasks", [])

		if not workers or not tasks:
			return {"type": "wait", "hours": 1.0}

		w = random.choice(workers)
		t = random.choice(tasks)
		return {"type": "assign", "worker_id": w["id"], "task_id": t["id"]}


class BasePolicy:
	def select_action(self, obs: Dict[str, Any]) -> Dict[str, Any]:
		raise NotImplementedError()


class DeadlineAwarePolicy(BasePolicy):
	"""Assigns the most urgent task (smallest deadline gap) to the most capable idle worker."""
	def select_action(self, obs: Dict[str, Any]) -> Dict[str, Any]:
		workers = [w for w in obs.get("workers", []) if w.get("is_busy") == 0]
		tasks = obs.get("tasks", [])

		if not workers or not tasks:
			return {"type": "wait", "hours": 1.0}

		# pick most urgent task by deadline_gap (smallest)
		tasks_sorted = sorted(tasks, key=lambda t: t.get("deadline_gap", float("inf")))
		task = tasks_sorted[0]

		# pick highest skill available worker
		worker = max(workers, key=lambda w: w.get("skill", 0.0))

		return {"type": "assign", "worker_id": worker["id"], "task_id": task["id"]}
