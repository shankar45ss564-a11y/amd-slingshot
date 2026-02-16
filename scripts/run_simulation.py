"""Run a simple simulation loop using PlannerAgent, ContextAgent and RandomPolicy.

This runner is intentionally minimal: it creates the services and agents,
then loops until all tasks are completed or failed, or a max step limit is reached.
"""
import argparse
import logging
import os
import sys
from typing import Dict, Any

# Ensure repo root is on sys.path when running the script directly
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from app.services.simulation_service import SimulationService
    from app.services.task_service import TaskService
    from app.agents.context import ContextAgent
    from app.agents.planner import PlannerAgent
    from app.rl.policy import RandomPolicy
except ModuleNotFoundError as e:
    print(f"Missing dependency while importing modules: {e}")
    print("Install project dependencies, for example:")
    print("  python3 -m pip install -r requirements.txt")
    print("or at minimum:")
    print("  python3 -m pip install pydantic")
    raise


def run(max_steps: int = 200, step_hours: float = 1.0, verbose: bool = True, policy_name: str = "random") -> Dict[str, Any]:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    logger = logging.getLogger("run_simulation")

    simulation = SimulationService()
    task_service = TaskService()
    context = ContextAgent(simulation, task_service)
    # Choose policy based on policy_name parameter
    if policy_name == "random":
        policy = RandomPolicy()
    elif policy_name == "deadline":
        from app.rl.policy import DeadlineAwarePolicy
        policy = DeadlineAwarePolicy()
    else:
        policy = RandomPolicy()
    planner = PlannerAgent(policy, task_service, simulation, context)

    step = 0
    done = False

    logger.info("Starting simulation run: max_steps=%s step_hours=%s", max_steps, step_hours)

    while not done and step < max_steps:
        step += 1
        obs = planner.observe()
        action = planner.decide()

        logger.debug("Step %d action: %s", step, action)

        result = planner.act(action)

        # If action was assign, advance time a bit to allow work to progress
        if action.get("type") == "assign":
            if isinstance(result, dict) and result.get("error"):
                logger.info("Assignment error: %s", result.get("error"))
            sim_result = simulation.step(step_hours)
            for ev in sim_result.get("events", []):
                context.record_event({"step": step, "event": ev})
                logger.info(ev)
        elif action.get("type") == "wait":
            # planner.act already invoked simulation.step via PlannerAgent
            sim_result = result
            if isinstance(sim_result, dict):
                for ev in sim_result.get("events", []):
                    context.record_event({"step": step, "event": ev})
                    logger.info(ev)

        # Check termination: no active TODO/IN_PROGRESS tasks
        state = simulation.get_state()
        active = [t for t in state.get("tasks", []) if t.get("status") in ["todo", "in_progress"]]
        if not active:
            done = True
            logger.info("All tasks finished or failed at simulation time %s after %d steps.", state.get("current_time"), step)

    # Summary
    final_state = simulation.get_state()
    completed = [t for t in final_state.get("tasks", []) if t.get("status") == "completed"]
    failed = [t for t in final_state.get("tasks", []) if t.get("status") == "failed"]

    summary = {
        "steps": step,
        "sim_time": final_state.get("current_time"),
        "completed_count": len(completed),
        "failed_count": len(failed),
        "events": context.get_history(),
    }

    logger.info("Run summary: %s", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run the Planner+Context simulation loop")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--step-hours", type=float, default=1.0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--policy", type=str, default="random", choices=["random", "deadline"],
                        help="Policy to use: random or deadline")
    args = parser.parse_args()

    # Pass policy name into run via a simple override
    # (we keep backward compatibility)
    run(max_steps=args.max_steps, step_hours=args.step_hours, verbose=not args.quiet, policy_name=args.policy)


if __name__ == "__main__":
    main()
