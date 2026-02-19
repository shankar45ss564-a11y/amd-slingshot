#!/usr/bin/env python3
"""
Demo script for AMD SlingShot Hackathon
Shows RL agent managing a project simulation with live updates and metrics
"""

import sys
import os
import time
import numpy as np
from typing import Dict, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from environment.project_env import ProjectEnv
from agents.dqn_agent import DQNAgent
from baselines.greedy_baseline import GreedyBaseline
from visualization.plot_demo import plot_simulation_metrics

def print_project_status(env: ProjectEnv, step: int, action_taken: str = "", reward: float = 0.0):
    """Print current project status in a readable format"""
    # Get current state info
    completed = len(env.completed_tasks)
    failed = len(env.failed_tasks)
    total_tasks = len(env.tasks) + completed + failed
    pending = len([t for t in env.tasks if t.status == 'todo'])
    in_progress = len([t for t in env.tasks if t.status == 'in_progress'])

    print(f"\n{'='*60}")
    print(f"STEP {step} - Time: {env.current_timestep:.1f}h")
    if action_taken:
        print(f"Action: {action_taken}")
    if reward != 0.0:
        print(f"Reward: {reward:+.2f}")

    print(f"Tasks: {pending} pending, {in_progress} in progress, {completed} completed, {failed} failed")

    print("Workers:")
    for worker in env.workers:
        fatigue_level = "Fresh" if worker.fatigue < 1 else "Tired" if worker.fatigue < 2 else "Exhausted" if worker.fatigue < 3 else "Burnout"
        task_info = f"working on Task {worker.current_task_id}" if worker.current_task_id else "idle"
        print(f"  Worker {worker.id}: {fatigue_level} ({worker.fatigue:.1f}), {task_info}")

def action_to_string(env: ProjectEnv, action: int) -> str:
    """Convert action index to human-readable string"""
    task_id, worker_id, action_type = env._decode_action(action)

    if action_type == 'assign':
        return f"Assign Task {task_id} to Worker {worker_id}"
    elif action_type == 'defer':
        return f"Defer Task {task_id}"
    elif action_type == 'escalate':
        return f"Escalate Task {task_id}"
    else:
        return f"Unknown action {action}"

def get_current_metrics(env: ProjectEnv) -> Dict:
    """Get current metrics for plotting"""
    # Compute current metrics
    metrics = env.compute_metrics()

    # Add additional real-time metrics
    active_tasks = len([t for t in env.tasks if t.status in ['todo', 'in_progress']])
    avg_fatigue = np.mean([w.fatigue for w in env.workers])
    worker_fatigue = [w.fatigue for w in env.workers]

    current_metrics = {
        'task_completion_rate': metrics['throughput'] / env.num_tasks if env.num_tasks > 0 else 0,
        'deadline_hit_rate': metrics['deadline_hit_rate'],
        'avg_worker_fatigue': avg_fatigue,
        'worker_fatigue': worker_fatigue,
        'active_tasks': active_tasks,
        'total_reward': env.episode_reward,
        'overload_events': metrics['overload_events'],
        'completed_tasks': len(env.completed_tasks),
        'failed_tasks': len(env.failed_tasks)
    }

    return current_metrics

def run_demo_episode(agent, env: ProjectEnv, max_steps: int = 50, delay: float = 1.0):
    """
    Run one episode of the simulation with live updates

    Args:
        agent: The agent to use (RL or baseline)
        env: The project environment
        max_steps: Maximum steps to run
        delay: Delay between steps for demo
    """
    print("🚀 Starting Project Management Demo")
    print("RL Agent will manage task allocation, deadlines, and worker fatigue")

    state = env.reset()
    total_reward = 0
    metrics_history = []

    for step in range(max_steps):
        # Get agent action
        if hasattr(agent, 'select_action'):
            # RL agent
            action = agent.select_action(state, training=False)
        else:
            # Baseline agent
            action = agent.decide(env)

        # Execute action
        next_state, reward, done, info = env.step(action)
        total_reward += reward

        # Record metrics
        metrics = get_current_metrics(env)
        metrics_history.append(metrics)

        # Print status
        action_desc = action_to_string(env, action)
        print_project_status(env, step, action_desc, reward)

        # Check if episode is done
        if done:
            print(f"\n✅ Episode completed! Total reward: {total_reward:.2f}")
            break

        state = next_state

        # Delay for demo effect
        if delay > 0:
            time.sleep(delay)

    return metrics_history

def main():
    print("AMD SlingShot Hackathon - RL Project Manager Demo")
    print("=" * 60)

    # Initialize environment
    env = ProjectEnv(seed=42, enable_diagnostics=True)

    # Choose agent
    use_rl = True  # Set to False to use baseline instead

    if use_rl:
        print("🤖 Using RL (DQN) Agent")
        agent = DQNAgent()

        # Try to load trained model, otherwise use untrained
        model_path = os.path.join(config.CHECKPOINT_DIR, "dqn_final.pth")
        if os.path.exists(model_path):
            print(f"📁 Loading trained model from {model_path}")
            agent.load(model_path)
        else:
            print("⚠️  No trained model found, using untrained agent")
            print("   (For better demo, train the agent first with training/train_dqn.py)")
    else:
        print("🎯 Using Greedy Baseline Agent")
        agent = GreedyBaseline()

    # Run demo episode
    metrics_history = run_demo_episode(agent, env, max_steps=30, delay=0.5)

    # Generate plots
    print("\n📊 Generating performance plots...")
    plot_simulation_metrics(metrics_history)

    # Print summary
    from visualization.plot_demo import create_demo_summary, print_demo_summary
    summary = create_demo_summary(metrics_history)
    print_demo_summary(summary)

    print("\n🎉 Demo completed!")
    print("Check the generated plots for performance analysis.")

if __name__ == "__main__":
    main()