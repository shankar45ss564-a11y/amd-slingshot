#!/usr/bin/env python3
"""
Enhanced Demo Script with Additional Features
- Agent comparison
- Real-time metrics tracking
- Better error handling
- Improved visualization
"""

import sys
import os
import time
import numpy as np
from typing import Dict, List, Tuple
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from environment.project_env import ProjectEnv
from agents.dqn_agent import DQNAgent
from baselines.greedy_baseline import GreedyBaseline
from baselines.skill_baseline import SkillBaseline
from baselines.hybrid_baseline import HybridBaseline
from baselines.random_baseline import RandomBaseline
from baselines.improved_agents import AdaptiveAgent, PrioritizedGreedyAgent, LoadBalancingAgent
from visualization.plot_demo import plot_simulation_metrics, create_demo_summary, print_demo_summary


def print_project_status(env: ProjectEnv, step: int, action_taken: str = "", reward: float = 0.0):
    """Print current project status in a readable format"""
    completed = len(env.completed_tasks)
    failed = len(env.failed_tasks)
    total_tasks = len(env.tasks) + completed + failed
    pending = len([t for t in env.tasks if t.status == 'todo'])
    in_progress = len([t for t in env.tasks if t.status == 'in_progress'])

    print(f"\n{'='*70}")
    print(f"STEP {step:3d} | Time: {env.current_timestep:5.1f}h | "
          f"Tasks: {pending:2d}⏳ {in_progress:2d}⚙️  {completed:2d}✓ {failed:2d}✗")
    
    if action_taken:
        print(f"Action: {action_taken}")
    if reward != 0.0:
        print(f"Reward: {reward:+7.2f}")
    
    # Worker status bar
    for worker in env.workers:
        fatigue_level = ("Fresh" if worker.fatigue < 1 else "Tired" if worker.fatigue < 2 
                        else "Exhausted" if worker.fatigue < 3 else "💔Burnout")
        task_info = f"Task {worker.current_task_id}" if worker.current_task_id else "idle"
        load_bar = "█" * int(worker.load) + "░" * (5 - int(worker.load))
        print(f"  W{worker.id}: {load_bar} Fatigue:{worker.fatigue:.1f} ({fatigue_level}) {task_info}")


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
    metrics = env.compute_metrics()
    
    active_tasks = len([t for t in env.tasks if t.status in ['todo', 'in_progress']])
    avg_fatigue = np.mean([w.fatigue for w in env.workers]) if env.workers else 0
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


def run_demo_episode(agent, env: ProjectEnv, agent_name: str, max_steps: int = 50, 
                     delay: float = 0.2, verbose: bool = True) -> Tuple[List[Dict], float]:
    """
    Run one episode of the simulation with live updates
    
    Args:
        agent: The agent to use (RL or baseline)
        env: The project environment
        agent_name: Name of the agent for display
        max_steps: Maximum steps to run
        delay: Delay between steps for demo
        verbose: Whether to print status
    
    Returns:
        Tuple of (metrics_history, total_reward)
    """
    if verbose:
        print(f"\n🚀 Starting Demo with {agent_name}")
        print(f"Managing project with {env.num_workers} workers and {env.num_tasks} tasks")
        print("=" * 70)
    
    state = env.reset()
    total_reward = 0
    metrics_history = []
    
    for step in range(max_steps):
        try:
            # Get agent action
            if hasattr(agent, 'select_action'):
                if hasattr(agent, '__class__') and agent.__class__.__name__ == 'DQNAgent':
                    action = agent.select_action(state, training=False)
                else:
                    action = agent.select_action(state)
            else:
                # Fallback for agents without select_action
                action = np.random.randint(0, config.ACTION_DIM)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            
            # Record metrics
            metrics = get_current_metrics(env)
            metrics_history.append(metrics)
            
            # Print status
            if verbose and step % 5 == 0:  # Print every 5 steps
                action_desc = action_to_string(env, action)
                print_project_status(env, step, action_desc, reward)
            
            if done:
                if verbose:
                    print(f"\n✅ Episode completed at step {step}!")
                break
            
            state = next_state
            
            # Delay for demo effect
            if delay > 0 and verbose:
                time.sleep(delay)
        
        except Exception as e:
            print(f"❌ Error at step {step}: {str(e)}")
            break
    
    return metrics_history, total_reward


def run_agent_comparison(num_episodes: int = 3, max_steps_per_episode: int = 50):
    """
    Compare multiple agents on the same environment
    
    Args:
        num_episodes: Number of episodes per agent
        max_steps_per_episode: Max steps per episode
    """
    print("\n" + "="*70)
    print("AGENT COMPARISON BENCHMARK")
    print("="*70)
    
    agents_config = [
        ("DQN", lambda env: DQNAgent()),
        ("Greedy", lambda env: GreedyBaseline(env)),
        ("Skill", lambda env: SkillBaseline(env)),
        ("Hybrid", lambda env: HybridBaseline(env)),
        ("Random", lambda env: RandomBaseline(env)),
        ("Adaptive", lambda env: AdaptiveAgent(env)),
        ("PrioritizedGreedy", lambda env: PrioritizedGreedyAgent(env)),
        ("LoadBalancing", lambda env: LoadBalancingAgent(env)),
    ]
    
    results = {}
    
    for agent_name, agent_builder in agents_config:
        print(f"\n📊 Testing {agent_name}...")
        
        episode_rewards = []
        episode_completions = []
        
        for ep in range(num_episodes):
            env = ProjectEnv(seed=42 + ep)
            agent = agent_builder(env)
            
            try:
                metrics_history, total_reward = run_demo_episode(
                    agent, env, agent_name, 
                    max_steps=max_steps_per_episode,
                    delay=0,
                    verbose=False
                )
                
                episode_rewards.append(total_reward)
                episode_completions.append(len(env.completed_tasks))
            except Exception as e:
                print(f"  Episode {ep}: Error - {str(e)}")
                episode_rewards.append(0)
                episode_completions.append(0)
        
        avg_reward = np.mean(episode_rewards)
        avg_completions = np.mean(episode_completions)
        
        results[agent_name] = {
            'avg_reward': avg_reward,
            'avg_completions': avg_completions,
            'std_reward': np.std(episode_rewards)
        }
        
        print(f"  Avg Reward: {avg_reward:7.2f} | Avg Completions: {avg_completions:5.1f} | "
              f"Std: {np.std(episode_rewards):.2f}")
    
    # Print ranking
    print("\n" + "="*70)
    print("RANKING (by average reward)")
    print("="*70)
    
    ranked = sorted(results.items(), key=lambda x: x[1]['avg_reward'], reverse=True)
    for rank, (agent_name, metrics) in enumerate(ranked, 1):
        print(f"{rank:2d}. {agent_name:20s} - Reward: {metrics['avg_reward']:7.2f} | "
              f"Completions: {metrics['avg_completions']:5.1f}")


def main():
    parser = argparse.ArgumentParser(description='AMD SlingShot Hackathon Demo')
    parser.add_argument('--agent', choices=['dqn', 'greedy', 'adaptive', 'all'], 
                       default='dqn', help='Agent to use for demo')
    parser.add_argument('--steps', type=int, default=50, help='Max steps per episode')
    parser.add_argument('--delay', type=float, default=0.2, help='Delay between steps (seconds)')
    parser.add_argument('--compare', action='store_true', help='Run agent comparison')
    parser.add_argument('--episodes', type=int, default=3, help='Episodes for comparison')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("AMD SlingShot Hackathon - RL Project Manager Demo")
    print("="*70)
    
    if args.compare:
        run_agent_comparison(num_episodes=args.episodes, max_steps_per_episode=args.steps)
        return
    
    # Single agent demo
    env = ProjectEnv(seed=42, enable_diagnostics=True)
    
    agent_map = {
        'dqn': ('DQN', DQNAgent()),
        'greedy': ('Greedy Baseline', GreedyBaseline(env)),
        'adaptive': ('Adaptive Agent', AdaptiveAgent(env))
    }
    
    agent_name, agent = agent_map.get(args.agent, agent_map['dqn'])
    
    # Try to load trained DQN model if available
    if isinstance(agent, DQNAgent):
        model_path = os.path.join(config.CHECKPOINT_DIR, "dqn_final.pth")
        if os.path.exists(model_path):
            print(f"📁 Loading trained model from {model_path}")
            agent.load(model_path)
        else:
            print("⚠️  No trained model found, using untrained agent")
    
    # Run demo
    metrics_history, total_reward = run_demo_episode(
        agent, env, agent_name, 
        max_steps=args.steps,
        delay=args.delay,
        verbose=True
    )
    
    # Print final status
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Total Steps: {len(metrics_history)}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Tasks Completed: {len(env.completed_tasks)}/{env.num_tasks}")
    print(f"Tasks Failed: {len(env.failed_tasks)}")
    print()
    
    # Generate plots
    print("📊 Generating performance plots...")
    plot_simulation_metrics(metrics_history)
    
    # Print summary
    summary = create_demo_summary(metrics_history)
    print_demo_summary(summary)
    
    print("\n🎉 Demo completed!")


if __name__ == "__main__":
    main()
