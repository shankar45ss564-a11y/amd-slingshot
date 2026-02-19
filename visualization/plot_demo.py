"""
Visualization functions for the project management demo
Generates plots showing simulation metrics over time
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Dict

def plot_simulation_metrics(metrics_history: List[Dict], save_path: str = "demo_metrics.png"):
    """
    Plot key simulation metrics over time

    Args:
        metrics_history: List of metrics dictionaries from each step
        save_path: Path to save the plot
    """
    if not metrics_history:
        print("No metrics to plot")
        return

    steps = range(len(metrics_history))

    # Extract metrics
    task_completion = [m.get('task_completion_rate', 0) * 100 for m in metrics_history]
    avg_fatigue = [m.get('avg_worker_fatigue', 0) for m in metrics_history]
    deadline_hits = [m.get('deadline_hit_rate', 0) * 100 for m in metrics_history]
    total_reward = [m.get('total_reward', 0) for m in metrics_history]
    active_tasks = [m.get('active_tasks', 0) for m in metrics_history]

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Project Management Simulation Metrics', fontsize=16)

    # Task completion rate
    axes[0, 0].plot(steps, task_completion, 'b-', linewidth=2, marker='o')
    axes[0, 0].set_title('Task Completion Rate (%)')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Completion Rate (%)')
    axes[0, 0].grid(True, alpha=0.3)

    # Average worker fatigue
    axes[0, 1].plot(steps, avg_fatigue, 'r-', linewidth=2, marker='s')
    axes[0, 1].set_title('Average Worker Fatigue')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Fatigue Level')
    axes[0, 1].axhline(y=2.5, color='orange', linestyle='--', alpha=0.7, label='Burnout Threshold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Deadline hit rate
    axes[1, 0].plot(steps, deadline_hits, 'g-', linewidth=2, marker='^')
    axes[1, 0].set_title('Deadline Hit Rate (%)')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Hit Rate (%)')
    axes[1, 0].grid(True, alpha=0.3)

    # Active tasks over time
    axes[1, 1].plot(steps, active_tasks, 'purple', linewidth=2, marker='d')
    axes[1, 1].set_title('Active Tasks')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Number of Active Tasks')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Plot saved to {save_path}")

    # Show plot if running interactively
    try:
        plt.show()
    except:
        pass  # May not work in headless environment

def plot_worker_fatigue_over_time(metrics_history: List[Dict], save_path: str = "worker_fatigue.png"):
    """
    Plot individual worker fatigue levels over time

    Args:
        metrics_history: List of metrics dictionaries
        save_path: Path to save the plot
    """
    if not metrics_history or 'worker_fatigue' not in metrics_history[0]:
        print("No worker fatigue data to plot")
        return

    steps = range(len(metrics_history))
    num_workers = len(metrics_history[0]['worker_fatigue'])

    plt.figure(figsize=(10, 6))

    for i in range(num_workers):
        fatigue_levels = [m['worker_fatigue'][i] for m in metrics_history]
        plt.plot(steps, fatigue_levels, label=f'Worker {i+1}', linewidth=2, marker='o')

    plt.title('Worker Fatigue Levels Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Fatigue Level')
    plt.axhline(y=2.5, color='red', linestyle='--', alpha=0.7, label='Burnout Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Worker fatigue plot saved to {save_path}")

    try:
        plt.show()
    except:
        pass

def create_demo_summary(metrics_history: List[Dict]) -> Dict:
    """
    Create a summary of the demo performance

    Args:
        metrics_history: List of metrics dictionaries

    Returns:
        Dictionary with summary statistics
    """
    if not metrics_history:
        return {}

    final_metrics = metrics_history[-1]

    summary = {
        'total_steps': len(metrics_history),
        'final_completion_rate': final_metrics.get('task_completion_rate', 0) * 100,
        'final_deadline_hit_rate': final_metrics.get('deadline_hit_rate', 0) * 100,
        'avg_fatigue': np.mean([m.get('avg_worker_fatigue', 0) for m in metrics_history]),
        'max_fatigue': max([m.get('avg_worker_fatigue', 0) for m in metrics_history]),
        'total_overload_events': sum([m.get('overload_events', 0) for m in metrics_history]),
        'total_completed': final_metrics.get('completed_tasks', 0),
        'total_failed': final_metrics.get('failed_tasks', 0)
    }

    return summary

def print_demo_summary(summary: Dict):
    """Print a formatted summary of the demo"""
    print("\n" + "="*50)
    print("DEMO SUMMARY")
    print("="*50)
    print(f"Total Steps: {summary.get('total_steps', 0)}")
    print(".1f")
    print(".1f")
    print(".2f")
    print(".2f")
    print(f"Total Overload Events: {summary.get('total_overload_events', 0)}")
    print(f"Tasks Completed: {summary.get('total_completed', 0)}")
    print(f"Tasks Failed: {summary.get('total_failed', 0)}")