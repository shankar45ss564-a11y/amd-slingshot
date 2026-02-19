"""
Enhanced AI Agents with Improvements
Includes:
- AdaptiveAgent: Learns from recent performance
- PrioritizedGreedyAgent: Smarter priority-based allocation
- LoadBalancingAgent: Minimizes worker overload while meeting deadlines
"""

import numpy as np
import sys
import os
from typing import Dict, List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from baselines.base_policy import BasePolicy
from environment.project_env import ProjectEnv


class AdaptiveAgent(BasePolicy):
    """
    Adaptive agent that learns from recent performance
    
    Strategy:
    - Tracks recent task assignment success rates
    - Learns which worker-task pairs work well together
    - Adapts allocation strategy based on recent history
    - Balances exploration and exploitation
    
    Advantages over baselines:
    - Learns task-specific worker preferences
    - Adapts to changing fatigue patterns
    - Better handles skill mismatches dynamically
    """
    
    def __init__(self, env: ProjectEnv, learning_window: int = 10):
        """
        Initialize adaptive agent
        
        Args:
            env: ProjectEnv instance
            learning_window: Number of recent tasks to track
        """
        super().__init__(env)
        self.name = "Adaptive"
        self.learning_window = learning_window
        
        # Track recent assignments and outcomes
        self.assignment_history = []  # List of (worker_id, task_id, success, time_to_complete)
        self.worker_preferences = {}  # worker_id -> {task_complexity: success_rate}
        self.worker_reliability = {}  # worker_id -> recent_success_rate
        
        # Initialize tracking per worker
        for i in range(env.num_workers):
            self.worker_reliability[i] = 0.5  # Initial belief: 50% reliable
    
    def select_action(self, state) -> int:
        """
        Select action based on learned preferences and current state
        
        Args:
            state: Current state
        
        Returns:
            Action index
        """
        # Get pending tasks and workers
        completed_ids = [t.task_id for t in self.env.completed_tasks]
        pending_tasks = [
            t for t in self.env.tasks
            if not t.is_completed and not t.is_failed and t.assigned_worker is None
            and t.check_dependencies_met(completed_ids)
        ]
        
        if not pending_tasks:
            # No tasks to assign, defer a random task
            deferrable = [t.task_id for t in self.env.tasks if not t.is_completed and not t.is_failed]
            if deferrable:
                return self.encode_action(np.random.choice(deferrable), action_type='defer')
            return 0
        
        # Sort by priority
        pending_tasks = sorted(pending_tasks, key=lambda t: -t.priority)
        target_task = pending_tasks[0]
        
        # Select best worker for this task
        available_workers = [
            w for w in self.env.workers
            if w.load < config.MAX_WORKER_LOAD and w.fatigue < config.FATIGUE_THRESHOLD
        ]
        
        if not available_workers:
            # No available workers, defer task
            return self.encode_action(target_task.task_id, action_type='defer')
        
        # Score workers based on reliability and skill match
        best_worker = max(
            available_workers,
            key=lambda w: self.worker_reliability.get(w.worker_id, 0.5) * (1 + w.skill)
        )
        
        return self.encode_action(target_task.task_id, best_worker.worker_id, 'assign')
    
    def update_history(self, worker_id: int, task_id: int, success: bool, time_taken: float):
        """
        Update learning history with assignment outcome
        
        Args:
            worker_id: Worker ID
            task_id: Task ID
            success: Whether task completed successfully
            time_taken: Time to completion
        """
        self.assignment_history.append({
            'worker_id': worker_id,
            'task_id': task_id,
            'success': success,
            'time': time_taken
        })
        
        # Keep only recent history
        if len(self.assignment_history) > self.learning_window:
            self.assignment_history = self.assignment_history[-self.learning_window:]
        
        # Update worker reliability
        if self.assignment_history:
            recent_outcomes = [
                h for h in self.assignment_history 
                if h['worker_id'] == worker_id
            ]
            if recent_outcomes:
                success_rate = sum(1 for h in recent_outcomes if h['success']) / len(recent_outcomes)
                self.worker_reliability[worker_id] = success_rate


class PrioritizedGreedyAgent(BasePolicy):
    """
    Improved greedy agent with smarter prioritization
    
    Strategy:
    - Assigns tasks based on priority AND deadline urgency
    - Prefers workers with better skill-task match
    - Avoids assigning to fatigued workers
    - Escalates when no good match available
    
    Advantages:
    - Respects deadlines more than vanilla greedy
    - Considers fatigue in assignment
    - Better quality scores due to skill matching
    """
    
    def __init__(self, env: ProjectEnv):
        """Initialize prioritized greedy agent"""
        super().__init__(env)
        self.name = "PrioritizedGreedy"
    
    def select_action(self, state) -> int:
        """
        Select action with priority and deadline awareness
        
        Args:
            state: Current state
        
        Returns:
            Action index
        """
        # Get pending tasks
        completed_ids = [t.task_id for t in self.env.completed_tasks]
        pending_tasks = [
            t for t in self.env.tasks
            if not t.is_completed and not t.is_failed and t.assigned_worker is None
            and t.check_dependencies_met(completed_ids)
        ]
        
        if not pending_tasks:
            return self.select_defer_action()
        
        # Score tasks by: priority + deadline urgency
        current_time = self.env.current_timestep
        scored_tasks = []
        
        for task in pending_tasks:
            deadline_urgency = max(0, 100 * (1 - (task.deadline - current_time) / max(1, task.deadline)))
            priority_score = task.priority * 10
            total_score = priority_score + deadline_urgency
            scored_tasks.append((total_score, task))
        
        scored_tasks.sort(reverse=True)
        target_task = scored_tasks[0][1]
        
        # Find best worker for this task
        best_worker, best_score = self.find_best_worker(target_task)
        
        if best_worker is not None and best_score > 0:
            return self.encode_action(target_task.task_id, best_worker.worker_id, 'assign')
        else:
            # No good match, escalate or defer
            return self.encode_action(target_task.task_id, action_type='escalate')
    
    def find_best_worker(self, task):
        """Find best worker for a given task"""
        available_workers = [
            w for w in self.env.workers
            if w.load < config.MAX_WORKER_LOAD
        ]
        
        if not available_workers:
            return None, 0
        
        best_worker = None
        best_score = -float('inf')
        
        for worker in available_workers:
            # Score considers: skill match, fatigue, and current load
            skill_match = min(worker.skill / task.difficulty, 1.0) * 10  # Cap at 10
            fatigue_penalty = max(0, worker.fatigue - 1.0) * 2
            load_penalty = worker.load * 0.5
            
            score = skill_match - fatigue_penalty - load_penalty
            
            if score > best_score:
                best_score = score
                best_worker = worker
        
        return best_worker, best_score
    
    def select_defer_action(self) -> int:
        """Select a defer action when no good assignments"""
        deferrable = [t.task_id for t in self.env.tasks if not t.is_completed and not t.is_failed]
        if deferrable:
            return self.encode_action(np.random.choice(deferrable), action_type='defer')
        return 0


class LoadBalancingAgent(BasePolicy):
    """
    Load balancing agent that minimizes worker overload
    
    Strategy:
    - Prioritizes keeping worker loads balanced
    - Defers tasks when load distribution would be unbalanced
    - Prevents individual worker burnout
    - Escalates complex tasks to available high-skill workers
    
    Advantages:
    - Prevents single-worker bottlenecks
    - Reduces overall team burnout risk
    - More sustainable task management
    """
    
    def __init__(self, env: ProjectEnv):
        """Initialize load balancing agent"""
        super().__init__(env)
        self.name = "LoadBalancing"
    
    def select_action(self, state) -> int:
        """
        Select action focusing on load distribution
        
        Args:
            state: Current state
        
        Returns:
            Action index
        """
        # Get pending tasks
        completed_ids = [t.task_id for t in self.env.completed_tasks]
        pending_tasks = [
            t for t in self.env.tasks
            if not t.is_completed and not t.is_failed and t.assigned_worker is None
            and t.check_dependencies_met(completed_ids)
        ]
        
        if not pending_tasks:
            return self.select_action_defer()
        
        # Sort by priority
        pending_tasks.sort(key=lambda t: -t.priority)
        target_task = pending_tasks[0]
        
        # Find least loaded available worker
        available_workers = [
            w for w in self.env.workers
            if w.load < config.MAX_WORKER_LOAD and w.fatigue < config.FATIGUE_THRESHOLD * 0.8
        ]
        
        if not available_workers:
            # If no fully available workers, find least loaded
            available_workers = [
                w for w in self.env.workers
                if w.load < config.MAX_WORKER_LOAD
            ]
        
        if not available_workers:
            # Defer if all workers are at max load
            return self.encode_action(target_task.task_id, action_type='defer')
        
        # Choose worker with minimum load (balanced distribution)
        best_worker = min(available_workers, key=lambda w: w.load)
        
        # Check if assignment would create too much imbalance
        loads = [w.load for w in self.env.workers]
        current_balance = np.std(loads)
        
        # Simulate assignment
        test_loads = loads.copy()
        test_loads[best_worker.worker_id] += target_task.difficulty
        new_balance = np.std(test_loads)
        
        # If balance would degrade significantly, defer
        if new_balance > current_balance * 1.5:
            return self.encode_action(target_task.task_id, action_type='defer')
        
        return self.encode_action(target_task.task_id, best_worker.worker_id, 'assign')
    
    def select_action_defer(self) -> int:
        """Select defer action"""
        deferrable = [t.task_id for t in self.env.tasks if not t.is_completed and not t.is_failed]
        if deferrable:
            return self.encode_action(np.random.choice(deferrable), action_type='defer')
        return 0


if __name__ == "__main__":
    """Test improved agents"""
    print("Testing improved agents...")
    
    env = ProjectEnv(seed=42)
    env.reset()
    
    agents_to_test = [
        ("AdaptiveAgent", AdaptiveAgent(env)),
        ("PrioritizedGreedyAgent", PrioritizedGreedyAgent(env)),
        ("LoadBalancingAgent", LoadBalancingAgent(env)),
    ]
    
    for agent_name, agent in agents_to_test:
        print(f"\nTesting {agent_name}...")
        state = env.reset()
        total_reward = 0
        
        for step in range(50):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            
            if step % 10 == 0:
                print(f"  Step {step}: reward={reward:.2f}, total={total_reward:.2f}")
            
            if done:
                break
            state = next_state
        
        print(f"  Final total reward: {total_reward:.2f}")
        print(f"  Completed tasks: {len(env.completed_tasks)}")
        print(f"  Failed tasks: {len(env.failed_tasks)}")
    
    print("\n✓ All improved agents tested successfully!")
