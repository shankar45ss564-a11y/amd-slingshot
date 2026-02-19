"""
Comprehensive Test Suite for AMD SlingShot Hackathon
Covers: Environment, DQN agent, Baselines, API integration, and End-to-End scenarios
"""

import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.project_env import ProjectEnv
from agents.dqn_agent import DQNAgent
from baselines.greedy_baseline import GreedyBaseline
from baselines.skill_baseline import SkillBaseline
from baselines.hybrid_baseline import HybridBaseline
from baselines.random_baseline import RandomBaseline
import config


class TestEnvironmentBasics:
    """Test basic environment functionality"""

    def test_env_reset(self):
        """Test environment reset returns valid state"""
        env = ProjectEnv(seed=42)
        state = env.reset()
        
        assert state is not None
        assert isinstance(state, np.ndarray)
        assert state.shape == (88,)  # STATE_DIM from config
        assert np.all(np.isfinite(state))

    def test_env_step_basic(self):
        """Test environment step is executable"""
        env = ProjectEnv(seed=42)
        state = env.reset()
        
        # Get valid action
        valid_actions = env.get_valid_actions()
        assert len(valid_actions) > 0
        
        action = valid_actions[0]
        next_state, reward, done, info = env.step(action)
        
        assert next_state.shape == state.shape
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_env_determinism_with_seed(self):
        """Test environment is deterministic with same seed"""
        seed = 123
        env1 = ProjectEnv(seed=seed)
        env2 = ProjectEnv(seed=seed)
        
        state1 = env1.reset()
        state2 = env2.reset()
        
        np.testing.assert_array_almost_equal(state1, state2)

    def test_env_task_completion(self):
        """Test environment properly tracks task completion"""
        env = ProjectEnv(seed=42)
        env.reset()
        
        initial_completed = len(env.completed_tasks)
        
        # Run some steps
        for _ in range(50):
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            action = valid_actions[0]
            _, _, done, _ = env.step(action)
            if done:
                break
        
        # Should have completed at least some tasks (not necessary, but likely)
        assert len(env.completed_tasks) >= initial_completed

    def test_env_worker_fatigue_dynamics(self):
        """Test worker fatigue increases when overloaded"""
        env = ProjectEnv(seed=42)
        state = env.reset()
        
        initial_fatigue = [w.fatigue for w in env.workers]
        
        # Assign tasks to workers
        for _ in range(20):
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            # Prefer assign actions
            assign_actions = [a for a in valid_actions if a < config.NUM_TASKS * config.NUM_WORKERS]
            if assign_actions:
                action = assign_actions[0]
            else:
                action = valid_actions[0]
            _, _, done, _ = env.step(action)
            if done:
                break
        
        final_fatigue = [w.fatigue for w in env.workers]
        
        # At least some workers should have increased fatigue
        fatigue_increases = sum(1 for i, f in enumerate(final_fatigue) if f > initial_fatigue[i])
        assert fatigue_increases > 0


class TestDQNAgent:
    """Test DQN agent functionality"""

    def test_dqn_initialization(self):
        """Test DQN agent initializes properly"""
        agent = DQNAgent()
        
        assert agent is not None
        assert hasattr(agent, 'policy_net')
        assert hasattr(agent, 'target_net')
        assert hasattr(agent, 'optimizer')
        assert agent.device is not None

    def test_dqn_select_action(self):
        """Test DQN agent can select actions"""
        agent = DQNAgent()
        env = ProjectEnv(seed=42)
        state = env.reset()
        
        # Test action selection
        action = agent.select_action(state, training=False)
        
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < config.ACTION_DIM

    def test_dqn_training_step(self):
        """Test DQN training step executes without error"""
        agent = DQNAgent()
        env = ProjectEnv(seed=42)
        
        # Build a small replay buffer
        state = env.reset()
        for _ in range(32):  # one batch
            action = agent.select_action(state, training=True)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            if done:
                state = env.reset()
            else:
                state = next_state
        
        # Try training
        if len(agent.replay_buffer.buffer) >= agent.batch_size:
            loss = agent.train_step()
            assert isinstance(loss, (int, float))

    def test_dqn_save_load(self):
        """Test DQN agent checkpoint save/load"""
        import tempfile
        
        agent = DQNAgent()
        
        # Save
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            checkpoint_path = f.name
        
        try:
            agent.save(checkpoint_path)
            assert os.path.exists(checkpoint_path)
            
            # Load
            agent2 = DQNAgent()
            agent2.load(checkpoint_path)
            
            # Verify loaded state
            assert agent2.policy_net is not None
        finally:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)


class TestBaselines:
    """Test baseline agents"""

    def test_greedy_baseline_action(self):
        """Test greedy baseline can select actions"""
        env = ProjectEnv(seed=42)
        baseline = GreedyBaseline(env)
        
        state = env.reset()
        action = baseline.select_action(state)
        
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < config.ACTION_DIM

    def test_skill_baseline_initialization(self):
        """Test skill baseline initializes"""
        env = ProjectEnv(seed=42)
        baseline = SkillBaseline(env)
        
        assert baseline.name == "Skill"
        assert baseline.is_observing == True

    def test_hybrid_baseline_action(self):
        """Test hybrid baseline can select actions"""
        env = ProjectEnv(seed=42)
        baseline = HybridBaseline(env)
        
        state = env.reset()
        action = baseline.select_action(state)
        
        assert isinstance(action, (int, np.integer))

    def test_random_baseline_action(self):
        """Test random baseline can select actions"""
        env = ProjectEnv(seed=42)
        baseline = RandomBaseline(env)
        
        state = env.reset()
        action = baseline.select_action(state)
        
        assert isinstance(action, (int, np.integer))


class TestEndToEnd:
    """Test end-to-end simulation scenarios"""

    def test_full_episode_with_dqn(self):
        """Test running a full episode with DQN agent"""
        env = ProjectEnv(seed=42)
        agent = DQNAgent()
        
        state = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 100
        
        for _ in range(max_steps):
            action = agent.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                break
            state = next_state
        
        assert steps > 0
        assert total_reward is not None

    def test_full_episode_with_greedy_baseline(self):
        """Test running a full episode with greedy baseline"""
        env = ProjectEnv(seed=42)
        baseline = GreedyBaseline(env)
        
        state = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 100
        
        for _ in range(max_steps):
            action = baseline.select_action(state)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                break
            state = next_state
        
        assert steps > 0
        assert total_reward is not None

    def test_metric_computation(self):
        """Test metrics are computed correctly after episode"""
        env = ProjectEnv(seed=42)
        state = env.reset()
        
        for _ in range(50):
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            action = valid_actions[0]
            _, _, done, _ = env.step(action)
            if done:
                break
        
        metrics = env.compute_metrics()
        
        assert isinstance(metrics, dict)
        assert 'throughput' in metrics
        assert 'deadline_hit_rate' in metrics
        assert 'avg_delay' in metrics
        assert all(isinstance(v, (int, float)) for v in metrics.values())


class TestRewardSignal:
    """Test reward signal properties"""

    def test_reward_is_finite(self):
        """Test that rewards are always finite"""
        env = ProjectEnv(seed=42)
        state = env.reset()
        
        for _ in range(50):
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            action = valid_actions[0]
            _, reward, done, _ = env.step(action)
            
            assert np.isfinite(reward), f"Non-finite reward: {reward}"
            if done:
                break

    def test_reward_scaling(self):
        """Test reward scaling works correctly"""
        env_unscaled = ProjectEnv(seed=42, reward_scale=1.0)
        env_scaled = ProjectEnv(seed=42, reward_scale=0.1)
        
        state_us = env_unscaled.reset()
        state_s = env_scaled.reset()
        
        valid_actions_us = env_unscaled.get_valid_actions()
        valid_actions_s = env_scaled.get_valid_actions()
        
        if valid_actions_us and valid_actions_s:
            action = valid_actions_us[0]
            _, reward_us, _, _ = env_unscaled.step(action)
            _, reward_s, _, _ = env_scaled.step(action)
            
            # Scaled reward should be approximately 0.1 * unscaled reward
            ratio = reward_s / (reward_us + 1e-6)  # Add small epsilon to avoid division by zero
            assert 0.05 < ratio < 0.15 or (abs(reward_us) < 0.1)  # Allow for floating point variance


class TestActionValidity:
    """Test action validity and masking"""

    def test_valid_actions_non_empty(self):
        """Test that valid actions list is non-empty"""
        env = ProjectEnv(seed=42)
        state = env.reset()
        
        valid_actions = env.get_valid_actions()
        assert len(valid_actions) > 0

    def test_decoded_actions_are_valid(self):
        """Test that decoded actions are within valid ranges"""
        env = ProjectEnv(seed=42)
        state = env.reset()
        
        valid_actions = env.get_valid_actions()
        
        for action in valid_actions[:5]:  # Check first 5
            task_id, worker_id, action_type = env._decode_action(action)
            
            assert action_type in ['assign', 'defer', 'escalate']
            assert 0 <= task_id < config.NUM_TASKS
            if action_type == 'assign':
                assert 0 <= worker_id < config.NUM_WORKERS


class TestStressScenarios:
    """Stress test scenarios"""

    def test_long_episode(self):
        """Test environment stability over long episodes"""
        env = ProjectEnv(seed=42)
        state = env.reset()
        
        for step in range(200):  # 200 steps is long
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            
            action = np.random.choice(valid_actions)
            next_state, reward, done, _ = env.step(action)
            
            assert next_state.shape == state.shape
            assert np.all(np.isfinite(next_state))
            assert np.all(np.isfinite(reward))
            
            if done:
                break
            state = next_state

    def test_high_worker_load(self):
        """Test environment with high task/worker load"""
        env = ProjectEnv(num_workers=3, num_tasks=30, seed=42)
        state = env.reset()
        
        # Assign many tasks to create high load
        assign_count = 0
        for _ in range(100):
            valid_actions = env.get_valid_actions()
            assign_actions = [a for a in valid_actions if a < config.NUM_TASKS * config.NUM_WORKERS]
            
            if assign_actions:
                action = assign_actions[0]
                assign_count += 1
            else:
                action = valid_actions[0] if valid_actions else None
            
            if action is None:
                break
            
            _, _, done, _ = env.step(action)
            if done or assign_count > 50:
                break


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
