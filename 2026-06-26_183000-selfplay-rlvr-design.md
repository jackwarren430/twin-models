# Self-Play RLVR with Independent Dual-Weight Models — Design Spec

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Design a self-play reinforcement learning framework where two independently-weighted LLM instances (A and B) specialize as alternating problem creators and solvers, with role-swap injection, a knowledge oracle, and multi-signal reward shaping to address known self-play RL failure modes.

**Architecture:** Two parameter-separated models derived from a shared frozen base, each fine-tuned independently via LoRA. A central replay buffer stores generated problem-solution pairs. Training alternates roles each iteration. Rewards come from three orthogonal signals: difficulty gradient fitting, solution consistency verification, and oracle-cost penalization.

**Tech Stack:** Python 3.11+, vLLM or llama-cpp for inference, HuggingFace Transformers + PEFT (LoRA), Ray/RLlib or custom PPO/GRPO loop, Weights & Biases for experiment tracking, GGUF quantization for local inference on Apple Silicon.

---

## 1. System Overview

### 1.1 Core Participants

```
┌──────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR (Controller)                     │
│  Manages: role assignment, replay buffer, training loop,         │
│            reward computation, checkpointing, logging            │
└──────────────────────────────────────────────────────────────────┘
       ▲                                    ▲
       │                                    │
  ┌────┴────┐                        ┌─────┴─────┐
  │  Model A │                        │  Model B  │
  │ (LoRA-A) │◄─── role swap ───────►│ (LoRA-B)  │
  └────┬────┘                        └─────┬─────┘
       │                                    │
       │   tools:                           │   tools:
       │   • code exec                      │   • code exec
       │   • search                         │   • search
       │   • oracle (frozen base)           │   • oracle (frozen base)
       │   • verifier                       │   • verifier
       └─────────────────────────────────────────────────────────┘
                                    │
                          ┌─────────┴─────────┐
                          │  Knowledge Oracle  │
                          │  (frozen pretrained │
                          │   base model)       │
                          └─────────────────────┘
```

### 1.2 Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Base model | Qwen2.5-14B (or Llama-3.1-13B) | Strong reasoning, good tool-use, fits in 32GB at 4-bit |
| Fine-tuning | LoRA adapters (rank 64-128) | Memory-efficient, enables independent weight tracking |
| RL algorithm | GRPO (Group Relative Policy Optimization) | Simpler than PPO, no critic needed, used in Qwen-R1 |
| Inference | vLLM (quantized GGUF) or llama-cpp | Apple Silicon support, fast batched generation |
| Replay buffer | In-memory + disk checkpoint | Problem-solution pairs with metadata |
| Quantization | 4-bit GGUF via llama.cpp | ~7GB for 14B model, leaves room for LoRA + overhead |
| Experiment tracking | Weights & Biases | Multi-agent metrics, reward curves, hyperparameter logging |

---

## 2. Component Specifications

### 2.1 Model Architecture

```python
# models/base.py
# ──────────────────────────────────────────────────────────────
# Shared frozen base model (loaded once, never updated)
# ──────────────────────────────────────────────────────────────

class FrozenBaseModel:
    """
    The frozen pretrained base model. Serves two roles:
    1. Knowledge oracle: provides stable factual grounding
    2. Weight initialization: both A and B start from here
    
    Loaded via llama.cpp (GGUF) for Apple Silicon efficiency.
    Never receives gradient updates.
    """
    model_path: str          # Path to GGUF file
    quantization: str        # "q4_k_m" recommended
    device: str              # "mps" on macOS
    context_length: int      # 8192 or 32768
    temperature: float       # For oracle queries
    
    def query(self, prompt: str, max_tokens: int = 1024) -> str: ...
    def get_embedding(self, text: str) -> np.ndarray: ...
    
class IndependentModel:
    """
    A trainable model with independent LoRA adapter.
    Can be loaded in either creator or solver role.
    """
    base_model: FrozenBaseModel
    lora_path: str           # Path to LoRA adapter weights
    lora_rank: int           # 64 or 128
    lora_alpha: float        # LoRA scaling factor
    role: Literal["creator", "solver"]
    model_id: int            # 0 or 1 (A or B)
    
    def load(self): ...
    def generate(self, prompt: str, **kwargs) -> str: ...
    def save_checkpoint(self, path: str): ...
    def get_weights_snapshot(self) -> dict: ...
```

### 2.2 Role Manager

```python
# roles/manager.py
# ──────────────────────────────────────────────────────────────
# Handles role assignment, swapping, and the hybridization step
# ──────────────────────────────────────────────────────────────

class RoleManager:
    """
    Manages the alternating role assignment between Model A and Model B.
    
    Role swap schedule:
    - Every N training iterations (configurable, default 500)
    - On swap: solver's LoRA weights are partially blended into creator's
      LoRA adapter (injection rate: 5-15%)
    
    This injects the solver's learned heuristics into the creator role,
    hypothesized to produce richer curricula over time.
    """
    
    swap_interval: int           # iterations between swaps
    injection_rate: float        # 0.05 - 0.15
    swap_history: list[dict]     # logs of all swaps
    
    def assign_roles(self, iteration: int) -> tuple[Literal["creator"], Literal["solver"]]:
        """Returns which model is creator and which is solver at this iteration."""
        should_swap = (iteration % self.swap_interval == 0)
        if should_swap and not self._first_assignment:
            self._execute_swap()
        return self._current_assignment
    
    def _execute_swap(self):
        """
        Partial weight injection from solver to creator:
        new_creator_lora = (1 - injection_rate) * old_creator_lora 
                         + injection_rate * solver_lora
        """
        solver_weights = self.solver_model.get_weights_snapshot()
        creator_weights = self.creator_model.get_weights_snapshot()
        
        blended = {}
        for key in solver_weights:
            blended[key] = (
                (1 - self.injection_rate) * creator_weights[key] +
                self.injection_rate * solver_weights[key]
            )
        
        self.creator_model.apply_blended_weights(blended)
        self._current_assignment = self._swap_roles()
```

### 2.3 Knowledge Oracle Tool

```python
# tools/oracle.py
# ──────────────────────────────────────────────────────────────
# The oracle provides stable factual grounding during RL training.
# Counteracts knowledge degradation from RL fine-tuning.
# ──────────────────────────────────────────────────────────────

class KnowledgeOracle:
    """
    Routes queries to the frozen pretrained base model.
    
    Design rationale:
    - RL fine-tuning causes factual knowledge degradation (the "catastrophic 
      forgetting" problem in RLVR)
    - The oracle provides a stable epistemic floor: factual queries are 
      answered from the pretrained distribution, not the fine-tuned policy
    - An oracle_cost tax in the reward function discourages over-reliance
    
    Usage by creator/solver:
    The model calls this tool via function calling / tool-use format.
    """
    
    def __init__(self, base_model: FrozenBaseModel, cost_per_query: float = 1.0):
        self.base_model = base_model
        self.cost_per_query = cost_per_query  # Used in reward calculation
    
    def query(self, question: str) -> dict:
        """
        Query the oracle for factual knowledge.
        
        Returns:
            {
                "answer": str,           # Oracle's response
                "confidence": float,     # 0-1 confidence estimate
                "sources": list[str],    # Retrieved reference snippets
                "cost": float,           # Oracle cost (accumulated)
                "tokens_used": int       # Total tokens consumed
            }
        """
        answer = self.base_model.generate(f"Answer this factually: {question}")
        return {
            "answer": answer,
            "confidence": self._estimate_confidence(answer),
            "sources": self._retrieve_sources(question),
            "cost": self.cost_per_query,
            "tokens_used": len(answer.split())
        }
    
    def _estimate_confidence(self, answer: str) -> float: ...
    def _retrieve_sources(self, query: str) -> list[str]: ...


# Tool definition for model function calling
ORACLE_TOOL = {
    "type": "function",
    "function": {
        "name": "knowledge_oracle",
        "description": "Query the frozen knowledge base for factual information. "
                       "Use sparingly — each query incurs a reward penalty.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The factual question to query"
                }
            },
            "required": ["question"]
        }
    }
}
```

### 2.4 Problem Generator (Creator Output Schema)

```python
# problems/schema.py
# ──────────────────────────────────────────────────────────────
# The creator must generate problem suites, not single problems.
# Each suite spans a full difficulty gradient (100% → 0% solve rate).
# ──────────────────────────────────────────────────────────────

@dataclass
class Problem:
    """A single problem in a suite."""
    problem_id: str
    text: str                    # Problem description
    difficulty: float            # 0.0 (trivial) to 1.0 (impossible)
    domain: str                  # e.g., "math", "logic", "programming"
    tools_required: list[str]    # Which tools needed to solve
    solution: str                # Complete solution
    explanation: str             # Step-by-step explanation
    verification_code: str | None  # Code to verify the solution
    
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> Problem: ...


@dataclass
class ProblemSuite:
    """
    A suite of problems spanning the full difficulty gradient.
    
    Required properties:
    - At least 5 problems
    - Linear difficulty spacing (0.0, 0.25, 0.5, 0.75, 1.0)
    - Each problem has a verified solution
    - Solve rate across the suite should be approximately linear:
      solve_rate(difficulty) ≈ 1.0 - difficulty
    """
    suite_id: str
    theme: str                   # High-level topic
    problems: list[Problem]
    metadata: dict               # Generation context, model version, etc.
    
    @property
    def solve_rate_curve(self) -> list[float]:
        """Expected solve rate at each difficulty level."""
        return [1.0 - p.difficulty for p in self.problems]
    
    def validate_gradient(self) -> dict:
        """
        Check that the difficulty gradient is valid:
        - Linear spacing between consecutive problems
        - Minimum 5 problems
        - All have solutions
        - Solutions are non-trivial
        """
        issues = []
        if len(self.problems) < 5:
            issues.append("Need at least 5 problems")
        
        difficulties = [p.difficulty for p in self.problems]
        for i in range(1, len(difficulties)):
            gap = difficulties[i] - difficulties[i-1]
            if abs(gap - (difficulties[-1] - difficulties[0]) / (len(difficulties) - 1)) > 0.1:
                issues.append(f"Non-linear gap at index {i}: {gap:.2f}")
        
        return {"valid": len(issues) == 0, "issues": issues}
```

### 2.5 Reward Engine

```python
# rewards/engine.py
# ──────────────────────────────────────────────────────────────
# Three orthogonal reward signals:
# 1. Difficulty gradient reward (creator)
# 2. Solve-rate reward (solver)
# 3. Consistency penalty (both, but affects creator more)
# ──────────────────────────────────────────────────────────────

class RewardEngine:
    """
    Computes orthogonal reward signals for both creator and solver.
    
    Reward decomposition:
    
    R_creator = λ₁ * R_gradient + λ₂ * R_consistency - λ₃ * R_oracle_cost
    R_solver  = λ₄ * R_solve + λ₅ * R_efficiency
    
    Where:
    - R_gradient: How well the suite's solve rate matches the target curve
    - R_consistency: Solver's judgment of creator's solution quality
    - R_oracle_cost: Penalty for oracle queries (encourages self-reliance)
    - R_solve: Binary reward for each solved problem
    - R_efficiency: Bonus for solving with fewer steps/tools
    """
    
    def __init__(
        self,
        lambda_gradient: float = 1.0,
        lambda_consistency: float = 0.5,
        lambda_oracle_cost: float = 0.1,
        lambda_solve: float = 1.0,
        lambda_efficiency: float = 0.2,
        target_solve_curve: list[float] = None,
    ):
        self.lambda_gradient = lambda_gradient
        self.lambda_consistency = lambda_consistency
        self.lambda_oracle_cost = lambda_oracle_cost
        self.lambda_solve = lambda_solve
        self.lambda_efficiency = lambda_efficiency
        self.target_solve_curve = target_solve_curve or [1.0, 0.75, 0.5, 0.25, 0.0]
    
    def compute_creator_rewards(
        self,
        suite: ProblemSuite,
        solver_results: dict,  # {problem_id: solved (bool)}
        oracle_queries: int,
    ) -> RewardResult:
        """
        Compute rewards for the creator model.
        
        R_gradient: Measures how closely the actual solve rate matches
        the target linear curve (100% → 0% across difficulty levels).
        
        R_consistency: Measures whether the creator's provided solution
        matches what the solver found (via external verifier).
        
        R_oracle_cost: Penalizes oracle usage.
        """
        # Gradient reward: negative MSE between actual and target solve rates
        actual_rates = self._compute_actual_solve_rates(suite, solver_results)
        gradient_score = -self._mse(actual_rates, self.target_solve_curve)
        
        # Normalize to [0, 1] via sigmoid
        gradient_reward = 1.0 / (1.0 + math.exp(-gradient_score))
        
        # Consistency reward
        consistency_score = self._compute_consistency(suite, solver_results)
        
        # Oracle cost penalty
        oracle_penalty = self.lambda_oracle_cost * oracle_queries
        
        return RewardResult(
            total=self.lambda_gradient * gradient_reward 
                   + self.lambda_consistency * consistency_score 
                   - oracle_penalty,
            components={
                "gradient": self.lambda_gradient * gradient_reward,
                "consistency": self.lambda_consistency * consistency_score,
                "oracle_cost": -oracle_penalty,
            }
        )
    
    def compute_solver_rewards(
        self,
        suite: ProblemSuite,
        solver_results: dict,  # {problem_id: solved (bool)}
        solver_steps: int,
    ) -> RewardResult:
        """
        Compute rewards for the solver model.
        
        R_solve: Direct reward for each problem solved.
        R_efficiency: Bonus for solving with fewer steps.
        """
        solved_count = sum(solver_results.values())
        solve_reward = self.lambda_solve * solved_count / len(suite.problems)
        
        efficiency_reward = self.lambda_efficiency * max(0, 1 - solver_steps / 20)
        
        return RewardResult(
            total=solve_reward + efficiency_reward,
            components={
                "solve": solve_reward,
                "efficiency": efficiency_reward,
            }
        )
    
    def _compute_actual_solve_rates(self, suite, results): ...
    def _compute_consistency(self, suite, results): ...
    def _mse(self, actual, target): ...


@dataclass
class RewardResult:
    total: float
    components: dict[str, float]
    metadata: dict = field(default_factory=dict)
```

### 2.6 External Verifier

```python
# verifiers/base.py
# ──────────────────────────────────────────────────────────────
# External verification of solution correctness.
# ──────────────────────────────────────────────────────────────

class Verifier:
    """
    External verifier that judges whether a solution is correct.
    
    Verification strategies depend on problem type:
    - Math: symbolic computation (SymPy)
    - Programming: test execution (code interpreter)
    - Logic: constraint satisfaction (Z3)
    - Open-ended: LLM-based judgment with rubric
    
    The verifier is used by the solver to judge the creator's 
    provided solution, not the solver's own solution.
    """
    
    def verify_math(self, problem: str, solution: str) -> VerificationResult: ...
    def verify_programming(self, problem: str, solution: str, test_cases: list) -> VerificationResult: ...
    def verify_logic(self, problem: str, solution: str) -> VerificationResult: ...
    def verify_open_ended(self, problem: str, solution: str, rubric: str) -> VerificationResult: ...


@dataclass
class VerificationResult:
    correct: bool
    confidence: float
    details: str
    score: float  # 0.0 - 1.0 continuous score
```

### 2.7 Replay Buffer

```python
# replay/buffer.py
# ──────────────────────────────────────────────────────────────
# Stores problem-solution pairs for off-policy training.
# ──────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Stores generated problem suites and their outcomes.
    
    Strategy:
    - Priority-based sampling: harder problems sampled more often
    - Decay: older entries gradually lose priority
    - Capacity: configurable max size, LRU eviction
    - Metadata: model version, reward scores, difficulty stats
    
    Enables off-policy training: the solver can learn from problems
    generated by the creator (and vice versa after role swap).
    """
    
    def __init__(self, max_size: int = 100000, priority_scale: float = 0.7):
        self.max_size = max_size
        self.priority_scale = priority_scale
        self.entries: list[ReplayEntry] = []
        self.priorities: list[float] = []
    
    def add(self, suite: ProblemSuite, rewards: RewardResult, metadata: dict): ...
    def sample(self, batch_size: int) -> list[ReplayEntry]: ...
    def update_priorities(self, indices: list[int], new_priorities: list[float]): ...
    def save_checkpoint(self, path: str): ...
    def load_checkpoint(self, path: str): ...
    
    def _compute_priority(self, entry: ReplayEntry) -> float:
        """Priority = reward * (1 + difficulty) * decay_factor"""
        base = entry.reward.total * (1 + entry.avg_difficulty)
        decay = math.exp(-entry.age_days / 30)  # 30-day half-life
        return base * decay
```

---

## 3. Training Loop

### 3.1 Main Loop (GRPO-based)

```python
# train/main_loop.py
# ──────────────────────────────────────────────────────────────
# The core training loop. Alternates roles each iteration.
# ──────────────────────────────────────────────────────────────

class SelfPlayTrainer:
    """
    Main training loop for dual-weight self-play RLVR.
    
    Algorithm overview:
    1. Assign roles (creator A + solver B, or vice versa)
    2. Creator generates problem suite
    3. Solver attempts to solve each problem (with tools)
    4. Verifier checks creator's provided solutions
    5. Compute orthogonal rewards
    6. Update both models via GRPO
    7. Periodically swap roles and inject weights
    8. Log metrics, checkpoint, repeat
    """
    
    def __init__(
        self,
        model_a: IndependentModel,
        model_b: IndependentModel,
        oracle: KnowledgeOracle,
        verifier: Verifier,
        reward_engine: RewardEngine,
        replay_buffer: ReplayBuffer,
        role_manager: RoleManager,
        config: TrainingConfig,
    ):
        self.model_a = model_a
        self.model_b = model_b
        self.oracle = oracle
        self.verifier = verifier
        self.reward_engine = reward_engine
        self.replay_buffer = replay_buffer
        self.role_manager = role_manager
        self.config = config
    
    def train(self, total_iterations: int = 10000):
        """Main training loop."""
        for iteration in range(1, total_iterations + 1):
            # Phase 1: Role assignment
            creator_model, solver_model = self.role_manager.assign_roles(iteration)
            phase = "swap" if self.role_manager.just_swapped else "normal"
            
            # Phase 2: Creator generates suite
            creator_output = self._run_creator(creator_model, iteration)
            suite = self._parse_suite(creator_output)
            
            # Phase 3: Solver attempts problems
            solver_output = self._run_solver(solver_model, suite, iteration)
            solver_results = self._parse_results(solver_output)
            
            # Phase 4: Verify creator's solutions
            consistency_scores = self._verify_solutions(suite)
            
            # Phase 5: Compute rewards
            creator_rewards = self.reward_engine.compute_creator_rewards(
                suite, solver_results, 
                oracle_queries=creator_output.oracle_count
            )
            solver_rewards = self.reward_engine.compute_solver_rewards(
                suite, solver_results,
                solver_steps=solver_output.steps
            )
            
            # Phase 6: Update replay buffer
            self.replay_buffer.add(suite, creator_rewards, {
                "iteration": iteration,
                "phase": phase,
                "creator_model": creator_model.model_id,
                "solver_model": solver_model.model_id,
            })
            
            # Phase 7: GRPO updates
            self._grpo_update(creator_model, suite, creator_rewards, "creator")
            self._grpo_update(solver_model, suite, solver_rewards, "solver")
            
            # Phase 8: Logging and checkpointing
            self._log_metrics(iteration, creator_rewards, solver_rewards, phase)
            
            if iteration % self.config.checkpoint_interval == 0:
                self._checkpoint(iteration)
            
            # Phase 9: Role swap if needed
            if self.role_manager.just_swapped:
                self._log_swap_info()
    
    def _run_creator(self, model: IndependentModel, iteration: int) -> dict: ...
    def _run_solver(self, model: IndependentModel, suite: ProblemSuite, iteration: int) -> dict: ...
    def _verify_solutions(self, suite: ProblemSuite) -> list[float]: ...
    def _grpo_update(self, model: IndependentModel, suite: ProblemSuite, rewards: RewardResult, role: str): ...
    def _log_metrics(self, iteration: int, creator_rewards: RewardResult, solver_rewards: RewardResult, phase: str): ...
    def _checkpoint(self, iteration: int): ...
```

### 3.2 GRPO Update Step

```python
# train/grpo.py
# ──────────────────────────────────────────────────────────────
# Group Relative Policy Optimization update.
# No critic network needed — advantages computed from group.
# ──────────────────────────────────────────────────────────────

def grpo_update(
    model: IndependentModel,
    prompts: list[str],
    responses: list[list[str]],  # Group of G responses per prompt
    rewards: list[float],        # One reward per response
    kl_coeff: float = 0.01,
    clip_range: float = 0.2,
    learning_rate: float = 1e-5,
) -> dict:
    """
    GRPO update step.
    
    For each prompt, we generate G responses. The advantage for
    response i is:
    
        A_i = (R_i - mean(R_group)) / (std(R_group) + epsilon)
    
    The policy is updated to maximize:
    
        L = E[min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)] - β * KL
    
    Where ratio = π_new / π_old (probability ratio).
    
    Compared to PPO:
    - No value function / critic network
    - Simpler implementation
    - Lower variance with larger groups (G=8-16)
    """
    # Reference model (frozen base) for KL penalty
    ref_model = model.base_model
    
    for prompt, group_responses, group_rewards in zip(prompts, responses, rewards):
        # Compute advantages from the group
        mean_r = np.mean(group_rewards)
        std_r = np.std(group_rewards) + 1e-8
        advantages = [(r - mean_r) / std_r for r in group_rewards]
        
        # Compute probability ratios
        ratios = []
        for response in group_responses:
            log_prob_new = model.log_prob(prompt, response)
            log_prob_ref = ref_model.log_prob(prompt, response)
            ratio = torch.exp(log_prob_new - log_prob_ref)
            ratios.append(ratio)
        
        # PPO-style clipped objective
        surr1 = [r * a for r, a in zip(ratios, advantages)]
        surr2 = [
            torch.clamp(r, 1 - clip_range, 1 + clip_range) * a
            for r, a in zip(ratios, advantages)
        ]
        
        # KL penalty relative to reference
        kl = torch.mean(model.kl_divergence(prompt, group_responses, ref_model))
        
        # Loss
        loss = -torch.mean(torch.min(surr1, surr2)) + kl_coeff * kl
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    return {"loss": loss.item(), "kl": kl.item(), "adv_mean": np.mean(advantages)}
```

### 3.3 Creator Prompt Template

```python
# prompts/creator.py
# ──────────────────────────────────────────────────────────────
# The creator's system prompt and interaction template.
# ──────────────────────────────────────────────────────────────

CREATOR_SYSTEM_PROMPT = """You are a problem CREATOR in a self-play reinforcement learning system.

Your job is to generate a SUITE of problems spanning the full difficulty gradient:
- Problem 1 (difficulty 0.0): trivially solvable
- Problem 2 (difficulty 0.25): easy
- Problem 3 (difficulty 0.5): medium
- Problem 4 (difficulty 0.75): hard
- Problem 5 (difficulty 1.0): at the edge of solvability

For EACH problem, you MUST provide:
1. The problem statement
2. The complete solution
3. A step-by-step explanation
4. The estimated difficulty (0.0-1.0)
5. Which tools are needed to solve it

IMPORTANT:
- You will be rewarded for creating problems that produce a LINEAR solve-rate curve
  (100% → 0% across the difficulty gradient). This means you must accurately 
  calibrate difficulty to the solver's capability.
- You will be penalized for providing solutions that are inconsistent with 
  verification. Always double-check your solutions.
- Each oracle query costs you reward. Use tools and your own reasoning first.
- Problems should be diverse and interesting, not repetitive.

Available tools:
- knowledge_oracle(question): Query frozen knowledge base for facts
- code_executor(code): Run code to verify computations
- search(query): Search for relevant information

Output your suite in the structured JSON format specified."""
```

### 3.4 Solver Prompt Template

```python
# prompts/solver.py
# ──────────────────────────────────────────────────────────────
# The solver's system prompt and interaction template.
# ──────────────────────────────────────────────────────────────

SOLVER_SYSTEM_PROMPT = """You are a problem SOLVER in a self-play reinforcement learning system.

You will be given a suite of problems at varying difficulty levels.
For each problem, attempt to solve it using your reasoning and available tools.

You will be rewarded for:
1. Solving problems (direct reward per solved problem)
2. Solving with fewer steps and tool calls (efficiency bonus)

Available tools:
- knowledge_oracle(question): Query frozen knowledge base for facts
- code_executor(code): Run code to verify computations
- search(query): Search for relevant information
- verifier(problem, solution): Verify a proposed solution

IMPORTANT:
- Each oracle query incurs a cost (penalty to the problem CREATOR, not you)
- Attempt all problems, even the hardest ones
- Show your reasoning step by step
- Use tools strategically — they help but have costs"""
```

---

## 4. File Structure

```
selfplay-rlvr/
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── base.yaml              # Default config
│   ├── small.yaml             # Quick experiment (smaller model)
│   └── full.yaml              # Full training run
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py            # FrozenBaseModel, IndependentModel
│   │   └── lora_adapter.py    # LoRA weight management
│   ├── roles/
│   │   ├── __init__.py
│   │   └── manager.py         # RoleManager, swap logic
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── oracle.py          # KnowledgeOracle
│   │   ├── code_executor.py   # Code execution sandbox
│   │   └── search.py          # Web search tool
│   ├── problems/
│   │   ├── __init__.py
│   │   ├── schema.py          # Problem, ProblemSuite
│   │   └── generator.py       # Parse/format problem output
│   ├── rewards/
│   │   ├── __init__.py
│   │   └── engine.py          # RewardEngine, RewardResult
│   ├── verifiers/
│   │   ├── __init__.py
│   │   ├── base.py            # Verifier base class
│   │   ├── math.py            # SymPy-based math verification
│   │   ├── code.py            # Code execution verification
│   │   └── logic.py           # Z3-based logic verification
│   ├── replay/
│   │   ├── __init__.py
│   │   └── buffer.py          # ReplayBuffer
│   ├── train/
│   │   ├── __init__.py
│   │   ├── main_loop.py       # SelfPlayTrainer
│   │   └── grpo.py            # GRPO update step
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── creator.py         # Creator system prompt
│   │   └── solver.py          # Solver system prompt
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logging.py         # W&B integration
│   │   ├── checkpoint.py      # Model checkpointing
│   │   └── metrics.py         # Tracking utilities
│   └── cli.py                 # CLI entry point
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_rewards.py
│   ├── test_verifiers.py
│   ├── test_replay_buffer.py
│   ├── test_roles.py
│   └── test_grpo.py
├── checkpoints/
│   ├── model_a/
│   │   ├── lora_r64/
│   │   └── lora_r128/
│   └── model_b/
│       ├── lora_r64/
│       └── lora_r128/
├── logs/
│   ├── wandb/
│   └── training/
├── data/
│   ├── init_suites/          # Initial problem suites for warmup
│   └── checkpoints/           # Replay buffer checkpoints
├── scripts/
│   ├── setup.sh               # Environment setup
│   ├── download_models.sh     # Download base models
│   └── run_training.sh        # Training launcher
└── docs/
    ├── architecture.md        # This document
    ├── api.md                 # API reference
    └── training_guide.md      # How to train
```

---

## 5. Configuration

```yaml
# configs/base.yaml

# Model settings
model:
  base_model: "Qwen2.5-14B-Instruct"
  base_model_path: "./models/qwen2.5-14b-instruct-Q4_K_M.gguf"
  lora_rank: 64
  lora_alpha: 16
  quantization: "q4_k_m"
  device: "mps"  # Apple Silicon
  max_context: 8192

# Training settings
training:
  total_iterations: 10000
  batch_size: 4
  grpo_group_size: 8
  learning_rate: 1.0e-5
  kl_coeff: 0.01
  clip_range: 0.2
  gradient_accumulation: 4
  
  # GRPO-specific
  grpo:
    advantage_normalization: true
    response_length_penalty: 0.0
    ref_free: false  # Use reference model for KL

# Role swap settings
roles:
  swap_interval: 500        # iterations
  injection_rate: 0.10      # 10% weight injection
  warmup_iterations: 100    # no swaps during warmup
  swap_logging: true

# Reward settings
rewards:
  lambda_gradient: 1.0
  lambda_consistency: 0.5
  lambda_oracle_cost: 0.1
  lambda_solve: 1.0
  lambda_efficiency: 0.2
  target_solve_curve: [1.0, 0.75, 0.5, 0.25, 0.0]
  reward_clipping: 10.0

# Oracle settings
oracle:
  enabled: true
  base_model_path: "./models/qwen2.5-14b-instruct-Q8_0.gguf"
  cost_per_query: 0.1
  max_queries_per_turn: 5
  temperature: 0.1

# Replay buffer
replay:
  max_size: 100000
  priority_scale: 0.7
  min_batch_size: 32
  priority_decay_days: 30

# Logging
logging:
  wandb_project: "selfplay-rlvr"
  wandb_entity: null
  checkpoint_interval: 500
  metric_log_interval: 10
  save_suite_examples: true
  num_examples: 5

# Problem generation
problems:
  min_suite_size: 5
  max_suite_size: 10
  max_difficulty: 1.0
  domains: ["math", "logic", "programming", "reasoning"]
  warmup_from_dataset: true
  warmup_dataset_path: "./data/init_suites/"
```

---

## 6. Training Phases

### Phase 1: Warmup (Iterations 0-100)
- Load frozen base model
- Initialize both LoRA adapters to zeros (or small random)
- Load warmup problem suites from dataset (if available)
- Run creator/solver with random or seeded prompts
- Collect initial statistics on solve rates
- **No gradient updates** — just establish baselines

### Phase 2: Early Training (Iterations 100-1000)
- Start GRPO updates on both models
- Creator learns to calibrate difficulty
- Solver learns problem-solving strategies
- First role swap at iteration 500
- Monitor for: proposer collapse, mode collapse, reward hacking

### Phase 3: Specialization (Iterations 1000-5000)
- Models begin specializing in their roles
- Role swaps inject solver heuristics into creator
- Difficulty calibration should improve
- Track: solve rate curve linearity, oracle usage, consistency scores

### Phase 4: Convergence (Iterations 5000-10000)
- Models should reach stable specialization
- Creator produces well-calibrated difficulty gradients
- Solver solves harder problems efficiently
- Oracle usage should decrease (self-reliance increases)

---

## 7. Failure Modes & Mitigations

| Failure Mode | Description | Mitigation |
|-------------|-------------|------------|
| **Proposer collapse** | Creator generates only trivial problems | Difficulty gradient reward forces diversity; gradient penalty on concentrated difficulty |
| **Mode collapse** | Both models converge to same strategy | Independent weights + role swap injection + diverse domain sampling |
| **Reward hacking** | Creator generates unsolvable problems with fake solutions | Consistency check via external verifier; penalize incoherence |
| **Oracle dependency** | Creator relies on oracle for everything | Oracle cost tax in reward; cap max queries per turn |
| **Knowledge degradation** | RL fine-tuning erases pretrained knowledge | Oracle provides stable epistemic floor; KL penalty to reference |
| **Overfitting to verifier** | Creator crafts problems that pass verifier but aren't meaningful | Diverse domain sampling; consistency with solver's independent solution |
| **Weight collapse after swap** | Injection rate too high, creator loses identity | Limited injection rate (5-15%); gradual blending, not full replacement |

---

## 8. Monitoring & Diagnostics

### Key Metrics to Track

```python
# Per-iteration metrics
metrics = {
    # Creator metrics
    "creator/oracle_queries": int,
    "creator/oracle_cost": float,
    "creator/avg_difficulty": float,
    "creator/suite_diversity": float,  # Domain entropy
    "creator/solution_consistency": float,  # 0-1 score
    
    # Solver metrics
    "solver/problems_solved": int,
    "solver/solve_rate_by_difficulty": list[float],  # Bin by difficulty
    "solver/avg_steps": float,
    "solver/tool_usage_distribution": dict,
    
    # Curriculum metrics
    "curriculum/solve_rate_curve": list[float],  # Actual vs target
    "curriculum/curve_mse": float,
    "curriculum/difficulty_entropy": float,
    
    # Model health
    "model_a/kl_divergence": float,
    "model_b/kl_divergence": float,
    "model_a/gradient_norm": float,
    "model_b/gradient_norm": float,
    
    # Role swap
    "swap/injection_rate": float,
    "swap/last_swap_iteration": int,
    "swap/weight_distance_a": float,  # L2 distance from init
    "swap/weight_distance_b": float,
}
```

### W&B Dashboard Layout

```
┌─────────────────────────────────────────────────────────────┐
│  EXPERIMENT: selfplay-rlvr / run-001                        │
├─────────────────────────────────────────────────────────────┤
│  [Line Chart] Loss curves (creator, solver, total)         │
│  [Line Chart] KL divergence (model A, model B)             │
├─────────────────────────────────────────────────────────────┤
│  [Line Chart] Solve rate by difficulty (actual vs target)  │
│  [Line Chart] Oracle queries per iteration                 │
├─────────────────────────────────────────────────────────────┤
│  [Scatter] Problem difficulty vs solve rate                │
│  [Histogram] Solution consistency scores                   │
├─────────────────────────────────────────────────────────────┤
│  [Table] Latest problem suites (creator output)            │
│  [Table] Latest solver attempts                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. Hardware Considerations (MacBook M5 32GB)

### Memory Budget
```
Component                    Memory (GB)
───────────────────────────────────────
Qwen2.5-14B Q4_K_M (GGUF)    ~8.0
Qwen2.5-14B Q8_0 (oracle)    ~14.0
LoRA adapters (x2)             ~0.2
Python + dependencies          ~2.0
OS + overhead                  ~8.0
───────────────────────────────────────
Total                         ~32.2
```

### Recommendations
1. **Use Q4_K_M for both inference and oracle** — saves ~6GB vs Q8_0
2. **Run oracle and inference models on different threads** — llama.cpp supports multi-model
3. **Reduce GRPO group size to 4** — balance memory vs gradient quality
4. **Use 4-bit LoRA** — further reduces adapter memory
5. **Disable W&B during training if memory constrained** — use local logging
6. **Monitor MPS memory** — Apple Silicon uses unified memory; watch for OOM

### Alternative: Two-Stage Training
If 32GB is too tight for both models simultaneously:
1. Train Model A first (frozen Model B as oracle)
2. Swap and train Model B (frozen Model A as oracle)
3. Resume full dual-model training

---

## 10. Implementation Priority

### Sprint 1: Foundation (Week 1)
1. Project scaffolding (pyproject, config, file structure)
2. Frozen base model loader (GGUF via llama.cpp)
3. LoRA adapter management
4. Problem/suite data schema
5. Unit tests for core data structures

### Sprint 2: Core Components (Week 2)
6. Knowledge oracle tool
7. External verifier (math + code)
8. Replay buffer with priority sampling
9. Reward engine (all three signals)
10. Unit tests for rewards and verifiers

### Sprint 3: Training Loop (Week 3)
11. GRPO implementation
12. Creator prompt + tool-use integration
13. Solver prompt + tool-use integration
14. Role manager with swap logic
15. Main training loop integration

### Sprint 4: Integration & Monitoring (Week 4)
16. W&B integration
17. Checkpointing system
18. CLI entry point
19. End-to-end test run
20. Documentation

---

## 11. Open Questions

1. **Base model choice**: Qwen2.5-14B vs Llama-3.1-13B vs Mistral-Nemo-12B?
   - Qwen2.5 has better tool-use and math capabilities
   - Llama-3.1 has wider ecosystem support
   - Recommendation: Qwen2.5-14B for reasoning quality

2. **GRPO vs PPO**: GRPO is simpler and doesn't need a critic, but PPO has more mature implementations. Recommendation: GRPO for simplicity, switch to PPO if gradient stability becomes an issue.

3. **Oracle model**: Should the oracle be the same model as the base (just frozen)? Or a larger/fine-tuned model?
   - Same model: simpler, no extra download
   - Larger model: more accurate answers, higher cost
   - Recommendation: Same model, Q8_0 quantization for accuracy

4. **Tool-use format**: Function calling (OpenAI-style) or ReAct-style text prompts?
   - Function calling: cleaner, structured
   - ReAct: more flexible, model-agnostic
   - Recommendation: Function calling if using Qwen2.5 (native support)

5. **Difficulty calibration**: How to measure actual solve rate during training?
   - Use the solver's own solve rate as a proxy
   - Requires enough training iterations for the solver to be a reliable judge
   - Consider a "calibration phase" where solve rates are collected before gradient reward kicks in

6. **Swap injection mechanism**: Partial blending vs full replacement vs momentum?
   - Partial blending (current design): smooth transition, preserves identity
   - Full replacement: clean break, but loses creator's accumulated knowledge
   - Recommendation: Partial blending at 5-15%

7. **Problem domains**: What domains should the system generate problems in?
   - Math, logic, programming are the core trio
   - Additional: reasoning puzzles, code debugging, algorithm design
   - Recommendation: Start with math + programming, add logic in Sprint 2

---

## 12. Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|-----------|------------|
| 32GB RAM insufficient for dual-model training | High | Medium | Two-stage training mode; reduce batch/group sizes |
| GRPO convergence instability | Medium | Medium | Start with PPO (more stable); KL coefficient tuning |
| Creator generates degenerate problems | High | High | Strong consistency penalty; verifier integration |
| Oracle answers are too helpful (short-circuits learning) | Medium | High | High oracle cost; query limits |
| Role swap causes catastrophic forgetting | Medium | Low | Low injection rate; KL penalty to reference |
| Reward hacking (generator finds loopholes) | High | Medium | Multiple orthogonal rewards; periodic manual review |
| Slow inference on Apple Silicon | Low | High | GGUF quantization; batch inference; vLLM if available |

---

*Design spec complete. Ready for implementation.*
