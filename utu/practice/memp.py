"""
MemP (Memory-based Practice) module for agent learning.

This module integrates the MemP approach into the youtu-agent framework,
providing procedural memory capabilities for agent improvement.
"""

import os
import sys
import json
from typing import Any, Dict, List, Optional

import yaml
from agents import custom_span, function_span, gen_trace_id, trace
from sqlalchemy.orm import make_transient

from ..config import TrainingFreeGRPOConfig
from ..config.eval_config import DataConfig
from ..utils import DIR_ROOT, FileUtils, SQLModelUtils, get_logger
from .data_manager import TrainingFreeGRPODataManager
from .rollout_manager import RolloutManager
from .utils import TaskRecorder

# Add MemP to path
MEMP_PATH = DIR_ROOT / "other-projects" / "MemP"
sys.path.insert(0, str(MEMP_PATH))

# Import MemP components
try:
    from ProcedureMem.memory import Memory
except ImportError as e:
    raise ImportError(
        f"Failed to import MemP components. Please ensure:\n"
        f"1. MemP is located at {MEMP_PATH / 'ProcedureMem'}\n"
        f"2. MemP dependencies are installed: pip install -r {MEMP_PATH / 'requirements.txt'}\n"
        f"Error: {e}"
    )

logger = get_logger(__name__)


class MemPPractice:
    """
    MemP-based agent practice system.
    
    Integrates procedural memory from MemP into the youtu-agent evaluation framework,
    allowing agents to learn from past trajectories and improve performance.
    """
    
    config: TrainingFreeGRPOConfig = None
    practice_rollout_manager: RolloutManager = None
    eval_rollout_manager: RolloutManager = None
    recorder: TaskRecorder = None
    memory: Optional[Memory] = None
    
    def __init__(self, config: TrainingFreeGRPOConfig, memory_config: Optional[Dict[str, Any]] = None):
        """
        Initialize MemP Practice system.
        
        Args:
            config: Training-free GRPO configuration
            memory_config: MemP memory configuration. If None, uses default config.
        """
        self.config = config
        self.recorder = TaskRecorder(experiment_name=config.exp_id)
        self.memory_config = memory_config or self._get_default_memory_config()
        self.eval_use_memory = self.memory_config.get("eval_use_memory", False)
        update_policy = self.memory_config.get("policy", {}).get("update")
        self.memory_update_enabled = self.memory_config.get("update_enabled", True)
        if not update_policy or update_policy == "none":
            self.memory_update_enabled = False
        self.prompts = FileUtils.load_prompts("practice/processor.yaml")
        
    def _get_default_memory_config(self) -> Dict[str, Any]:
        """Get default MemP memory configuration."""
        memory_dir = DIR_ROOT / "workspace" / "memp_memory" / self.config.exp_id
        os.makedirs(memory_dir, exist_ok=True)
        
        return {
            "is_cold_start": False,  # Online learning mode - no pre-existing trajectories
            "policy": {
                "build": "direct",  # Use direct workflow generation
                "retrieve": "query",  # Retrieve by query similarity
                "update": "reflect",  # Use reflection to adjust failed workflows
            },
            "traj_file_path": None,  # No trajectory file in online learning mode
            "retrieve_num": 5,
            "memory_dir": str(memory_dir),
            "memory_size": 300,
        }
        
    async def run(self) -> str:
        """
        Run the complete MemP practice process.
        
        Returns:
            str: Path to the agent configuration file with memory-enhanced instructions
        """
        logger.info("Starting MemP practice...")
        
        # Stage 0: Build components
        if self.practice_rollout_manager is None:
            logger.info("Stage 0: Building MemP components...")
            await self.build()
            
        try:
            # Stage 1: Initialize memory
            logger.info("Stage 1: Initializing procedural memory...")
            self._initialize_memory()
            
            # Stage 2: Run practice with memory
            logger.info("Stage 2: Running practice with procedural memory...")
            await self.practice()
            
            # Stage 3: Create agent config with memory
            logger.info("Stage 3: Creating memory-enhanced agent config...")
            agent_config_path = self._create_agent_config_with_memory()
            
            return agent_config_path
            
        except Exception as e:
            logger.error(f"Error during MemP practice: {e}", exc_info=True)
            raise
            
    async def build(self):
        """Build all components needed for MemP practice."""
        
        # 1. Load dataset
        data_manager = TrainingFreeGRPODataManager(self.config.evaluation)
        
        # Check practice dataset exists
        if not data_manager.check_dataset(self.config.data.practice_dataset_name):
            raise ValueError(
                f"Practice dataset {self.config.data.practice_dataset_name} does not exist in db. "
                f"Please load it first."
            )
            
        # Check eval dataset if needed
        if self.config.evaluation.data.dataset and not data_manager.check_dataset(
            self.config.evaluation.data.dataset
        ):
            raise ValueError(
                f"Evaluation dataset {self.config.evaluation.data.dataset} does not exist in db. "
                f"Please load it first."
            )
            
        # 2. Create practice rollout manager
        practice_eval_config = self.config.evaluation.model_copy()
        practice_eval_config.pass_k = 1  # MemP uses single trajectory per task
        practice_eval_config.data = DataConfig(dataset=self.config.data.practice_dataset_name)
        practice_eval_config.concurrency = self.config.practice.rollout_concurrency
        
        self.practice_rollout_manager = RolloutManager(
            config=practice_eval_config, batch_size=self.config.practice.batch_size
        )
        
        # 3. Create eval rollout manager (if different from practice)
        self.eval_rollout_manager = None
        if self.config.practice.do_eval:
            eval_eval_config = self.config.evaluation.model_copy()
            eval_eval_config.exp_id = eval_eval_config.exp_id + "_eval"
            self.eval_rollout_manager = RolloutManager(
                config=eval_eval_config, batch_size=self.config.practice.batch_size
            )
            
        logger.info("MemP components built successfully")
        
    def _initialize_memory(self):
        """Initialize MemP memory system."""
        logger.info("Initializing MemP memory system...")
        
        # Initialize Memory with config
        self.memory = Memory(**self.memory_config)
        
        # Load existing documents even in online mode (if not cold_start)
        if not self.memory_config["is_cold_start"]:
            documents_path = os.path.join(
                self.memory.memory_dir, 
                self.memory.build_policy, 
                "documents.json"
            )
            if os.path.exists(documents_path):
                try:
                    with open(documents_path, "r") as f:
                        from langchain_core.documents import Document
                        docs_data = json.load(f)
                        self.memory.documents = [Document(**d) for d in docs_data]
                        logger.info(f"Loaded {len(self.memory.documents)} existing documents from {documents_path}")
                        
                        # Rebuild vector store if documents exist
                        if len(self.memory.documents) > 0:
                            self.memory._save_documents()
                            logger.info("Vector store rebuilt from existing documents")
                except Exception as e:
                    logger.warning(f"Failed to load existing documents: {e}")
        
        # Set vector_store to None if no documents
        if len(self.memory.documents) == 0:
            logger.info("Initializing empty memory for online learning mode")
            self.memory.vector_store = None
        
        logger.info(
            f"Memory initialized with {len(self.memory.documents)} documents, "
            f"policy: {self.memory.policy}"
        )
        
    async def practice(self):
        """Run MemP practice process."""
        for epoch in range(self.config.practice.epochs):
            logger.info(f"Start Epoch {epoch}")
            
            # Prepare epoch data
            epoch_data = self.practice_rollout_manager.load_epoch_data(
                epoch, shuffle=self.config.practice.shuffle_data, 
                truncate=self.config.practice.rollout_data_truncate
            )
            
            # Make all samples transient to completely detach from SQLAlchemy session
            # This prevents DetachedInstanceError when accessing attributes later
            for sample in epoch_data:
                make_transient(sample)
            
            # Check the batch size (consistent with training_free_grpo)
            assert len(epoch_data) % self.config.practice.grpo_n == 0, (
                f"Epoch data size {len(epoch_data)} is not divisible by grpo_n {self.config.practice.grpo_n}"
            )
            if len(epoch_data) < self.config.practice.batch_size * self.config.practice.grpo_n:
                raise ValueError(
                    f"Epoch {epoch} data size {len(epoch_data) // self.config.practice.grpo_n} is smaller than "
                    f"batch size {self.config.practice.batch_size}."
                )
            if len(epoch_data) % (self.config.practice.batch_size * self.config.practice.grpo_n) != 0:
                logger.warning(
                    f"Epoch {epoch} data size {len(epoch_data) // self.config.practice.grpo_n} is not divisible by "
                    f"batch size {self.config.practice.batch_size}. Some data will be dropped."
                )
                
            # Inner loop for each batch
            num_batches = len(epoch_data) // (self.config.practice.batch_size * self.config.practice.grpo_n)
            for batch_idx in range(num_batches):
                step = epoch * num_batches + batch_idx
                logger.info(f"Step {step} (Epoch {epoch}, Batch {batch_idx})")
                
                step_trace_id = gen_trace_id()
                with trace(f"[{self.recorder.experiment_name}] Step {step} practice", trace_id=step_trace_id):
                    # Get current stat
                    stats = self.recorder.stats or {}
                    if f"step_{step}" not in stats:
                        stats[f"step_{step}"] = {"epoch": epoch, "batch": batch_idx, "complete": False}
                        
                    # 1. Enhance batch data with memory
                    with custom_span("Enhance batch data with memory"):
                        # Get batch data (consistent with training_free_grpo batch indexing)
                        start_idx = batch_idx * self.config.practice.batch_size * self.config.practice.grpo_n
                        end_idx = (batch_idx + 1) * self.config.practice.batch_size * self.config.practice.grpo_n
                        batch_data = epoch_data[start_idx:end_idx]
                        enhanced_data = self._enhance_batch_with_memory(batch_data)
                        
                    # 2. Rollout batch data
                    with custom_span("Process the batch data"):
                        # Temporarily update the batch data with enhanced prompts
                        original_data = self._update_batch_prompts(batch_data, enhanced_data)
                        
                        # Manually run pipeline phases to avoid preprocessing (which would overwrite augmented_question)
                        # Save augmented_question to database so rollout can use it
                        # Follow training_free_grpo pattern: get samples from db (in session) and update them
                        db_samples = self.practice_rollout_manager._get_batch_samples(
                            batch_idx=batch_idx,
                            stage=None,  # Get all samples regardless of stage
                        )
                        # Create a mapping from id to sample for quick lookup
                        db_sample_map = {s.id: s for s in db_samples if s.id is not None}
                        # Update database samples with enhanced prompts (like training_free_grpo does)
                        for sample in batch_data:
                            if sample.id is not None and sample.id in db_sample_map:
                                # Update the database sample (in session) with new values
                                db_sample = db_sample_map[sample.id]
                                db_sample.augmented_question = sample.augmented_question
                                db_sample.stage = "init"
                                self.practice_rollout_manager.dataset.save(db_sample)
                            else:
                                # New sample or not found, update and save directly
                                sample.stage = "init"
                                self.practice_rollout_manager.dataset.save(sample)
                        
                        # Run rollout and judge
                        with custom_span("Rollout batch samples"):
                            await self.practice_rollout_manager.rollout_batch(batch_idx)
                        with custom_span("Judge batch samples"):
                            await self.practice_rollout_manager.judge_batch(batch_idx)
                        
                        # Get results and calculate stats
                        rollouts = self.practice_rollout_manager._get_batch_samples(batch_idx=batch_idx, stage="judged")
                        batch_stat = await self.practice_rollout_manager.stat_batch(batch_idx)
                        stat = batch_stat[0]["metrics"] if batch_stat else {}
                        
                        # Cleanup
                        await self.practice_rollout_manager.cleanup()
                        
                        # Restore original data
                        self._restore_batch_prompts(batch_data, original_data)
                        
                        stats[f"step_{step}"]["rollout"] = stat
                        
                    # 3. Update memory with rollout results
                    with custom_span("Update memory with rollouts"):
                        self._update_memory_from_rollouts(rollouts, enhanced_data)
                        
                    stats[f"step_{step}"]["complete"] = True
                    self.recorder.stat_update({f"step_{step}": stats[f"step_{step}"]})
                    
                    # 4. Evaluation based on strategy
                    if self.eval_rollout_manager and self._should_evaluate(step, batch_idx, num_batches):
                        eval_trace_id = gen_trace_id()
                        with trace(
                            f"[{self.recorder.experiment_name}] Step {step} evaluation", 
                            trace_id=eval_trace_id
                        ):
                            logger.info(f"Running evaluation at step {step}")
                            eval_data = self.eval_rollout_manager.load_epoch_data(
                                epoch=epoch, shuffle=False, 
                                truncate=self.config.practice.eval_data_truncate
                            )
                            logger.info(f"Evaluation dataset loaded with {len(eval_data)} records")
                            use_cache = self._should_use_cache(step)
                            if self.eval_use_memory:
                                eval_stats = await self._run_eval_with_memory(eval_data, use_cache=use_cache)
                            else:
                                _, eval_stats = await self.eval_rollout_manager.main(
                                    recorder=self.recorder, use_cache=use_cache
                                )
                            
                            with function_span("Record evaluation stats") as eval_stat_span:
                                eval_stat_span.span_data.output = eval_stats
                            stats[f"step_{step}"]["eval"] = eval_stats
                            self.recorder.stat_update({f"step_{step}": stats[f"step_{step}"]})
                            
                    # 5. Record stats to tracing
                    with function_span("Record current stats") as stat_span:
                        stat_span.span_data.output = stats[f"step_{step}"]
                        
    def _enhance_batch_with_memory(self, batch_data: List) -> List[Dict[str, Any]]:
        """
        Enhance batch data with procedural memory.
        
        Args:
            batch_data: List of EvaluationSample objects
            
        Returns:
            List of enhancement info containing retrieved workflows
        """
        enhanced_data = []
        
        for sample in batch_data:
            # Use raw_question or augmented_question from EvaluationSample
            # Note: Attributes are pre-loaded in practice() to avoid DetachedInstanceError
            query = sample.raw_question if hasattr(sample, "raw_question") else str(sample)
            
            # Retrieve relevant workflows from memory
            # Check both documents and vector_store to ensure memory is ready
            if len(self.memory.documents) > 0 and hasattr(self.memory, 'vector_store') and self.memory.vector_store is not None:
                try:
                    retrieved = self.memory.retrieve(query)
                    logger.info("MemP recall hits: %s", len(retrieved))
                    
                    # Format retrieved workflows
                    workflows = []
                    for doc, score in retrieved:
                        workflow = doc.metadata.get("workflow", "")
                        if workflow:
                            workflows.append({
                                "task_name": doc.metadata.get("query", ""),
                                "guidelines": workflow,
                                "score": score,
                            })
                            
                    enhanced_data.append({
                        "sample": sample,
                        "workflows": workflows,
                        "has_memory": len(workflows) > 0,
                    })
                except Exception as e:
                    logger.warning(f"Failed to retrieve memory for query: {e}")
                    enhanced_data.append({
                        "sample": sample,
                        "workflows": [],
                        "has_memory": False,
                    })
            else:
                enhanced_data.append({
                    "sample": sample,
                    "workflows": [],
                    "has_memory": False,
                })
                
        return enhanced_data
        
    def _update_batch_prompts(self, batch_data: List, enhanced_data: List[Dict[str, Any]]) -> List[str]:
        """
        Update batch data with memory-enhanced prompts using template.
        
        Args:
            batch_data: Original batch data (EvaluationSample objects)
            enhanced_data: Enhanced data with memory
            
        Returns:
            List of original augmented_question for restoration
        """
        original_prompts = []
        
        for sample, enhanced in zip(batch_data, enhanced_data):
            # IMPORTANT: Store original BEFORE any modification
            original_prompts.append(
                sample.augmented_question if (hasattr(sample, "augmented_question") and sample.augmented_question) else sample.raw_question
            )
            
            # Add memory guidelines to augmented_question if available
            if enhanced["has_memory"]:
                # Format workflows as a list of guidelines (similar to Training-Free GRPO experiences)
                formatted_workflows = "\n".join([
                    f"[Task: {wf['task_name']}]\n{wf['guidelines']}" 
                    for wf in enhanced["workflows"][:3]
                ])
                
                # Only modify the question if we have valid formatted workflows
                if formatted_workflows.strip():
                    # Use Jinja2 template to format the prompt (consistent with Training-Free GRPO)
                    sample.augmented_question = FileUtils.get_jinja_template_str(
                        self.prompts["PROBLEM_WITH_MEMORY_TEMPLATE"]
                    ).render(
                        problem=sample.raw_question,
                        workflows=formatted_workflows,
                    )
            else:
                sample.augmented_question = sample.raw_question
                    
        return original_prompts
        
    def _restore_batch_prompts(self, batch_data: List, original_prompts: List[str]):
        """Restore original augmented_question after rollout."""
        for sample, original_prompt in zip(batch_data, original_prompts):
            if hasattr(sample, "augmented_question"):
                sample.augmented_question = original_prompt
                
    def _update_memory_from_rollouts(self, rollouts: List, enhanced_data: List[Dict[str, Any]]):
        """
        Update memory with rollout results.
        
        Args:
            rollouts: Rollout results from practice
            enhanced_data: Enhanced data with memory info
        """
        if not self.memory_update_enabled:
            logger.info("Memory update disabled; skipping update step.")
            return
        query_list = []
        trajectory_list = []
        reward_list = []
        workflow_list = []
        memory_list = []
        
        for rollout, enhanced in zip(rollouts, enhanced_data):
            # Extract query from original sample
            sample = enhanced["sample"]
            query = sample.raw_question if hasattr(sample, "raw_question") else str(sample)
            
            # Extract trajectory from rollout (EvaluationSample)
            trajectory = self._extract_trajectory_from_rollout(rollout)
            
            # Extract reward from EvaluationSample (rollout is from recent DB query, may need eager loading)
            try:
                reward = rollout.correct if hasattr(rollout, "correct") and rollout.correct else False
            except Exception:
                reward = False
            
            # Get workflow if memory was used
            workflow = ""
            memory_name = ""
            if enhanced["has_memory"] and len(enhanced["workflows"]) > 0:
                workflow = enhanced["workflows"][0]["guidelines"]
                memory_name = enhanced["workflows"][0]["task_name"]
                
            query_list.append(query)
            trajectory_list.append(trajectory)
            reward_list.append(reward)
            workflow_list.append(workflow)
            memory_list.append(memory_name)
            
        # Update memory
        if len(query_list) > 0:
            try:
                self.memory.update(query_list, trajectory_list, reward_list, workflow_list, memory_list)
                logger.info(f"Memory updated with {len(query_list)} new trajectories")
            except Exception as e:
                logger.error(f"Failed to update memory: {e}", exc_info=True)

    async def _run_eval_with_memory(self, eval_data: List, use_cache: bool) -> dict:
        """Run evaluation with memory-enhanced prompts, without updating memory."""
        if not self.eval_rollout_manager:
            return {}

        if use_cache:
            eval_samples = [sample for sample in eval_data if getattr(sample, "stage", "init") == "init"]
        else:
            eval_samples = eval_data

        if not eval_samples:
            return {}

        enhanced_data = self._enhance_batch_with_memory(eval_samples)
        original_prompts = self._update_batch_prompts(eval_samples, enhanced_data)

        for sample in eval_samples:
            sample.stage = "init"
            self.eval_rollout_manager.dataset.save(sample)

        with custom_span("Eval rollout batch samples"):
            await self.eval_rollout_manager.rollout_batch()
        with custom_span("Eval judge batch samples"):
            await self.eval_rollout_manager.judge_batch()
        eval_stats = await self.eval_rollout_manager.stat_batch()
        await self.eval_rollout_manager.cleanup()

        self._restore_batch_prompts(eval_samples, original_prompts)
        return eval_stats[0]["metrics"] if eval_stats else {}
                
    def _extract_trajectory_from_rollout(self, rollout) -> List[Dict[str, str]]:
        """
        Extract trajectory (messages) from rollout.
        
        Args:
            rollout: EvaluationSample from rollout
            
        Returns:
            List of message dicts with role and content (or string trajectory)
        """
        # EvaluationSample has trajectories field (JSON string)
        if hasattr(rollout, "trajectories") and rollout.trajectories:
            try:
                return json.loads(rollout.trajectories)
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Failed to parse trajectories JSON: {rollout.trajectories}")
                
        # Fallback to trajectory field (deprecated but may still be used)
        if hasattr(rollout, "trajectory") and rollout.trajectory:
            return rollout.trajectory
            
        # Default empty trajectory
        return []
        
    def _should_use_cache(self, step: int) -> bool:
        """Determine if cached results should be used for current step.

        Restart behavior:
        - restart_step=None: Use cache for all steps (if available)
        - restart_step=N: Use cache for steps < N, execute fresh from step N onwards
        - restart_step=0: Execute all steps fresh (no caching)
        """
        restart_step = self.config.practice.restart_step
        return restart_step is None or step < restart_step

    def _should_evaluate(self, total_steps: int, batch_idx: int, num_batches: int) -> bool:
        """Determine if evaluation should be performed at current step."""
        if self.config.practice.eval_strategy == "epoch":
            # Evaluate at the end of each epoch
            return batch_idx == num_batches - 1
        elif self.config.practice.eval_strategy == "steps":
            # Evaluate every eval_steps
            return total_steps % self.config.practice.eval_steps == 0
        return False
        
    def _create_agent_config_with_memory(self) -> str:
        """
        Create agent configuration with memory-enhanced instructions.
        
        Returns:
            Path to the created config file
        """
        # Load the original agent config
        base_config = self.config.evaluation.agent
        config_dict = base_config.model_dump(exclude_none=True)
        
        # Add memory information to instructions
        current_instructions = config_dict.get("agent", {}).get("instructions", "You are a helpful assistant.")
        
        memory_instruction = (
            "\n\nYou have access to procedural memory that contains guidelines from similar tasks. "
            "When solving problems, carefully consider these guidelines and adapt them to the current task. "
            f"The memory system has learned from {len(self.memory.documents)} successful task trajectories."
        )
        
        config_dict["agent"]["instructions"] = current_instructions + memory_instruction
        
        # Remove unnecessary fields
        remain_default_keys = ["type", "model", "agent", "toolkits"]
        for key in list(config_dict.keys()):
            if key not in remain_default_keys:
                del config_dict[key]
                
        # Convert to YAML format
        yaml_config = yaml.dump(config_dict, default_flow_style=False, allow_unicode=True, sort_keys=False)
        config_header = "# @package _global_\ndefaults:\n  - _self_\n\n"
        
        # Save to file
        config_filename = f"{self.config.evaluation.exp_id}_memp_agent.yaml"
        config_dir = str(DIR_ROOT / "configs" / "agents" / "practice")
        full_path = os.path.join(config_dir, os.path.basename(config_filename))
        
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(config_header + yaml_config)
            
        logger.info(f"Memory-enhanced agent config saved to: {full_path}")
        return full_path
