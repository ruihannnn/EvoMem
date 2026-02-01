#!/usr/bin/env python3
"""Command line interface for MemP (Memory-based Practice)."""

import asyncio
import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utu.practice import MemPPractice
from utu.practice.utils import parse_training_free_grpo_config
from utu.utils import get_logger

logger = get_logger(__name__)


def parse_memp_config():
    """Parse MemP configuration from command line arguments."""
    # First parse the base training-free GRPO config
    base_config = parse_training_free_grpo_config()
    
    # Add MemP-specific arguments
    parser = argparse.ArgumentParser(
        description="Run MemP (Memory-based Practice) for agent learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # MemP-specific arguments
    memp_group = parser.add_argument_group("MemP Arguments")
    memp_group.add_argument(
        "--memory_build_policy",
        type=str,
        default="direct",
        choices=["direct", "round"],
        help="Memory build policy: 'direct' for direct workflow generation, 'round' for event-based",
    )
    memp_group.add_argument(
        "--memory_retrieve_policy",
        type=str,
        default="query",
        choices=["query", "facts", "random", "ave_fact"],
        help="Memory retrieval policy",
    )
    memp_group.add_argument(
        "--memory_update_policy",
        type=str,
        default="reflect",
        choices=["vanilla", "validation", "reflect", "none"],
        help=(
            "Memory update policy: 'vanilla' (all), 'validation' (successful only), "
            "'reflect' (with reflection), 'none' (no updates)"
        ),
    )
    memp_group.add_argument(
        "--memory_retrieve_num",
        type=int,
        default=5,
        help="Number of memories to retrieve",
    )
    memp_group.add_argument(
        "--memory_size",
        type=int,
        default=300,
        help="Maximum memory size",
    )
    memp_group.add_argument(
        "--memory_dir",
        type=str,
        default=None,
        help="Directory to store MemP memory (optional)",
    )
    memp_group.add_argument(
        "--memory_traj_file",
        type=str,
        default=None,
        help="Path to trajectory file for cold start (optional)",
    )
    memp_group.add_argument(
        "--memory_cold_start",
        action="store_true",
        help="Enable cold start with existing trajectories",
    )
    memp_group.add_argument(
        "--do_eval",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Whether to run evaluation after practice (true/false)",
    )
    memp_group.add_argument(
        "--eval_use_memory",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Whether to use memory retrieval during evaluation (true/false)",
    )
    
    args, _ = parser.parse_known_args()
    
    # Apply do_eval to base config
    if hasattr(args, 'do_eval'):
        base_config.practice.do_eval = args.do_eval
    
    # Build memory config
    update_enabled = args.memory_update_policy != "none"
    update_policy = "" if args.memory_update_policy == "none" else args.memory_update_policy
    memory_config = {
        "is_cold_start": args.memory_cold_start,
        "policy": {
            "build": args.memory_build_policy,
            "retrieve": args.memory_retrieve_policy,
            "update": update_policy,
        },
        "traj_file_path": args.memory_traj_file,
        "retrieve_num": args.memory_retrieve_num,
        "memory_size": args.memory_size,
        "update_enabled": update_enabled,
        "eval_use_memory": args.eval_use_memory,
    }
    if args.memory_dir:
        memory_config["memory_dir"] = args.memory_dir
    
    return base_config, memory_config


async def main():
    """Run MemP from command line."""
    try:
        # Parse configuration
        config, memory_config = parse_memp_config()
        
        logger.info("=" * 60)
        logger.info("Starting MemP (Memory-based Practice)")
        logger.info("=" * 60)
        logger.info(f"Experiment ID: {config.exp_id}")
        logger.info(f"Practice dataset: {config.data.practice_dataset_name}")
        logger.info(f"Epochs: {config.practice.epochs}")
        logger.info(f"Batch size: {config.practice.batch_size}")
        logger.info(f"Memory build policy: {memory_config['policy']['build']}")
        logger.info(f"Memory retrieve policy: {memory_config['policy']['retrieve']}")
        logger.info(f"Memory update policy: {memory_config['policy']['update']}")
        logger.info(f"Memory retrieve num: {memory_config['retrieve_num']}")
        logger.info(f"Cold start: {memory_config['is_cold_start']}")
        logger.info("=" * 60)
        
        # Initialize MemP
        memp = MemPPractice(config, memory_config)
        
        # Run MemP practice
        result = await memp.run()
        
        logger.info("=" * 60)
        logger.info("MemP practice completed successfully!")
        logger.info(f"Memory-enhanced agent config saved at: {result}")
        logger.info(f"Total memories: {len(memp.memory.documents)}")
        logger.info("=" * 60)
        
        return result
        
    except Exception as e:
        logger.error(f"MemP practice failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
