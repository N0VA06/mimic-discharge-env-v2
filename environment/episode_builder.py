# episode_builder.py
# The real MIMIC-IV EpisodeBuilder lives in old_episode_builder.py.
# This module re-exports it so the rest of the codebase imports from a stable name.
from .old_episode_builder import EpisodeBuilder  # noqa: F401

__all__ = ["EpisodeBuilder"]
