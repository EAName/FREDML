import abc
import logging
import os

import yaml


class BasePipeline(abc.ABC):
    """
    Abstract base class for all data pipelines.
    Handles config loading, logging, and pipeline orchestration.
    """

    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.logger = self.setup_logger()

    @staticmethod
    def load_config(config_path: str):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def setup_logger(self):
        log_cfg = self.config.get("logging", {})
        log_level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
        log_file = log_cfg.get("file", "pipeline.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        return logging.getLogger(self.__class__.__name__)

    @abc.abstractmethod
    def run(self):
        """Run the pipeline (to be implemented by subclasses)."""
        pass
