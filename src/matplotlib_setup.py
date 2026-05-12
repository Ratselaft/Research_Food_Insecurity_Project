"""Small Matplotlib setup shared by the analysis scripts."""

import os


def use_project_matplotlib_config():
    """Use a writable project-local Matplotlib config/cache folder."""
    project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_folder = os.path.join(project_folder, "outputs", "matplotlib_config")

    os.makedirs(config_folder, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", config_folder)

