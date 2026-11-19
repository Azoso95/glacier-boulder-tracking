"""Resolve survey rasters whether stored locally or referenced in metadata."""
from pathlib import Path
import yaml


def survey_raster(survey: Path) -> Path:
    metadata = survey / 'survey.yaml'
    filename = 'orthomosaic.tif'
    if metadata.is_file():
        filename = yaml.safe_load(metadata.read_text(encoding='utf-8')).get('raster', filename)
    path = (survey / filename).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path
