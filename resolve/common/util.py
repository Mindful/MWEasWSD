from pathlib import Path
from typing import List, Any, Dict, Union

import yaml


def flatten(matrix: List[List[Any]]) -> List[Any]:
    return [x for sublist in matrix for x in sublist]


def yaml_to_dict(filepath: Union[Path, str]) -> Dict[str, Any]:
    with open(filepath, 'rb') as f:
        return yaml.load(f, yaml.FullLoader)
