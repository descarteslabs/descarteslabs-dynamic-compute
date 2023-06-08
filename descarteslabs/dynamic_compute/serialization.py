"""Module for serialization and deserialization of dynamic compute types"""

from __future__ import annotations

import dataclasses
import json
from typing import Dict


class DataclassJSONEncoder(json.JSONEncoder):
    """Custom josn encoder for dataclasses"""

    def default(self, obj):
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        return super().default(obj)


@dataclasses.dataclass
class BaseSerializationModel:
    """Base serialization model for dynamic compute types"""

    def json(self) -> str:
        return json.dumps(self, cls=DataclassJSONEncoder)

    def dict(self) -> Dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, data: str) -> BaseSerializationModel:
        return cls(**json.loads(data))
