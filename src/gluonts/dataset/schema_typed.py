# Standard library imports
from typing import List, NamedTuple


class StartField(NamedTuple):
    name: str


class TensorField(NamedTuple):
    name: str


class DatasetSchema(NamedTuple):
    start: StartField = StartField('start')
    target: TensorField = TensorField('target')
    feat_dynamic_cat: List[TensorField] = []
    feat_static_cat: List[TensorField] = []
    feat_dynamic_real: List[TensorField] = []
    feat_static_real: List[TensorField] = []

    def disable_fields(self, feat_names: List[str]) -> 'DatasetSchema':
        raise NotImplementedError
