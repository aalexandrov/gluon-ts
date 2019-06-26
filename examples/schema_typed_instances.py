from gluonts.dataset.schema_typed import StartField, TensorField, DatasetSchema
from gluonts.core.serde import dump_json, dump_code
from typing import NamedTuple


def construct_schema_01():
    # default schema
    return DatasetSchema()


def construct_schema_02():
    # custom start field
    return DatasetSchema(start=StartField('start_date'))


def construct_schema_03():
    # custom start field
    return DatasetSchema(start=StartField('start_date'))


def construct_schema_04():
    # custom features
    return DatasetSchema(
        feat_dynamic_cat=[TensorField('is_holiday')],
        feat_static_cat=[TensorField('product_category')],
    )


if __name__ == '__main__':
    print(dump_json(construct_schema_01(), indent=4))
