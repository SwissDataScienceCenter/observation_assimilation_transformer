import pandas as pd
from oat_test.conftest import TABLE_PATH

def test_batch_fixture_structure(batch_fixture):
    """
    Test that the test batch fixture matches the shapes and dtype of the
    batch_shapes.pickle table.
    """

    df = pd.read_pickle(TABLE_PATH)

    for group_name in df['group'].unique():
        group_df = df[df['group'] == group_name]
        for _, row in group_df.iterrows():
            key = row['key']
            shape = row['shape']
            dtype = row['dtype']

            assert batch_fixture[group_name][key].shape == shape
            assert batch_fixture[group_name][key].dtype == dtype
            #assert that if the tensor is non-float, it's full of 0s
            if not getattr(dtype, "is_floating_point", False):
                assert (batch_fixture[group_name][key] == 0).all()
