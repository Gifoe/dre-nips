import numpy as np

from biodynformer.features import compute_sync_edge


def test_constant_channel_sync_edge_does_not_warn(recwarn):
    segment = np.ones((2, 100), dtype=np.float32)

    edge = compute_sync_edge(segment)

    assert edge.shape == (2, 2)
    assert len(recwarn) == 0
