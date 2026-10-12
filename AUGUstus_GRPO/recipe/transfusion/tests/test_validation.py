# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from recipe.transfusion.validation import (
    _contiguous_shard_indices,
    aggregate_shard_results,
)


def test_contiguous_shard_indices_cover_without_duplicates():
    parts = [_contiguous_shard_indices(11, 3, rank) for rank in range(3)]
    assert parts == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10]]
    assert sorted(sum(parts, [])) == list(range(11))


def test_aggregate_shard_results_exact_means():
    results = [
        {
            "rank": 0,
            "count": 2,
            "seconds": {"generate": 1.0},
            "metric_sums": {"score": 1.0, "pdms_reward": 0.5},
            "metric_counts": {"score": 2, "pdms_reward": 1},
        },
        {
            "rank": 1,
            "count": 1,
            "seconds": {"generate": 2.0},
            "metric_sums": {"score": 2.0, "pdms_reward": 1.0},
            "metric_counts": {"score": 1, "pdms_reward": 1},
        },
    ]

    metrics = aggregate_shard_results(results, data_source="navsim_transfusion")

    assert metrics["val/num_samples"] == 3.0
    assert metrics["val-aux/navsim_transfusion/score/mean"] == 1.0
    assert metrics["val-aux/navsim_transfusion/pdms_reward/mean"] == 0.75
    assert metrics["val-core/navsim_transfusion/score/mean"] == 1.0
    assert metrics["val-core/navsim_transfusion/pdms_reward/mean"] == 0.75
    assert metrics["val-aux/navsim_transfusion/time_generate_sum"] == 3.0
