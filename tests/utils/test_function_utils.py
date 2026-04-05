import datetime

import numpy as np
import pytest
import torch

from hirad.utils.function_utils import (
    InfiniteSampler,
    StackedRandomGenerator,
    get_time_from_range,
    time_range,
)


############################################################################
#                              time_range                                  #
############################################################################


class TestTimeRange:
    """Tests for the time_range generator."""

    def test_basic_hourly_range(self):
        start = datetime.datetime(2024, 1, 1, 0, 0)
        end = datetime.datetime(2024, 1, 1, 3, 0)
        step = datetime.timedelta(hours=1)
        result = list(time_range(start, end, step))
        assert result == [
            datetime.datetime(2024, 1, 1, 0, 0),
            datetime.datetime(2024, 1, 1, 1, 0),
            datetime.datetime(2024, 1, 1, 2, 0),
        ]

    def test_inclusive_range(self):
        start = datetime.datetime(2024, 1, 1, 0, 0)
        end = datetime.datetime(2024, 1, 1, 3, 0)
        step = datetime.timedelta(hours=1)
        result = list(time_range(start, end, step, inclusive=True))
        assert result == [
            datetime.datetime(2024, 1, 1, 0, 0),
            datetime.datetime(2024, 1, 1, 1, 0),
            datetime.datetime(2024, 1, 1, 2, 0),
            datetime.datetime(2024, 1, 1, 3, 0),
        ]

    def test_exclusive_does_not_include_end(self):
        start = datetime.datetime(2024, 6, 1, 12, 0)
        end = datetime.datetime(2024, 6, 1, 14, 0)
        step = datetime.timedelta(hours=1)
        result = list(time_range(start, end, step, inclusive=False))
        assert end not in result

    def test_empty_range_when_start_equals_end(self):
        t = datetime.datetime(2024, 1, 1, 0, 0)
        result = list(time_range(t, t, datetime.timedelta(hours=1)))
        assert result == []

    def test_inclusive_single_element_when_start_equals_end(self):
        t = datetime.datetime(2024, 1, 1, 0, 0)
        result = list(time_range(t, t, datetime.timedelta(hours=1), inclusive=True))
        assert result == [t]

    def test_empty_range_when_start_after_end(self):
        start = datetime.datetime(2024, 1, 2)
        end = datetime.datetime(2024, 1, 1)
        result = list(time_range(start, end, datetime.timedelta(hours=1)))
        assert result == []

    def test_sub_hourly_step(self):
        start = datetime.datetime(2024, 1, 1, 0, 0)
        end = datetime.datetime(2024, 1, 1, 0, 30)
        step = datetime.timedelta(minutes=10)
        result = list(time_range(start, end, step))
        assert len(result) == 3
        assert result == [
            datetime.datetime(2024, 1, 1, 0, 0),
            datetime.datetime(2024, 1, 1, 0, 10),
            datetime.datetime(2024, 1, 1, 0, 20),
        ]

    def test_daily_step(self):
        start = datetime.datetime(2024, 1, 1)
        end = datetime.datetime(2024, 1, 4)
        step = datetime.timedelta(days=1)
        result = list(time_range(start, end, step))
        assert len(result) == 3
        assert result[0] == datetime.datetime(2024, 1, 1)
        assert result[-1] == datetime.datetime(2024, 1, 3)


############################################################################
#                          get_time_from_range                             #
############################################################################


class TestGetTimeFromRange:
    """Tests for get_time_from_range."""

    def test_basic_range_with_default_interval(self):
        times = get_time_from_range(["2024-01-01T00:00:00", "2024-01-01T03:00:00"])
        assert len(times) == 4  # inclusive: 00, 01, 02, 03
        assert times[0] == "2024-01-01T00:00:00"
        assert times[-1] == "2024-01-01T03:00:00"

    def test_custom_interval(self):
        times = get_time_from_range(
            ["2024-01-01T00:00:00", "2024-01-01T06:00:00", 2]
        )
        assert len(times) == 4  # 00, 02, 04, 06
        assert times == [
            "2024-01-01T00:00:00",
            "2024-01-01T02:00:00",
            "2024-01-01T04:00:00",
            "2024-01-01T06:00:00",
        ]

    def test_single_time_when_start_equals_end(self):
        times = get_time_from_range(["2024-06-15T12:00:00", "2024-06-15T12:00:00"])
        assert times == ["2024-06-15T12:00:00"]

    def test_multi_day_range(self):
        times = get_time_from_range(
            ["2024-01-01T00:00:00", "2024-01-02T00:00:00", 6]
        )
        assert len(times) == 5  # 00, 06, 12, 18, 00+1day
        assert times[-2] == "2024-01-01T18:00:00"

    def test_custom_time_format(self):
        fmt = "%Y%m%d-%H%M"
        times = get_time_from_range(["20240101-0000", "20240101-0300"], time_format=fmt)
        assert len(times) == 4
        assert times[0] == "20240101-0000"
        assert times[-1] == "20240101-0300"

    def test_returns_strings(self):
        times = get_time_from_range(["2024-01-01T00:00:00", "2024-01-01T02:00:00"])
        assert all(isinstance(t, str) for t in times)


############################################################################
#                        StackedRandomGenerator                            #
############################################################################


class TestStackedRandomGenerator:
    """Tests for StackedRandomGenerator."""

    def test_randn_shape(self):
        gen = StackedRandomGenerator(device="cpu", seeds=[1, 2, 3])
        out = gen.randn([3, 4, 5])
        assert out.shape == (3, 4, 5)

    def test_randn_batch_mismatch_raises(self):
        gen = StackedRandomGenerator(device="cpu", seeds=[1, 2])
        with pytest.raises(ValueError, match="Expected first dimension"):
            gen.randn([5, 4])

    def test_randn_reproducibility(self):
        gen1 = StackedRandomGenerator(device="cpu", seeds=[42, 99])
        gen2 = StackedRandomGenerator(device="cpu", seeds=[42, 99])
        out1 = gen1.randn([2, 8])
        out2 = gen2.randn([2, 8])
        assert torch.allclose(out1, out2)

    def test_randn_different_seeds_give_different_output(self):
        gen1 = StackedRandomGenerator(device="cpu", seeds=[1, 2])
        gen2 = StackedRandomGenerator(device="cpu", seeds=[3, 4])
        out1 = gen1.randn([2, 100])
        out2 = gen2.randn([2, 100])
        assert not torch.allclose(out1, out2)

    def test_randn_like(self):
        gen = StackedRandomGenerator(device="cpu", seeds=[10, 20])
        template = torch.zeros(2, 3, 4)
        out = gen.randn_like(template)
        assert out.shape == template.shape
        assert out.dtype == template.dtype

    def test_randint_shape(self):
        gen = StackedRandomGenerator(device="cpu", seeds=[1, 2, 3])
        out = gen.randint(0, 10, size=[3, 5])
        assert out.shape == (3, 5)

    def test_randint_batch_mismatch_raises(self):
        gen = StackedRandomGenerator(device="cpu", seeds=[1])
        with pytest.raises(ValueError, match="Expected first dimension"):
            gen.randint(0, 10, size=[4, 5])

    def test_randint_values_in_range(self):
        gen = StackedRandomGenerator(device="cpu", seeds=[7, 8])
        out = gen.randint(0, 5, size=[2, 100])
        assert (out >= 0).all()
        assert (out < 5).all()

    def test_randint_reproducibility(self):
        gen1 = StackedRandomGenerator(device="cpu", seeds=[42, 99])
        gen2 = StackedRandomGenerator(device="cpu", seeds=[42, 99])
        out1 = gen1.randint(0, 100, size=[2, 50])
        out2 = gen2.randint(0, 100, size=[2, 50])
        assert torch.equal(out1, out2)

    def test_randint_different_seeds_give_different_output(self):
        gen1 = StackedRandomGenerator(device="cpu", seeds=[1, 2])
        gen2 = StackedRandomGenerator(device="cpu", seeds=[3, 4])
        out1 = gen1.randint(0, 100, size=[2, 50])
        out2 = gen2.randint(0, 100, size=[2, 50])
        assert not torch.equal(out1, out2)

    def test_single_seed(self):
        gen = StackedRandomGenerator(device="cpu", seeds=[0])
        out = gen.randn([1, 10])
        assert out.shape == (1, 10)


############################################################################
#                           InfiniteSampler                                #
############################################################################


class TestInfiniteSampler:
    """Tests for InfiniteSampler."""

    @pytest.fixture
    def simple_dataset(self):
        """A minimal dataset with 10 items."""
        return list(range(10))

    def test_yields_indices(self, simple_dataset):
        sampler = InfiniteSampler(simple_dataset, shuffle=False)
        it = iter(sampler)
        indices = [next(it) for _ in range(10)]
        assert indices == list(range(10))

    def test_infinite_iteration(self, simple_dataset):
        sampler = InfiniteSampler(simple_dataset, shuffle=False)
        it = iter(sampler)
        # Should be able to draw more samples than the dataset size
        indices = [next(it) for _ in range(25)]
        assert len(indices) == 25

    def test_loops_over_dataset(self, simple_dataset):
        sampler = InfiniteSampler(simple_dataset, shuffle=False)
        it = iter(sampler)
        first_pass = [next(it) for _ in range(10)]
        second_pass = [next(it) for _ in range(10)]
        assert first_pass == list(range(10))
        assert second_pass == list(range(10))

    def test_shuffle_produces_different_order(self, simple_dataset):
        sampler = InfiniteSampler(simple_dataset, shuffle=True, seed=42)
        it = iter(sampler)
        indices = [next(it) for _ in range(10)]
        # With shuffling, the indices should not be in sorted order
        # (extremely unlikely for seed=42 with 10 items)
        assert indices != list(range(10))

    def test_going_through_full_dataset_with_shuffle(self, simple_dataset):
        sampler = InfiniteSampler(simple_dataset, shuffle=True, seed=123)
        it = iter(sampler)
        seen = set()
        for _ in range(10):
            idx = next(it)
            assert idx not in seen  # should see each index once before repeats
            seen.add(idx)

    def test_seed_reproducibility(self, simple_dataset):
        sampler1 = InfiniteSampler(simple_dataset, shuffle=True, seed=123)
        sampler2 = InfiniteSampler(simple_dataset, shuffle=True, seed=123)
        it1 = iter(sampler1)
        it2 = iter(sampler2)
        for _ in range(30):
            assert next(it1) == next(it2)

    def test_different_seeds_different_order(self, simple_dataset):
        sampler1 = InfiniteSampler(simple_dataset, shuffle=True, seed=1)
        sampler2 = InfiniteSampler(simple_dataset, shuffle=True, seed=999)
        it1 = iter(sampler1)
        it2 = iter(sampler2)
        seq1 = [next(it1) for _ in range(20)]
        seq2 = [next(it2) for _ in range(20)]
        assert seq1 != seq2

    def test_distributed_sampling(self, simple_dataset):
        """Each rank should yield non-overlapping indices."""
        sampler0 = InfiniteSampler(
            simple_dataset, rank=0, num_replicas=2, shuffle=False
        )
        sampler1 = InfiniteSampler(
            simple_dataset, rank=1, num_replicas=2, shuffle=False
        )
        it0 = iter(sampler0)
        it1 = iter(sampler1)
        indices0 = [next(it0) for _ in range(5)]
        indices1 = [next(it1) for _ in range(5)]
        # The two ranks should receive different indices
        assert set(indices0) != set(indices1)

    def test_start_idx(self, simple_dataset):
        sampler_default = InfiniteSampler(simple_dataset, shuffle=False, start_idx=0)
        sampler_offset = InfiniteSampler(simple_dataset, shuffle=False, start_idx=5)
        it_default = iter(sampler_default)
        it_offset = iter(sampler_offset)
        # Skip the first 5 from the default sampler
        for _ in range(5):
            next(it_default)
        # Now they should be aligned
        for _ in range(10):
            assert next(it_default) == next(it_offset)

    def test_start_idx_larger_than_dataset_size(self, simple_dataset):
        sampler = InfiniteSampler(simple_dataset, shuffle=False, start_idx=12)
        it = iter(sampler)
        # start_idx=12 should wrap around to index 2 on the first yield
        assert next(it) == 2

    def test_window_size_zero_no_shuffle_effect(self, simple_dataset):
        sampler = InfiniteSampler(
            simple_dataset, shuffle=True, seed=42, window_size=0.0
        )
        it = iter(sampler)
        # With window_size=0, window rounds to 0, so no swapping occurs.
        # Items come out in seed-shuffled initial order but stay fixed.
        indices = [next(it) for _ in range(10)]
        assert len(set(indices)) == 10  # all unique in first pass

    # --- Validation tests ---

    def test_empty_dataset_raises(self):
        with pytest.raises(ValueError, match="at least one item"):
            InfiniteSampler([])

    def test_invalid_num_replicas_raises(self, simple_dataset):
        with pytest.raises(ValueError, match="num_replicas must be positive"):
            InfiniteSampler(simple_dataset, num_replicas=0)

    def test_invalid_rank_raises(self, simple_dataset):
        with pytest.raises(ValueError, match="rank must be non-negative"):
            InfiniteSampler(simple_dataset, rank=-1, num_replicas=2)

    def test_rank_exceeds_replicas_raises(self, simple_dataset):
        with pytest.raises(ValueError, match="rank must be non-negative"):
            InfiniteSampler(simple_dataset, rank=3, num_replicas=2)

    def test_invalid_window_size_raises(self, simple_dataset):
        with pytest.raises(ValueError, match="window_size must be between"):
            InfiniteSampler(simple_dataset, window_size=1.5)

    def test_negative_window_size_raises(self, simple_dataset):
        with pytest.raises(ValueError, match="window_size must be between"):
            InfiniteSampler(simple_dataset, window_size=-0.1)

