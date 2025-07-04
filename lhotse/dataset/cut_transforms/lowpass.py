import math
import random
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union

from lhotse import CutSet
from lhotse.augmentation.torchaudio import get_resample_backend, set_resample_backend
from lhotse.dataset.dataloading import resolve_seed


@dataclass
class Lowpass:
    """
    Applies a low-pass filter to each Cut in a CutSet.

    The filter is applied with a probability of ``p``. When applied, the filter
    randomly selects a cutoff frequency from the list of provided frequencies,
    with optional weights controlling the selection.

    :param frequencies: A list of cutoff frequencies.
    :param weights: Optional weights for each frequency (default: equal weights).
    :param p: The probability of applying the low-pass filter (default: 0.5).
    :param frequencies: A list of cutoff frequencies.
    :param frequency_weights: Optional weights for each frequency (default: equal weights).
    :param filter_types: List of filter types to use. One or more of: "butter", "cheby1", "cheby2", "ellip", "bessel", "resample" (default: ["butter"]).
    :param filter_type_weights: Optional weights for each filter type (default: equal weights).
    :param order: Filter order. Can be a single value or an interval. If an interval is provided, the value is sampled uniformly.
    :param stopband_attenuation_db: Stopband attenuation in dB. Used for Chebyshev II and Elliptical filters. Can be a single value or an interval. If an interval is provided, the value is sampled uniformly.
    :param ripple_db: Passband ripple in dB. Used for Chebyshev I and Elliptical filters. Can be a single value or an interval. If an interval is provided, the value is sampled uniformly.
    :param randgen: An optional random number generator (default: a new instance).
    :param preserve_id: Whether to preserve the original cut ID (default: False).
    """

    frequencies: List[float]
    weights: Optional[List[float]] = None
    p: float = 0.5
    randgen: random.Random = None
    preserve_id: bool = False

    def __post_init__(self) -> None:
        if self.weights:
            assert len(self.weights) == len(self.frequencies)
        else:
            # all codecs have equal weights by default
            self.weights = [1.0 for _ in self.frequencies]

    def __call__(self, cuts: CutSet) -> CutSet:
        if self.randgen is None:
            self.randgen = random.Random()

        lowpassed_cuts = []
        for cut in cuts:
            if self.randgen.random() <= self.p:
                frequency, *_ = self.randgen.choices(
                    self.frequencies, weights=self.weights
                )

                filter_type, *_ = self.randgen.choices(
                    self.filter_types, weights=self.filter_type_weights
                )

                if isinstance(self.stopband_attenuation_db, tuple):
                    min_db, max_db = self.stopband_attenuation_db
                    stopband_attenuation = self.randgen.uniform(min_db, max_db)
                else:
                    stopband_attenuation = self.stopband_attenuation_db

                if isinstance(self.ripple_db, tuple):
                    min_db, max_db = self.ripple_db
                    ripple = self.randgen.uniform(min_db, max_db)
                else:
                    ripple = self.ripple_db

                if isinstance(self.order, tuple):
                    min_order, max_order = self.order
                    order = self.randgen.randint(min_order, max_order)
                else:
                    order = self.order

                new_cut = cut.lowpass(
                    frequency=frequency,
                    stopband_attenuation_db=stopband_attenuation,
                    ripple_db=ripple,
                    order=order,
                    filter_type=filter_type,
                )
                if not self.preserve_id:
                    new_cut.id = (
                        f"{cut.id}_lowpassed{frequency:.0f}_{filter_type}_{order}"
                    )
                lowpassed_cuts.append(new_cut)
            else:
                lowpassed_cuts.append(cut)

        return CutSet(lowpassed_cuts)


@dataclass
class LowpassUsingResampling:
    """
    Applies a low-pass filter to each Cut in a CutSet by resampling the audio back and forth.
    """

    p: float = 0.5
    frequencies_interval: Tuple[float, float] = (3500, 8000)
    seed: Union[int, Literal["trng", "randomized"]] = 42
    rng: Optional[random.Random] = None
    preserve_id: bool = False
    backend: Literal["default", "sox"] = "default"

    def __post_init__(self) -> None:
        if self.rng is not None and self.seed is not None:
            raise ValueError("Either rng or seed must be provided, not both")
        if self.rng is None:
            self.rng = random.Random(resolve_seed(self.seed))

    def __call__(self, cuts: CutSet) -> CutSet:
        original_backend = get_resample_backend()
        set_resample_backend(self.backend)

        lowpassed_cuts = []
        for cut in cuts:
            if self.rng.random() <= self.p:
                low, high = self.frequencies_interval
                if high > cut.sampling_rate // 2:
                    raise ValueError(
                        f"Upper frequency limit {high} is greater than sampling rate / 2 ({cut.sampling_rate // 2})"
                    )

                # sampling from log-uniform[low, high] distribution
                cutoff_frequency = math.exp(
                    self.rng.uniform(math.log(low), math.log(high))
                )
                cutoff_frequency = int(cutoff_frequency)

                new_cut = cut.resample(cutoff_frequency * 2).resample(cut.sampling_rate)
                if not self.preserve_id:
                    new_cut.id = f"{cut.id}_lowpassed{cutoff_frequency:.0f}"
                lowpassed_cuts.append(new_cut)
            else:
                lowpassed_cuts.append(cut)

        set_resample_backend(original_backend)

        return CutSet(lowpassed_cuts)
