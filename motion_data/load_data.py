from dataclasses import dataclass
from typing import Tuple

import jax
import numpy as np
import jax.numpy as jnp

@dataclass
class DataSegment:
    name: str
    start_idx: int
    end_idx: int
    start_time: float
    end_time: float

class DataLoader():
    """Class to make loading and working with the motion data easier.

    Can define and name segments of the data to easily access those segments and compute relative times.
    """
    def __init__(self, data_file: str) -> None:
        """Initialize the data loader by loading the data from the given file."""
        self.data_file = data_file

        # Load the data
        self.data = np.loadtxt(data_file, delimiter=',')
        self.data = jnp.array(self.data)
        self.times = jnp.arange(self.data.shape[0])*(1/30.0)     # 30 FPS

        print(f"Loaded data from {data_file}.")
        print(f"Data shape: {self.data.shape}")

        self.segments = []


    def create_segment(self, name: str, start_idx: int, stop_idx: int) -> None:
        """Creates a segment in the data the user can reference later."""
        # Data is at 30Hz from LAFAN1 dataset
        start_time = start_idx/30.0
        end_time = stop_idx/30.0

        self.segments.append(DataSegment(name, start_idx, stop_idx, start_time, end_time))

    def get_config_idx(self, idx: int) -> jax.Array:
        """Returns the configuration of the given index."""
        return self.data[idx]

    def get_config(self, time: float) -> jax.Array:
        """Returns the configuration of the given time."""
        # Just round the time to the nearest index
        idx = jnp.int32(time * 30.0)

        idx = jnp.clip(idx, 0, self.data.shape[0] - 1)

        return self.data[idx]

    def get_config_seg_idx(self, name: str, idx: jax.Array) -> jax.Array:
        """Returns the configuration of the given index relative to the start of the segment."""
        seg = self.get_segment(name)
        return self.data[idx + seg.start_idx]

    def get_config_seg(self, name: str, time: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Returns the configuration of the given time relative to the start of the segment.
        Wraps around."""
        seg = self.get_segment(name)

        # Just round the time to the nearest index
        idx = jnp.int32(time * 30.0)

        # jax.debug.print("idx: {}", idx + seg.start_idx)

        # idx = jnp.clip(idx + seg.start_idx, seg.start_idx, seg.end_idx - 1)

        # jax.debug.print("before idx: {}", idx)

        idx = jnp.mod(idx, seg.end_idx - seg.start_idx)

        # jax.debug.print("after idx: {}", idx)


        return self.data[idx + seg.start_idx], idx + seg.start_idx

    def get_segment(self, name: str) -> DataSegment:
        """Returns a data segment accessed by name."""
        return next(seg for seg in self.segments if seg.name == name)