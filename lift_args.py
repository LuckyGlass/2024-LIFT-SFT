from dataclasses import dataclass, field
from typing import Optional

@dataclass
class LIFTDataArguments:
    data_path: str = field(default="", metadata={"help": "Path to the training data."})
    len_segment: int = field(default=2)
    len_offset: int = field(default=1)
    block_size: int = field(default=1024)
    input_cache_path: Optional[str] = field(default=None)
    ignore_index: int = field(default=-1)
