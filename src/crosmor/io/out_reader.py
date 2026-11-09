"""Parser for legacy CROSMOR .out output files for regression and verification testing."""

from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import numpy as np


class OutBlock:
    """Represents a data block from a CROSMOR .out file."""

    def __init__(
        self,
        block_id: str,
        time: float,
        title: str,
        headers: List[str],
        data: np.ndarray,
        metadata: Dict[str, Any],
    ):
        self.block_id = block_id
        self.time = time
        self.title = title
        self.headers = headers
        self.data = data
        self.metadata = metadata

    @property
    def nrows(self) -> int:
        return self.data.shape[0]

    @property
    def ncols(self) -> int:
        return self.data.shape[1] if self.data.ndim > 1 else 1


class OutFileResult:
    """Parsed content of a complete .out file."""

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        self.blocks: List[OutBlock] = []
        self.times: List[float] = []
        self.balance_metrics: Dict[float, Dict[str, float]] = {}

    def get_blocks_at_time(self, time: float) -> List[OutBlock]:
        return [b for b in self.blocks if abs(b.time - time) < 1e-3]

    def get_block_by_id(self, block_id: str) -> Optional[OutBlock]:
        for b in self.blocks:
            if b.block_id.upper() == block_id.upper():
                return b
        return None


def read_out(file_path: Union[str, Path]) -> OutFileResult:
    """Parse a CROSMOR .out file into structured OutBlock objects."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f".out file not found: {path}")

    result = OutFileResult(path)
    current_time = 0.0
    current_comments: List[str] = []
    metadata: Dict[str, Any] = {}

    with open(path, "r", encoding="latin-1") as f:
        lines = f.readlines()

    idx = 0
    n_lines = len(lines)

    while idx < n_lines:
        line = lines[idx].strip()
        idx += 1

        if not line:
            continue

        if line.startswith("*"):
            # Comment or metadata line
            upper = line.upper()
            if "TIME (S)" in upper:
                try:
                    # e.g. *TIME (S) =           0.0
                    parts = line.split("=")
                    if len(parts) > 1:
                        current_time = float(parts[1].strip().split()[0])
                        if current_time not in result.times:
                            result.times.append(current_time)
                except Exception:
                    pass
            elif "BALANCE" in upper and ":" in line:
                try:
                    val = float(line.split(":")[1].split()[0])
                    if current_time not in result.balance_metrics:
                        result.balance_metrics[current_time] = {}
                    result.balance_metrics[current_time]["balance"] = val
                except Exception:
                    pass
            elif "SCROSS(X=0)" in upper and ":" in line:
                try:
                    val = float(line.split(":")[1].split()[0])
                    if current_time not in result.balance_metrics:
                        result.balance_metrics[current_time] = {}
                    result.balance_metrics[current_time]["scross_x0"] = val
                except Exception:
                    pass
            elif "SCROSS(X=XR)" in upper and ":" in line:
                try:
                    val = float(line.split(":")[1].split()[0])
                    if current_time not in result.balance_metrics:
                        result.balance_metrics[current_time] = {}
                    result.balance_metrics[current_time]["scross_xr"] = val
                except Exception:
                    pass

            current_comments.append(line)
            continue

        # Check if line is a Block ID like B001, B002, etc.
        if (line.startswith("B") or line.startswith("b")) and len(line) <= 6:
            block_id = line
            # Next line usually has nrows and ncols: e.g. " 151 10"
            if idx >= n_lines:
                break
            dim_line = lines[idx].strip()
            idx += 1
            dim_parts = dim_line.split()
            if len(dim_parts) >= 2:
                try:
                    nrows = int(dim_parts[0])
                    ncols = int(dim_parts[1])
                except ValueError:
                    continue

                # Read nrows of numerical data
                data_rows = []
                for _ in range(nrows):
                    if idx >= n_lines:
                        break
                    row_line = lines[idx].strip()
                    idx += 1
                    if row_line.startswith("*"):
                        continue
                    row_vals = [float(v) for v in row_line.replace(",", " ").split()]
                    if row_vals:
                        data_rows.append(row_vals)

                if data_rows:
                    # Handle possible Fortran format mismatches where nrows/ncols had 10 but output has 11
                    max_cols = max(len(r) for r in data_rows)
                    padded_rows = []
                    for r in data_rows:
                        if len(r) < max_cols:
                            padded_rows.append(r + [np.nan] * (max_cols - len(r)))
                        else:
                            padded_rows.append(r[:max_cols])
                    data_arr = np.array(padded_rows, dtype=np.float64)
                else:
                    data_arr = np.empty((0, 0), dtype=np.float64)

                # Deduce headers from last comment lines
                headers = []
                title = ""
                if current_comments:
                    title = current_comments[0]
                    # Look for column header lines (e.g. * X HD HMSL ...)
                    for c in reversed(current_comments):
                        if c.startswith("*") and not c.startswith("*TIME") and not c.startswith("* RHOW"):
                            clean_c = c.lstrip("*").strip()
                            if len(clean_c.split()) >= 3:
                                headers = clean_c.split()
                                break

                block = OutBlock(
                    block_id=block_id,
                    time=current_time,
                    title=title,
                    headers=headers,
                    data=data_arr,
                    metadata=dict(metadata),
                )
                result.blocks.append(block)
                current_comments = []

    return result
