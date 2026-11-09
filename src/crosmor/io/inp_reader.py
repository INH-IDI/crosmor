"""Parser for legacy CROSMOR .inp input files matching Fortran 77 list-directed READ semantics."""

import os
from pathlib import Path
from typing import List, Tuple, Union, Optional

from crosmor.models.config import (
    SimulationConfig,
    TidePoint,
    StormPoint,
    BathymetrySection,
    WaveClass,
    WaveTimePoint,
    SedimentProfilePoint,
    HydroParams,
    MorphoParams,
)


class InpParseError(Exception):
    """Custom exception raised when an .inp file cannot be parsed."""
    pass


class FortranRecordReader:
    """Emulates Fortran 77 list-directed (READ(iuni, *)) file reading.

    In Fortran 77:
    - Lines starting with '*' (or empty lines via SKCOMA) are skipped.
    - Each READ statement consumes values and then discards any remaining
      values on the current line.
    - If a READ statement requires more values than present on the current line,
      it continues reading from the next non-comment line.
    """

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        if not self.file_path.is_file():
            raise FileNotFoundError(f"Input file not found: {self.file_path}")

        self.data_lines: List[List[str]] = []
        with open(self.file_path, "r", encoding="latin-1") as f:
            for line in f:
                # Remove DOS EOF (\x1a)
                clean_line = line.replace("\x1a", "").strip()
                if not clean_line or clean_line.startswith("*"):
                    continue
                # Split comma/space separated tokens
                parts = clean_line.replace(",", " ").split()
                if parts:
                    self.data_lines.append(parts)

        self.line_idx = 0
        self.col_idx = 0

    def has_more(self) -> bool:
        return self.line_idx < len(self.data_lines)

    def read_tokens(self, count: int, expected_name: str = "") -> List[str]:
        """Read `count` tokens, advancing across lines if necessary, then end record."""
        tokens = []
        while len(tokens) < count:
            if self.line_idx >= len(self.data_lines):
                raise InpParseError(
                    f"Unexpected end of file while reading '{expected_name}' "
                    f"(expected {count} tokens, got {len(tokens)}) in {self.file_path.name}"
                )
            current_line = self.data_lines[self.line_idx]
            needed = count - len(tokens)
            available = len(current_line) - self.col_idx
            take = min(needed, available)

            tokens.extend(current_line[self.col_idx : self.col_idx + take])
            self.col_idx += take

            if self.col_idx >= len(current_line):
                self.line_idx += 1
                self.col_idx = 0

        # Discard remainder of the last line where READ finished (Fortran record termination)
        if self.col_idx > 0:
            self.line_idx += 1
            self.col_idx = 0

        return tokens

    def read_record_flexible(self, min_count: int, max_count: int, expected_name: str = "") -> List[str]:
        """Read all tokens available on the current line (between min_count and max_count) and finish record."""
        if self.line_idx >= len(self.data_lines):
            raise InpParseError(f"Unexpected end of file while reading '{expected_name}' in {self.file_path.name}")
        current_line = self.data_lines[self.line_idx]
        tokens = current_line[self.col_idx :]
        self.line_idx += 1
        self.col_idx = 0

        if len(tokens) < min_count:
            # Need more tokens from next line
            needed = min_count - len(tokens)
            more = self.read_tokens(needed, expected_name)
            tokens.extend(more)

        return tokens


def read_inp(file_path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None) -> SimulationConfig:
    """Parse a CROSMOR .inp file and return a fully typed SimulationConfig."""
    file_path = Path(file_path)
    if base_dir is None:
        base_dir = file_path.parent
    else:
        base_dir = Path(base_dir)

    reader = FortranRecordReader(file_path)

    # 1. Output options: READ(iuni, *) iprint
    iprint = int(float(reader.read_tokens(1, "iprint")[0]))

    # 2. Tidal conditions: READ(iuni, *) ntide
    ntide = int(float(reader.read_tokens(1, "ntide")[0]))
    tide_series = []
    for i in range(ntide):
        vals = [float(x) for x in reader.read_tokens(4, f"tide[{i}]")]
        tide_series.append(TidePoint(time=vals[0], vnul1=vals[1], vnul2=vals[2], ws=vals[3]))

    # 3. Bathymetry table: READ(iuni, *) nhmsl
    nhmsl = int(float(reader.read_tokens(1, "nhmsl")[0]))
    bathymetry = []
    for i in range(nhmsl - 1):
        vals = reader.read_tokens(3, f"bathymetry[{i}]")
        bathymetry.append(
            BathymetrySection(
                x=float(vals[0]),
                hmsl=float(vals[1]),
                nstap=int(float(vals[2])),
            )
        )
    # Last point has 2 values in Fortran: READ(iuni, *) xmsl(nhmsll), hmsl(nhmsll)
    # (Any trailing token on this line is discarded)
    last_vals = reader.read_tokens(2, "bathymetry_last")
    bathymetry.append(BathymetrySection(x=float(last_vals[0]), hmsl=float(last_vals[1]), nstap=None))

    # 4. Minimum depth and beach module switch: READ(iuni, *) hgrens, istrand
    hg_vals = reader.read_tokens(2, "hgrens, istrand")
    hgrens = float(hg_vals[0])
    istrand = int(float(hg_vals[1]))

    # 5. Storm surge table: READ(iuni, *) nhsto
    nhsto = int(float(reader.read_tokens(1, "nhsto")[0]))
    storm_series = []
    for i in range(nhsto):
        vals = [float(x) for x in reader.read_tokens(2, f"storm[{i}]")]
        storm_series.append(StormPoint(time=vals[0], hstorm=vals[1]))

    # 6. Wave options (igolf = 1, 2, -2): READ(iuni, *) igolf
    igolf = int(float(reader.read_tokens(1, "igolf")[0]))
    wave_classes = []
    wave_series = []
    wave_file = None
    nh_wave_classes = 5

    if igolf == 1:
        nh = int(float(reader.read_tokens(1, "wave_classes.nh")[0]))
        nh_wave_classes = nh
        for i in range(nh):
            if i == 0:
                vals = [float(x) for x in reader.read_tokens(5, "wave_class[0]")]
                wave_classes.append(WaveClass(h=vals[0], t=vals[1], p=vals[2], teta=vals[3], w10=vals[4]))
            else:
                vals = [float(x) for x in reader.read_tokens(4, f"wave_class[{i}]")]
                wave_classes.append(WaveClass(h=vals[0], t=vals[1], p=vals[2], teta=vals[3], w10=0.0))
    elif igolf == 2:
        nh = int(float(reader.read_tokens(1, "rayleigh.nh")[0]))
        nh_wave_classes = nh
        nht = int(float(reader.read_tokens(1, "rayleigh.nht")[0]))
        for i in range(nht):
            vals = [float(x) for x in reader.read_tokens(5, f"wave_series[{i}]")]
            wave_series.append(WaveTimePoint(time=vals[0], hrms=vals[1], t=vals[2], teta=vals[3], w10=vals[4]))
    elif igolf == -2:
        nh = int(float(reader.read_tokens(1, "rayleigh.nh")[0]))
        nh_wave_classes = nh
        wave_file = reader.read_tokens(1, "wave_filename")[0]
        ext_wave_path = base_dir / wave_file
        if not ext_wave_path.is_file():
            ext_wave_path = file_path.parent / wave_file
        if not ext_wave_path.is_file() and file_path.parent.is_dir():
            # Case-insensitive lookup for legacy DOS/Windows files
            for cand in file_path.parent.iterdir():
                if cand.name.lower() == wave_file.lower():
                    ext_wave_path = cand
                    break
        if ext_wave_path.is_file():
            ext_reader = FortranRecordReader(ext_wave_path)
            nht = int(float(ext_reader.read_tokens(1, "nht")[0]))
            for i in range(nht):
                vals = [float(x) for x in ext_reader.read_tokens(5, f"ext_wave_series[{i}]")]
                wave_series.append(WaveTimePoint(time=vals[0], hrms=vals[1], t=vals[2], teta=vals[3], w10=vals[4]))

    # 7. Hydrodynamic parameters (Line 1):
    # READ(iuni, *) kk1, brol, coef6, clh, csw, kk2, kk3, SL, sef1, KL
    h1 = reader.read_tokens(10, "hydro_params_1")
    hydro_line1 = {
        "switch1_asymmetry": int(float(h1[0])),
        "roller_coef": float(h1[1]),
        "coef6": float(h1[2]),
        "clh": float(h1[3]),
        "csw": float(h1[4]),
        "switch2_nielsen": int(float(h1[5])),
        "switch3_phaselag": int(float(h1[6])),
        "sl_dune_slope": float(h1[7]),
        "sef1": float(h1[8]),
        "switch4_longwaves": int(float(h1[9])),
    }

    # 8. Hydrodynamic parameters (Line 2):
    # READ(iuni, *) coefsr, gamma, coef5, els, nit, SW
    h2 = reader.read_tokens(6, "hydro_params_2")
    hydro = HydroParams(
        **hydro_line1,
        coefsr=float(h2[0]),
        gamma=float(h2[1]),
        coef5=float(h2[2]),
        els=float(h2[3]),
        nit=int(float(h2[4])),
        sw=float(h2[5]),
    )

    # 9. Transport and sediment parameters (from LEES)
    # READ(iuni, *) iprant, nout
    tr_out = reader.read_tokens(2, "iprant, nout")
    iprant = int(float(tr_out[0]))
    nout = int(float(tr_out[1]))

    # READ(iuni, *) dl, rhos, por, facbed, facsusc, facsusw, tend
    # Note: legacy files (like dms1fr) have 6 parameters: dl, rhos, por, coeftr, asfac, tend
    trans_vals = reader.read_record_flexible(6, 7, "transport_params")
    dl = float(trans_vals[0])
    rhos = float(trans_vals[1])
    por = float(trans_vals[2])
    if len(trans_vals) == 6:
        facbed = float(trans_vals[3])
        facsusc = float(trans_vals[4])
        facsusw = float(trans_vals[4])
        tend = float(trans_vals[5])
    else:
        facbed = float(trans_vals[3])
        facsusc = float(trans_vals[4])
        facsusw = float(trans_vals[5])
        tend = float(trans_vals[6])

    # READ(iuni, *) factime, facbed_smooth, facgrain_smooth
    step_vals = reader.read_tokens(3, "step_smoothing_params")
    factime = float(step_vals[0])
    facbed_smooth = float(step_vals[1])
    facgrain_smooth = float(step_vals[2])

    # READ(iuni, *) nfr, nsed, te, sa (or dummy in Fortran: read(iuni, *) i, j, te, sa)
    fluid_vals = reader.read_tokens(4, "nfr, nsed, te, sa")
    nfr = int(float(fluid_vals[0]))
    nsed = int(float(fluid_vals[1]))
    te = float(fluid_vals[2])
    sa = float(fluid_vals[3])

    # Sediment points along profile: xs, rc, rw, pmud, frip
    sed_meta = []
    for i in range(nsed):
        vals = [float(x) for x in reader.read_tokens(5, f"sed_point[{i}]")]
        sed_meta.append(vals)

    # Grain diameters for each fraction: ds(j) for j=1..nfr
    grain_tokens = reader.read_tokens(nfr, "grain_diameters")
    grain_diameters = [float(x) for x in grain_tokens]

    # Fraction percentages matrix (nsed points, each with x and nfr fractions)
    sediment_points = []
    for i in range(nsed):
        # Reads: xx, (ps(i, j), j=1, nfr)
        frac_row = [float(x) for x in reader.read_tokens(1 + nfr, f"fraction_row[{i}]")]
        xs, rc, rw, pmud, frip = sed_meta[i]
        sediment_points.append(
            SedimentProfilePoint(
                x=xs,
                rc=rc,
                rw=rw,
                pmud=pmud,
                frip=frip,
                fractions=frac_row[1:],
            )
        )

    # Depth contours for partial integration
    depth_contours = []
    if reader.has_more():
        ndc_val = reader.read_tokens(1, "ndc")
        ndc = int(float(ndc_val[0]))
        if ndc > 0:
            depth_tokens = reader.read_tokens(ndc, "depth_contours")
            depth_contours = [float(x) for x in depth_tokens]

    # X coordinates for partial integration
    x_contours = []
    if reader.has_more():
        ndcx_val = reader.read_tokens(1, "ndcx")
        ndcx = int(float(ndcx_val[0]))
        if ndcx > 0:
            x_tokens = reader.read_tokens(ndcx, "x_contours")
            x_contours = [float(x) for x in x_tokens]

    morpho = MorphoParams(
        iprant=iprant,
        nout=nout,
        dl=dl,
        rhos=rhos,
        por=por,
        facbed=facbed,
        facsusc=facsusc,
        facsusw=facsusw,
        tend=tend,
        factime=factime,
        facbed_smooth=facbed_smooth,
        facgrain_smooth=facgrain_smooth,
        water_temp=te,
        salinity=sa,
        grain_diameters=grain_diameters,
        depth_contours=depth_contours,
        x_contours=x_contours,
    )

    config = SimulationConfig(
        title=file_path.stem,
        iprint=iprint,
        tide_series=tide_series,
        bathymetry=bathymetry,
        hgrens=hgrens,
        istrand=istrand,
        storm_series=storm_series,
        igolf=igolf,
        wave_classes=wave_classes,
        nh_wave_classes=nh_wave_classes,
        wave_file=wave_file,
        wave_series=wave_series,
        hydro=hydro,
        morpho=morpho,
        sediment_points=sediment_points,
    )

    return config
