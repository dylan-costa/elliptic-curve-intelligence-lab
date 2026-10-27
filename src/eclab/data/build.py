"""Build data/curves.parquet from the source file and validate it.

    python -m eclab.data.build [--source PATH] [--out data/curves.parquet]

The validation report goes next to the table as validation.json.
"""

import argparse
import json
import time
from pathlib import Path

from eclab.data.ecdata import DEFAULT_TABLE, read_ecdata, source_path
from eclab.data.validate import validate


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=Path, default=None, help="default: $ECLAB_ECDATA or data/raw/ecdata_v2.txt")
    ap.add_argument("--out", type=Path, default=DEFAULT_TABLE)
    args = ap.parse_args(argv)
    source = args.source or source_path()

    t0 = time.perf_counter()
    df = read_ecdata(source)
    print(f"parsed {len(df):,} curves from {source} in {time.perf_counter() - t0:.0f}s")

    t0 = time.perf_counter()
    report = validate(df)
    report["source"] = source.name
    print(f"validated in {time.perf_counter() - t0:.0f}s")
    for name, n in report["failures"].items():
        print(f"  {name:32s} {n:>9,}" + (f"   e.g. {report['examples'][name]}" if n else ""))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    (args.out.parent / "validation.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")
    if any(report["failures"].values()):
        raise SystemExit("validation failed")


if __name__ == "__main__":
    main()
