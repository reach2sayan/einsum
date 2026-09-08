#!/usr/bin/env python3
"""Stage the native NuGet package tree from two install prefixes.

The layout `nuget pack contrib/nuget/einsum.nuspec -BasePath <stage>` expects:

    <stage>/
        README.md, LICENSE.txt
        build/native/einsum.targets
        build/native/include/**                            einsum + the
                                                           vendored Boost,
                                                           Eigen and mdspan
        build/native/lib/x64/{Release,Debug}/einsum_rt.lib
        build/native/bin/x64/{Release,Debug}/einsum_rt.dll

Headers come from the Release prefix; the two prefixes differ only in the
binaries.  Two flavours because rt/parse.hpp hands back a type holding a
boost::container::static_vector by value, so the CRT the DLL was built against
has to be the CRT the caller was.

A straight copytree is enough for the headers, unlike the sibling Seitz
package's script, and the three passes it makes are all absent here on purpose:

  * no versioned `boost-1_XX/` to unwrap.  einsum never runs Boost's own
    install -- cmake/EinsumDependencies.cmake points FetchContent at
    SOURCE_SUBDIR einsum-does-not-build-boost, so the archive is unpacked and
    nothing else -- and cmake/EinsumInstall.cmake places the vendored subsets
    itself, at the relative paths cmake/EinsumVendoredHeaders.cmake names.  The
    result is one flat include directory holding einsum/, boost/, Eigen/,
    experimental/ and mdspan/, which is the single path einsum.targets adds.
  * no dotfiles to delete.  Boost ships boost/headers/.gitkeep, and an empty
    directory is what makes `nuget pack` die with "String cannot be empty.
    Parameter name: entryName" -- but EinsumInstall.cmake globs
    "${root}/${dir}/*", and file(GLOB) follows shell convention: a leading *
    does not match a leading dot.  boost/headers is not on the list either.
  * no empty directories to prune.  A listed directory that holds no files is a
    configure-time FATAL_ERROR in EinsumInstall.cmake, and an unlisted
    intermediate exists only as the parent of one that does.

Missing pieces stop the script -- a package staged short would pack and publish
without complaint.

Usage:
    python3 pack_nuget.py --release <prefix> --debug <prefix> --stage <dir>
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# One per vendored dependency plus our own umbrella header: if the install
# rules stop placing a subset, the package still builds and only fails in a
# consumer's translation unit, a long way from here.
EXPECTED_HEADERS = (
    "einsum/einsum.hpp",
    "boost/version.hpp",
    "Eigen/Core",
    "experimental/mdspan",
)


def directory_size(root: Path) -> int:
    """Total bytes of every file under `root`."""
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def main() -> int:
    """Stage the package tree; non-zero if anything it should carry is absent."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--release", type=Path, required=True, help="Release install prefix"
    )
    ap.add_argument("--debug", type=Path, required=True, help="Debug install prefix")
    ap.add_argument(
        "--stage", type=Path, required=True, help="output directory (recreated)"
    )
    args = ap.parse_args()

    shutil.rmtree(args.stage, ignore_errors=True)
    native = args.stage / "build" / "native"
    native.mkdir(parents=True)

    # The .targets filename has to equal the package id; that match is the whole
    # reason NuGet imports it into a referencing vcxproj.
    shutil.copy2(ROOT / "contrib" / "nuget" / "einsum.targets", native)
    shutil.copy2(ROOT / "README.md", args.stage)
    # LICENSE.txt already carries an extension, which NuGet needs at the package
    # root: it decides file-versus-folder by that alone.
    shutil.copy2(ROOT / "LICENSE.txt", args.stage)
    shutil.copytree(args.release / "include", native / "include")

    missing = [
        native / "include" / rel
        for rel in EXPECTED_HEADERS
        if not (native / "include" / rel).is_file()
    ]

    for config, prefix in (("Release", args.release), ("Debug", args.debug)):
        for sub, name in (("lib", "einsum_rt.lib"), ("bin", "einsum_rt.dll")):
            src = prefix / sub / name
            if not src.is_file():
                missing.append(src)
                continue
            dst = native / sub / "x64" / config
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    if missing:
        for path in missing:
            print(f"missing: {path}", file=sys.stderr)
        return 1

    # Printed before the workflow's size gate, so a package that trips it has
    # the number in the log either way.
    print(f"staged {args.stage} ({directory_size(args.stage) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
