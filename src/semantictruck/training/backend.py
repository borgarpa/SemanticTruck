"""Finding Pointcept, and running one of its tools with our dataset registered.

Pointcept is a **dependency, not part of this repository**. It is a research framework installed
from source rather than from PyPI, so it is located at runtime rather than declared in
`pyproject.toml`: pass `--pointcept-root`, set `$POINTCEPT_ROOT`, or have it already importable.

**Why a launcher exists at all.** Pointcept resolves a dataset by name through a registry that is
populated by import, and the classes it knows about are the ones under `pointcept/datasets/`. Ours
lives here instead, so something has to import `semantictruck.training.dataset` -- which registers
it -- before Pointcept builds anything.

Putting that import in the config file does not work. Pointcept executes a config to read it, so
the import would fire on the first pass; but it then *dumps* the resolved config back out with
`Config.dump`, which writes `pretty_text` -- a pretty-printed dictionary. Every import statement is
lost. Testing reloads that dumped file, and the dataset would be unregistered by the time it did.

So instead we run the Pointcept tool through this module: it puts Pointcept on the path, imports
our dataset, and only then hands over. `runpy` rather than a subprocess-within-a-subprocess because
Pointcept's `launch()` calls the worker in-process whenever the world size is one, which is the
single-GPU case this campaign runs.

    python -m semantictruck.training.backend <pointcept>/tools/train.py \\
        --config-file ... --options ...
"""

from __future__ import annotations

import os
import runpy
import subprocess
import sys

ENV_VAR = 'POINTCEPT_ROOT'

INSTALL_HINT = """Pointcept was not found. It is a dependency of the training stage and is
installed from source:

    git clone https://github.com/Pointcept/Pointcept.git
    cd Pointcept && <build libs/pointops for your CUDA toolchain>

then point this command at it, by either

    semantictruck-train --pointcept-root /path/to/Pointcept
    export {}=/path/to/Pointcept

See docs/reproducing.md. Note the training stage needs its own environment -- its torch and NumPy
requirements are the opposite of the ones the geometry stage pins.""".format(ENV_VAR)


def pointcept_root(explicit: str | None = None) -> str:
    """Where Pointcept is: an explicit path, the environment, or an already-importable package."""
    for candidate in (explicit, os.environ.get(ENV_VAR)):
        if candidate:
            root = os.path.abspath(os.path.expanduser(candidate))
            if not os.path.isdir(os.path.join(root, 'pointcept')):
                raise SystemExit('{} does not look like a Pointcept checkout: no pointcept/ '
                                 'directory in it'.format(root))
            return root

    try:                                    # already on the path, e.g. a pip install -e of it
        import pointcept
        return os.path.dirname(os.path.dirname(os.path.abspath(pointcept.__file__)))
    except ImportError:
        raise SystemExit(INSTALL_HINT)


def ensure_importable(root: str) -> None:
    """Put Pointcept on `sys.path` and check that it imports."""
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        import pointcept                                        # noqa: F401
    except ImportError as missing:
        raise SystemExit('{}\n\nPointcept is at {} but will not import. Its dependencies -- torch, '
                         'spconv, flash-attn and the compiled libs/pointops -- have to be '
                         'installed in this environment.'.format(missing, root))


def tool_path(root: str, name: str) -> str:
    """`<root>/tools/<name>.py`, checked, so a wrong root fails here rather than deep inside."""
    path = os.path.join(root, 'tools', name + '.py')
    if not os.path.exists(path):
        raise SystemExit('no {} in the Pointcept checkout at {}'.format(
            os.path.join('tools', name + '.py'), root))
    return path


def run_tool(name: str, options: list[str], root: str, num_gpus: int = 1,
             config_file: str | None = None) -> None:
    """Run a Pointcept tool in its own process, with our dataset registered first.

    A fresh process per fold, because that is what gives each one a clean CUDA context -- on a
    small card the difference decides whether the next fold fits at all. Run with the Pointcept
    checkout as the working directory, which is what its tools assume.
    """
    command = [sys.executable, '-m', __spec__.name if __spec__ else
               'semantictruck.training.backend', tool_path(root, name)]
    if config_file:
        command += ['--config-file', config_file]
    command += ['--num-gpus', str(num_gpus), '--options'] + options

    print('\n$ ' + ' '.join(command), flush=True)
    completed = subprocess.run(command, cwd=root)
    if completed.returncode != 0:
        raise RuntimeError('{} failed with exit code {}'.format(name, completed.returncode))


def main(argv=None) -> int:
    """Import our dataset into Pointcept's registry, then run the tool named on the command line."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise SystemExit('usage: python -m semantictruck.training.backend <pointcept tool> [args]')

    tool = os.path.abspath(argv[0])
    ensure_importable(os.path.dirname(os.path.dirname(tool)))

    # The whole point of this module: the decorator in here registers TruckSegmDataset with
    # Pointcept before any config is built.
    import semantictruck.training.dataset                       # noqa: F401

    sys.argv = [tool] + argv[1:]
    runpy.run_path(tool, run_name='__main__')
    return 0


if __name__ == '__main__':
    sys.exit(main())
