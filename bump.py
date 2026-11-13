import re
import sys
from pathlib import Path

# neat/__init__.py is the source of truth, neat.nimble is a mirror (nimble requires a literal)
CANONICAL = Path('neat/__init__.py')
MIRRORS = [Path('neat.nimble')]

def read_version(path, name):
    match = re.search(rf'(?m)^{name}\s*=\s*"([^"]+)"', path.read_text())

    if not match:
        raise SystemExit(f'{name} not found in {path}')

    return match.group(1)

def set_version(path, name, version):
    text = path.read_text()
    updated, count = re.subn(rf'(?m)^{name}\s*=\s*"[^"]+"', f'{name} = "{version}"', text, count = 1)

    if count != 1:
        raise SystemExit(f'{name} not found in {path}')

    path.write_text(updated)

if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == '--check':
        version = read_version(CANONICAL, '__version__')
        stale = [str(path) for path in MIRRORS if read_version(path, 'version') != version]

        if stale:
            raise SystemExit(f'version {version} out of sync in {", ".join(stale)}')

        print(f'version {version} in sync')
        sys.exit(0)
    elif len(sys.argv) != 2:
        raise SystemExit('usage: python3 bump.py <version> | --check')

    version = sys.argv[1]

    set_version(CANONICAL, '__version__', version)

    for path in MIRRORS:
        set_version(path, 'version', version)

    print(f'version {version} set in {CANONICAL} (pyproject reads it dynamically), mirrored to {", ".join(map(str, MIRRORS))}')
