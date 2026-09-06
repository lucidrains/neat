import re
import sys

VERSION_PATTERN = r'version\s*=\s*"([^"]+)"'
MANIFESTS = ['pyproject.toml', 'neat.nimble']

def set_version(path, version):
    with open(path) as f:
        content = f.read()

    updated = re.sub(VERSION_PATTERN, f'version = "{version}"', content, count = 1)

    if updated == content:
        raise SystemExit(f'version not found in {path}')

    with open(path, 'w') as f:
        f.write(updated)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: python3 bump.py <version>')

    version = sys.argv[1]

    for path in MANIFESTS:
        set_version(path, version)

    print(f'version {version} set in {", ".join(MANIFESTS)}')
