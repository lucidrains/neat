# nim

curl https://nim-lang.org/choosenim/init.sh -sSf | sh

choosenim 2.2.4

# uv

pip install uv

# python dep

uv pip install '.[test]' --system

# nim dep

nimble install
