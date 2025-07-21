# nim

curl https://nim-lang.org/choosenim/init.sh -sSf | sh

echo 'export PATH="$HOME/.nimble/bin:$PATH"' >> "$HOME/.bash_profile"
. "$HOME/.bash_profile"

# nim dep

choosenim 2.2.4
nimble install

# box2d

export OS_TYPE=$(uname -s)

if [[ "$OS_TYPE" == "Linux" ]]; then
  apt install swig -y
elif [[ "$OS_TYPE" == "Darwin" ]]; then
  brew install swig

# uv

pip install uv

# python dep

export UV_HTTP_TIMEOUT=60

uv pip install '.[gym]' --system
