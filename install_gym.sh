# nim

curl https://nim-lang.org/choosenim/init.sh -sSf | sh

echo 'export PATH="$HOME/.nimble/bin:$PATH"' >> "$HOME/.bash_profile"
. "$HOME/.bash_profile"

# box2d

apt install swig -y

# uv

pip install uv

# python dep

uv pip install '.[gym]' --system

# nim dep

choosenim 2.2.4
nimble install
