source "$HOME/.nvm/nvm.sh" && nvm use 22
export HTTP_PROXY=http://127.0.0.1:1087
export HTTPS_PROXY=http://127.0.0.1:1087
export NO_PROXY=127.0.0.1,localhost
unset CLAUDECODE
cc-connect --config ./claude.toml