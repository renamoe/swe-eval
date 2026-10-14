#!/usr/bin/env bash
set -euo pipefail
# Use machine-local configuration; see qwen35-ssh.conf.example.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ssh_config="${QWEN35_SSH_CONFIG:-$script_dir/qwen35-ssh.conf}"
if [[ ! -f "$ssh_config" ]]; then
  echo 'Missing SSH configuration: copy scripts/qwen35-ssh.conf.example to scripts/qwen35-ssh.conf and edit it.' >&2
  exit 1
fi
case "${1:-start}" in
  start)
    if ssh -F "$ssh_config" -O check seetacloud-qwen35 2>/dev/null; then
      exit 0
    fi
    ssh -F "$ssh_config" -f -N -o BatchMode=yes -o ControlMaster=yes \
      -L 127.0.0.1:8000:127.0.0.1:8000 seetacloud-qwen35
    ;;
  status|stop)
    operation=check
    if [[ "$1" == stop ]]; then operation=exit; fi
    ssh -F "$ssh_config" -O "$operation" seetacloud-qwen35
    ;;
  *) echo "Usage: $0 [start|status|stop]" >&2; exit 2 ;;
esac
