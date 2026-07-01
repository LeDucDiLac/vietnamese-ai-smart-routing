#!/usr/bin/env bash
# replay_common.sh — shared helpers for the replay control scripts. SOURCE, don't run.

# is_alive PID — true only if the process exists AND is not a zombie.
#
# Why not just `kill -0`: in this container PID 1 is jupyter-lab, not a reaping
# init, so a killed/finished replay child is never reaped and lingers as a zombie
# (<defunct>, state Z). `kill -0` returns SUCCESS for a zombie because the PID slot
# still exists — so a naive check reports a dead run as RUNNING, `stop` thinks its
# SIGKILL failed, and `launch` refuses to start. Reading the process state fixes it:
# a zombie is effectively dead (holds no CPU/RAM/GPU and cannot be signalled).
is_alive() {
  local pid="$1" state
  [ -n "$pid" ] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  state="$(ps -o state= -p "$pid" 2>/dev/null | tr -d '[:space:]')"
  [ -n "$state" ] && [ "${state#Z}" = "$state" ]
}
