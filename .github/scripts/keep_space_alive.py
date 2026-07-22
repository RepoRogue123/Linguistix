"""Keep the Linguistix Hugging Face Space reachable.

Free `cpu-basic` Spaces are paused after 48h of inactivity, and a paused Space
stays down until the owner restarts it -- visitors cannot wake it. Crashes
(RUNTIME_ERROR) never self-heal either. This script runs on a schedule from
outside the Space and does two things:

  1. Restarts the Space if it is paused, stopped, or crashed.
  2. Otherwise pings it, which resets the inactivity timer so the 48h pause
     threshold is never reached.

Build/config errors are deliberately NOT restarted: they are deterministic
failures, so a restart would just loop. The run fails instead, so the failure
surfaces as a GitHub notification.
"""

import os
import sys
import time
import urllib.error
import urllib.request

from huggingface_hub import HfApi

SPACE_ID = os.environ.get("HF_SPACE_ID", "").strip()
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
PING_TIMEOUT = 180  # a cold start pulls in torch + the model; give it room

# Dormant but recoverable: a restart is the correct action.
RESTARTABLE = {"PAUSED", "STOPPED", "RUNTIME_ERROR"}
# Broken code/config: restarting cannot fix these, so fail loudly instead.
BROKEN = {"BUILD_ERROR", "CONFIG_ERROR", "NO_APP_FILE"}
# Already on its way up: nothing to do.
IN_PROGRESS = {"BUILDING", "APP_STARTING", "RUNNING_BUILDING"}


def space_url(space_id: str) -> str:
    """owner/My_Space -> https://owner-my-space.hf.space"""
    owner, _, name = space_id.partition("/")
    subdomain = f"{owner}-{name}".lower().replace("_", "-").replace(".", "-")
    return f"https://{subdomain}.hf.space"


def ping(url: str) -> None:
    """Hit the Space over HTTP. This is what resets the inactivity timer."""
    req = urllib.request.Request(
        f"{url}/health", headers={"User-Agent": "linguistix-keepalive"}
    )
    try:
        with urllib.request.urlopen(req, timeout=PING_TIMEOUT) as resp:
            print(f"ping {url}/health -> {resp.status}")
    except urllib.error.HTTPError as err:
        # Any response still counts as activity, even a 5xx from a warming app.
        print(f"ping {url}/health -> HTTP {err.code} (still counts as activity)")
    except Exception as err:  # noqa: BLE001 - a failed ping must not fail the job
        print(f"ping {url}/health failed: {err}")


def main() -> int:
    if not SPACE_ID:
        print("HF_SPACE_ID is not set. Add it as a repository variable.", file=sys.stderr)
        return 1
    if not HF_TOKEN:
        print("HF_TOKEN is not set. Add it as a repository secret.", file=sys.stderr)
        return 1

    api = HfApi(token=HF_TOKEN)
    runtime = api.get_space_runtime(repo_id=SPACE_ID)
    stage = str(runtime.stage)
    print(f"{SPACE_ID} stage={stage} hardware={runtime.hardware}")

    if stage in BROKEN:
        print(
            f"Space is in {stage}. A restart will not fix this -- check the build "
            f"logs: hf spaces logs {SPACE_ID} --build",
            file=sys.stderr,
        )
        return 1

    if stage in RESTARTABLE:
        print(f"Space is {stage}; restarting.")
        api.restart_space(repo_id=SPACE_ID)
        time.sleep(30)  # let the restart register before we probe it
        ping(space_url(SPACE_ID))
        return 0

    if stage in IN_PROGRESS:
        print("Space is already starting; nothing to do.")
        return 0

    # RUNNING or SLEEPING: the ping both wakes a sleeper and resets the timer.
    ping(space_url(SPACE_ID))
    return 0


if __name__ == "__main__":
    sys.exit(main())
