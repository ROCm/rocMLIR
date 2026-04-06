#!/usr/bin/env python3
"""Auto-merge PRs targeting develop when CI passes, retrigger on failure."""

import json
import logging
import os
import subprocess
import sys
import time

MAX_RETRIES = 100
POLL_INTERVAL_SEC = 60
LOG_FILE = "auto_merge.log"

logger = logging.getLogger("auto_merge")
repo_dir: str = ""


def setup_logging():
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(LOG_FILE)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    logger.debug("Running: %s", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True, check=check, cwd=repo_dir or None)


def gh_json(args: list[str]) -> object:
    result = run(["gh"] + args)
    return json.loads(result.stdout)


def is_based_on_develop(branch: str) -> bool:
    """Check if the PR branch is based on the latest origin/develop."""
    result = run(["git", "merge-base", "--is-ancestor", "origin/develop", f"origin/{branch}"], check=False)
    return result.returncode == 0


def get_my_prs() -> list[dict]:
    prs = gh_json([
        "pr", "list",
        "--author", "@me",
        "--base", "develop",
        "--state", "open",
        "--json", "number,headRefName,title,reviewDecision",
        "--limit", "100",
    ])
    return [pr for pr in prs if pr.get("reviewDecision") == "APPROVED"]


def check_ci(pr_number: int) -> str:
    """Return aggregate CI status: 'pass', 'fail', or 'pending'."""
    checks = gh_json(["pr", "checks", str(pr_number), "--json", "bucket,name,state"])
    if not checks:
        return "pending"

    buckets = {c["bucket"] for c in checks}

    if buckets <= {"pass", "skipping"}:
        return "pass"
    if "fail" in buckets or "cancel" in buckets:
        return "fail"
    return "pending"


def merge_pr(pr_number: int) -> bool:
    result = run(["gh", "pr", "merge", str(pr_number), "--merge"], check=False)
    if result.returncode == 0:
        logger.info("PR #%d merged successfully.", pr_number)
        return True
    logger.error("Failed to merge PR #%d: %s", pr_number, result.stderr.strip())
    return False


def retrigger_ci(branch: str) -> bool:
    try:
        run(["git", "fetch", "origin", branch])
        run(["git", "checkout", branch])
        run(["git", "commit", "--allow-empty", "-m", "retrigger CI"])
        run(["git", "push", "origin", branch])
        return True
    except subprocess.CalledProcessError as exc:
        logger.error("Failed to retrigger CI on %s: %s", branch, exc.stderr.strip())
        return False


def main():
    global repo_dir
    setup_logging()

    repo_dir = input("Enter path to the git repository: ").strip()
    repo_dir = os.path.expanduser(repo_dir)
    if not os.path.isdir(os.path.join(repo_dir, ".git")):
        logger.error("Not a valid git repository: %s", repo_dir)
        return
    logger.info("Using repository at %s", repo_dir)
    logger.info("Starting auto-merge monitor.")

    prs = get_my_prs()
    if not prs:
        logger.info("No open PRs targeting develop found. Exiting.")
        return

    retry_counts: dict[int, int] = {pr["number"]: 0 for pr in prs}

    for pr in prs:
        logger.info("Tracking PR #%d: %s (branch: %s)", pr["number"], pr["title"], pr["headRefName"])

    confirm = input(f"Proceed with {len(prs)} PR(s)? [y/N] ").strip().lower()
    if confirm != "y":
        logger.info("Aborted by user.")
        return

    try:
        while prs:
            run(["git", "fetch", "--all"])
            remaining = []
            for pr in prs:
                num = pr["number"]
                branch = pr["headRefName"]
                status = check_ci(num)
                logger.info("PR #%d CI status: %s", num, status)

                if status == "pass":
                    if not is_based_on_develop(branch):
                        logger.warning("PR #%d branch %s is not based on origin/develop. Dropping.", num, branch)
                        continue
                    if merge_pr(num):
                        continue
                    remaining.append(pr)

                elif status == "fail":
                    if retry_counts[num] >= MAX_RETRIES:
                        logger.warning("PR #%d exceeded %d retries. Giving up.", num, MAX_RETRIES)
                        continue
                    retry_counts[num] += 1
                    logger.info(
                        "PR #%d CI failed. Retriggering (attempt %d/%d).",
                        num, retry_counts[num], MAX_RETRIES,
                    )
                    retrigger_ci(branch)
                    remaining.append(pr)

                else:
                    remaining.append(pr)

            prs = remaining
            if prs:
                logger.info("Sleeping %d seconds before next poll...", POLL_INTERVAL_SEC)
                time.sleep(POLL_INTERVAL_SEC)

    except KeyboardInterrupt:
        logger.info("Interrupted by user. Shutting down.")

    logger.info("Auto-merge monitor finished.")


if __name__ == "__main__":
    main()
