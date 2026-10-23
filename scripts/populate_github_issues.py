#!/usr/bin/env python3
"""
Populate GitHub Issues from the AIP backlog CSV (aip_jira_backlog_phases1-5.csv).

Same logic as the Jira import:
  1. Labels      - one per team, phase and priority (created if missing)
  2. Milestones  - one per phase, due on the last due date in that phase
  3. Tasks       - created first (Parent ID empty)
  4. Sub-tasks   - created next, then attached to their parent as native GitHub sub-issues
  5. Depends On  - added as native "blocked by" relationships
  6. (optional)  - every issue added to a GitHub Project with Start date / Target date set

Safe to re-run: issues are matched by their "[ID]" title prefix, so nothing is created twice.

Usage:
  export GITHUB_TOKEN=...            # or log in with `gh auth login` and the script will use it
  python populate_github_issues.py --repo OWNER/REPO --csv aip_jira_backlog_phases1-5.csv --dry-run
  python populate_github_issues.py --repo OWNER/REPO --csv aip_jira_backlog_phases1-5.csv
  python populate_github_issues.py --repo OWNER/REPO --csv ... --project-owner OWNER --project-number 1

Options:
  --team GPU            only import one team (Team Key: GPU, ML, HW, CPP)
  --limit 5             only import the first N tasks (and their subtasks) - good for a test run
  --skip-subissues      create subtasks as issues but don't link them as sub-issues
  --skip-deps           don't add blocked-by relationships
  --delay 1.0           seconds between write calls (GitHub limits fast content creation)
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time

try:
    import requests
except ImportError:
    sys.exit("Missing dependency: run  pip install requests")

API = "https://api.github.com"
GQL = "https://api.github.com/graphql"
ID_RE = re.compile(r"^\[([A-Z]+-\d+(?:\.\d+)?)\]")

TEAM_COLORS = {"GPU-CUDA": "5B9BD5", "ML-CV": "ED7D31", "Hardware-Robotics": "70AD47", "Cpp-Graphics": "7F6FA8"}
PRIORITY_COLORS = {"P0": "C00000", "P1": "FFC000", "P2": "A5A5A5"}
PRIORITY_TEXT = {"P0": "Critical: blocks the MVP or another team",
                 "P1": "High: needed for success criteria",
                 "P2": "Medium: first to slip"}


# --------------------------------------------------------------------------- HTTP helpers
class GitHub:
    def __init__(self, token, delay, dry_run):
        self.s = requests.Session()
        self.s.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        self.delay = delay
        self.dry_run = dry_run

    def _request(self, method, url, **kw):
        for attempt in range(6):
            r = self.s.request(method, url, timeout=30, **kw)
            if r.status_code in (403, 429) and ("rate limit" in r.text.lower() or "retry-after" in r.headers):
                wait = int(r.headers.get("retry-after", 0)) or 60 * (attempt + 1)
                reset = r.headers.get("x-ratelimit-reset")
                if r.headers.get("x-ratelimit-remaining") == "0" and reset:
                    wait = max(wait, int(reset) - int(time.time()) + 5)
                print(f"  rate limited, waiting {wait}s ...")
                time.sleep(wait)
                continue
            if r.status_code >= 500:
                time.sleep(5 * (attempt + 1))
                continue
            return r
        return r

    def get(self, path, **params):
        r = self._request("GET", API + path, params=params)
        r.raise_for_status()
        return r.json()

    def paged(self, path, **params):
        params.setdefault("per_page", 100)
        page = 1
        while True:
            params["page"] = page
            items = self.get(path, **params)
            if not items:
                return
            yield from items
            if len(items) < params["per_page"]:
                return
            page += 1

    def write(self, method, path, payload):
        if self.dry_run:
            return None
        r = self._request(method, API + path, json=payload)
        time.sleep(self.delay)
        return r

    def graphql(self, query, variables):
        if self.dry_run and query.lstrip().startswith("mutation"):
            return {}
        r = self._request("POST", GQL, json={"query": query, "variables": variables})
        r.raise_for_status()
        data = r.json()
        if "errors" in data:
            raise RuntimeError(json.dumps(data["errors"], indent=2))
        if query.lstrip().startswith("mutation"):
            time.sleep(self.delay)
        return data["data"]


def get_token():
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        return token
    try:
        return subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        sys.exit("No token. Set GITHUB_TOKEN, or install the GitHub CLI and run `gh auth login`.")


# --------------------------------------------------------------------------- CSV
def load_rows(path, team, limit):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    required = {"Issue ID", "Parent ID", "Issue Type", "Team", "Team Key", "Phase", "Summary", "Description",
                "Criticality", "Story Points", "Start Week", "End Week", "Start Date", "Due Date", "Depends On"}
    missing = required - set(rows[0].keys())
    if missing:
        sys.exit(f"CSV is missing columns: {sorted(missing)}")
    if team:
        rows = [r for r in rows if r["Team Key"].upper() == team.upper()]
    if limit:
        keep = [r["Issue ID"] for r in rows if r["Issue Type"] == "Task"][:limit]
        rows = [r for r in rows if r["Issue ID"] in keep or r["Parent ID"] in keep]
    return rows


def phase_num(phase_text):
    m = re.search(r"Phase (\d)", phase_text)
    return int(m.group(1)) if m else 0


def labels_for(row):
    return [f"team: {row['Team']}", f"phase: {phase_num(row['Phase'])}", f"priority: {row['Criticality']}"] + \
           (["subtask"] if row["Issue Type"] == "Sub-task" else ["task"])


def body_for(row, id_to_number):
    def ref(issue_id):
        n = id_to_number.get(issue_id)
        return f"#{n} ({issue_id})" if n else issue_id

    deps = [d.strip() for d in row["Depends On"].split(",") if d.strip()]
    lines = [
        row["Description"],
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| ID | `{row['Issue ID']}` |",
        f"| Team | {row['Team']} |",
        f"| Phase | {row['Phase']} |",
        f"| Priority | {row['Criticality']} – {PRIORITY_TEXT.get(row['Criticality'], '')} |",
    ]
    if row["Story Points"]:
        lines.append(f"| Story points | {row['Story Points']} |")
    lines += [
        f"| Weeks | {row['Start Week']}–{row['End Week']} |",
        f"| Start date | {row['Start Date']} |",
        f"| Due date | {row['Due Date']} |",
    ]
    if row["Parent ID"]:
        lines.append(f"| Parent | {ref(row['Parent ID'])} |")
    if deps:
        lines.append(f"| Depends on | {', '.join(ref(d) for d in deps)} |")
    lines += ["", "<!-- generated by scripts/populate_github_issues.py from the AIP backlog CSV -->"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- steps
def ensure_labels(gh, repo, rows):
    existing = {l["name"] for l in gh.paged(f"/repos/{repo}/labels")}
    wanted = {}
    for r in rows:
        wanted[f"team: {r['Team']}"] = TEAM_COLORS.get(r["Team"], "BFBFBF")
        wanted[f"phase: {phase_num(r['Phase'])}"] = "1F3864"
        wanted[f"priority: {r['Criticality']}"] = PRIORITY_COLORS.get(r["Criticality"], "BFBFBF")
    wanted["task"] = "0E8A16"
    wanted["subtask"] = "C2E0C6"
    for name, color in sorted(wanted.items()):
        if name not in existing:
            print(f"  + label {name}")
            gh.write("POST", f"/repos/{repo}/labels", {"name": name, "color": color})


def ensure_milestones(gh, repo, rows):
    due = {}
    names = {}
    for r in rows:
        p = phase_num(r["Phase"])
        names[p] = r["Phase"]
        due[p] = max(due.get(p, ""), r["Due Date"])
    existing = {m["title"]: m["number"] for m in gh.paged(f"/repos/{repo}/milestones", state="all")}
    out = {}
    for p in sorted(names):
        title = names[p]
        if title in existing:
            out[p] = existing[title]
            continue
        print(f"  + milestone {title} (due {due[p]})")
        r = gh.write("POST", f"/repos/{repo}/milestones",
                     {"title": title, "due_on": f"{due[p]}T23:59:59Z"})
        if r is not None:
            r.raise_for_status()
            out[p] = r.json()["number"]
    return out


def existing_issues(gh, repo):
    found = {}
    for it in gh.paged(f"/repos/{repo}/issues", state="all"):
        if "pull_request" in it:
            continue
        m = ID_RE.match(it["title"])
        if m:
            found[m.group(1)] = {"number": it["number"], "id": it["id"], "node_id": it["node_id"]}
    return found


def create_issues(gh, repo, rows, milestones, issues, kind):
    id_to_number = {k: v["number"] for k, v in issues.items()}
    batch = [r for r in rows if r["Issue Type"] == kind]
    for i, r in enumerate(batch, 1):
        iid = r["Issue ID"]
        if iid in issues:
            continue
        payload = {
            "title": r["Summary"],
            "body": body_for(r, id_to_number),
            "labels": labels_for(r),
        }
        ms = milestones.get(phase_num(r["Phase"]))
        if ms:
            payload["milestone"] = ms
        print(f"  [{i}/{len(batch)}] create {r['Summary'][:80]}")
        resp = gh.write("POST", f"/repos/{repo}/issues", payload)
        if resp is None:
            continue
        if resp.status_code >= 300:
            print(f"    ! failed ({resp.status_code}): {resp.text[:200]}")
            continue
        j = resp.json()
        issues[iid] = {"number": j["number"], "id": j["id"], "node_id": j["node_id"]}
        id_to_number[iid] = j["number"]


def link_subissues(gh, repo, rows, issues):
    for r in rows:
        if r["Issue Type"] != "Sub-task":
            continue
        parent, child = issues.get(r["Parent ID"]), issues.get(r["Issue ID"])
        if not parent or not child:
            continue
        resp = gh.write("POST", f"/repos/{repo}/issues/{parent['number']}/sub_issues",
                        {"sub_issue_id": child["id"]})
        if resp is not None and resp.status_code >= 300 and "already" not in resp.text.lower():
            print(f"    ! sub-issue {r['Issue ID']} -> {r['Parent ID']} ({resp.status_code}): {resp.text[:150]}")


def link_dependencies(gh, repo, rows, issues):
    for r in rows:
        blocked = issues.get(r["Issue ID"])
        for dep in [d.strip() for d in r["Depends On"].split(",") if d.strip()]:
            blocker = issues.get(dep)
            if not blocked or not blocker:
                continue
            resp = gh.write("POST", f"/repos/{repo}/issues/{blocked['number']}/dependencies/blocked_by",
                            {"issue_id": blocker["id"]})
            if resp is not None and resp.status_code >= 300 and "already" not in resp.text.lower():
                print(f"    ! dependency {r['Issue ID']} blocked by {dep} ({resp.status_code}): {resp.text[:150]}")


# --------------------------------------------------------------------------- Projects (optional)
def project_setup(gh, owner, number):
    q = """query($login:String!,$number:Int!){
      user(login:$login){projectV2(number:$number){id title fields(first:50){nodes{... on ProjectV2Field{id name dataType}}}}}
    }"""
    q_org = q.replace("user(login", "organization(login")
    data = None
    for query, key in ((q_org, "organization"), (q, "user")):
        try:
            d = gh.graphql(query, {"login": owner, "number": number})
            if d.get(key) and d[key].get("projectV2"):
                data = d[key]["projectV2"]
                break
        except RuntimeError:
            continue
    if not data:
        sys.exit(f"Project #{number} not found for {owner}. Check --project-owner/--project-number and token scopes.")
    fields = {f["name"]: f["id"] for f in data["fields"]["nodes"] if f and f.get("dataType") == "DATE"}
    for name in ("Start date", "Target date"):
        if name not in fields:
            print(f"  + project field {name}")
            d = gh.graphql("""mutation($p:ID!,$n:String!){createProjectV2Field(input:{projectId:$p,dataType:DATE,name:$n})
                              {projectV2Field{... on ProjectV2Field{id}}}}""", {"p": data["id"], "n": name})
            fields[name] = (d or {}).get("createProjectV2Field", {}).get("projectV2Field", {}).get("id")
    print(f"  project: {data['title']}")
    return data["id"], fields


def add_to_project(gh, project_id, fields, rows, issues):
    set_date = """mutation($p:ID!,$i:ID!,$f:ID!,$d:Date!){updateProjectV2ItemFieldValue(
                    input:{projectId:$p,itemId:$i,fieldId:$f,value:{date:$d}}){projectV2Item{id}}}"""
    for n, r in enumerate(rows, 1):
        it = issues.get(r["Issue ID"])
        if not it:
            continue
        d = gh.graphql("""mutation($p:ID!,$c:ID!){addProjectV2ItemById(input:{projectId:$p,contentId:$c}){item{id}}}""",
                       {"p": project_id, "c": it["node_id"]})
        item_id = (d or {}).get("addProjectV2ItemById", {}).get("item", {}).get("id")
        if not item_id:
            continue
        for name, col in (("Start date", "Start Date"), ("Target date", "Due Date")):
            if fields.get(name) and r[col]:
                gh.graphql(set_date, {"p": project_id, "i": item_id, "f": fields[name], "d": r[col]})
        if n % 25 == 0:
            print(f"  project: {n}/{len(rows)} items")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="OWNER/REPO")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--team", help="Team Key to import (GPU, ML, HW, CPP)")
    ap.add_argument("--limit", type=int, help="Only the first N tasks (plus their subtasks)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-subissues", action="store_true")
    ap.add_argument("--skip-deps", action="store_true")
    ap.add_argument("--project-owner", help="User or org that owns the GitHub Project")
    ap.add_argument("--project-number", type=int, help="Project number from its URL (.../projects/N)")
    ap.add_argument("--delay", type=float, default=1.0)
    a = ap.parse_args()

    rows = load_rows(a.csv, a.team, a.limit)
    tasks = sum(r["Issue Type"] == "Task" for r in rows)
    print(f"Loaded {len(rows)} rows ({tasks} tasks, {len(rows) - tasks} subtasks){' [DRY RUN]' if a.dry_run else ''}")

    gh = GitHub(get_token(), a.delay, a.dry_run)
    gh.get(f"/repos/{a.repo}")  # fails early if repo/token is wrong

    print("1/6 labels");      ensure_labels(gh, a.repo, rows)
    print("2/6 milestones");  milestones = ensure_milestones(gh, a.repo, rows)
    issues = existing_issues(gh, a.repo)
    print(f"    {len(issues)} matching issues already exist (will be skipped)")
    print("3/6 tasks");       create_issues(gh, a.repo, rows, milestones, issues, "Task")
    print("4/6 subtasks");    create_issues(gh, a.repo, rows, milestones, issues, "Sub-task")
    if not a.skip_subissues:
        print("    linking sub-issues"); link_subissues(gh, a.repo, rows, issues)
    if not a.skip_deps:
        print("5/6 dependencies"); link_dependencies(gh, a.repo, rows, issues)
    if a.project_number:
        print("6/6 project")
        pid, fields = project_setup(gh, a.project_owner or a.repo.split("/")[0], a.project_number)
        add_to_project(gh, pid, fields, rows, issues)
    print("Done." if not a.dry_run else "Dry run complete: nothing was written.")


if __name__ == "__main__":
    main()
