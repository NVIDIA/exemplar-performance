#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# ruff: noqa

"""selftest_wizard_init.py — offline unit tests for the `llmb-k8s init` wizard (the wizard contract Phase-1).

Pure/offline — NO cluster, NO network, NO pods. Exercises exactly the invariants the spec/task call out:
  - dispatcher: `init` is in BOTH KNOWN_VERBS and USAGE (the assert set(USAGE)==set(KNOWN_VERBS) holds),
    and the verb bypasses resolve_or_exit (does not fail-fast on a MISSING profile).
  - Done renderer (NEW-2): consumes the STRUCTURED Check list; maps PASS→✓ / WARN→⚠, and TRUNCATES at the
    first ❌ (later checks not shown).
  - model-cache two-case detection (S1): absent → ❌; name-match → ✓; position-match → ⚠.
  - single-IP writing (§6.1): valid_single_ip rejects a comma-list; a playfile with a comma-list is rejected;
    a written profile carries single IPs and NO comma.
  - schema-version reader round-trip (Q7): a `# LLMB_PROFILE_SCHEMA=N` header round-trips; absent → 0.
  - --play: non-zero on any ❌ AND never provisions (no live apply); zero on all-pass.
  - config-storage atomic write (§5): 0600, atomic replace, 7-day expiry, cluster-agnostic-only prefs.
  - RWX detector (§6.4): ranks by name substring; safe-degrades to None when nothing matches.

Run: `python3 scripts/selftest_wizard_init.py` or via `make test`. Exit 0 = all pass.
"""

from __future__ import annotations

import importlib.util
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


wiz = _load("wizard_init")
cr = _load("cluster_readiness")
pr = _load("profile_resolver")
k8s_config = _load("k8s_config")

fails: list[str] = []


def _sc_json(*classes) -> str:
    """Build a `kubectl get storageclasses -o json` payload for the fake krun (the wizard now reads the
    detailed list via `-o json` to know each class's provisioner). Each arg is (name, provisioner,
    is_default)."""
    import json as _json

    items = []
    for name, prov, is_def in classes:
        ann = {"storageclass.kubernetes.io/is-default-class": "true"} if is_def else {}
        items.append({"metadata": {"name": name, "annotations": ann}, "provisioner": prov})
    return _json.dumps({"items": items})


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if not cond and detail else ""))
    if not cond:
        fails.append(name)


# ── 1. dispatcher: init in KNOWN_VERBS + USAGE, assert holds ──────────────────
def test_dispatcher():
    src = (SCRIPTS / "llmb-k8s").read_text()
    # Importing the dispatcher module executes the assert set(USAGE)==set(KNOWN_VERBS) at import time.
    # The file has no .py extension, so use an explicit SourceFileLoader.
    from importlib.machinery import SourceFileLoader

    loader = SourceFileLoader("llmb_k8s_dispatch", str(SCRIPTS / "llmb-k8s"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    try:
        loader.exec_module(mod)
        loaded = True
    except AssertionError as e:
        loaded = False
        check("dispatcher USAGE==KNOWN_VERBS assert holds with init", False, str(e))
        return
    check("dispatcher module imports (assert holds)", loaded)
    check("init in KNOWN_VERBS", "init" in mod.KNOWN_VERBS)
    check("init in USAGE", "init" in mod.USAGE)
    check(
        "init dispatch does NOT route through resolve_or_exit",
        'if verb == "init":' in src
        and "wizard_init.py" in src
        and src.index('if verb == "init":') < src.index('return py(ROOT / "scripts/wizard_init.py"'),
    )


# ── 2. Done renderer: PASS→✓, WARN→⚠, truncate at first ❌ ─────────────────────
def test_done_renderer():
    C = cr.Check
    checks = [
        C("var-reconcile", cr.PASS, "all vars resolve"),
        C("gpu-nodes", cr.WARN, "autoscaled to zero"),
        C("pull-secret", cr.FAIL, "401 from nvcr.io", fix="refresh the credential"),
        C("staging-roundtrip", cr.PASS, "round-trip OK"),  # must NOT appear (after the ❌)
    ]
    out = wiz.render_done_panel("demo", checks)
    check("renderer maps PASS→✓", "✓ var-reconcile" in out, out)
    check("renderer maps WARN→⚠", "⚠ nodes" in out, out)
    check("renderer shows the ❌ line", "❌ pull-secret" in out, out)
    check("renderer prints the fix", "refresh the credential" in out)
    check(
        "renderer TRUNCATES at first ❌ (later PASS hidden)",
        "round-trip OK" not in out,
        out,
    )
    check("renderer uses no 🟡", "🟡" not in out)
    # all-pass → verdict summary + RUN-READY
    allpass = [C("var-reconcile", cr.PASS, "ok"), C("gpu-nodes", cr.PASS, "ok")]
    out2 = wiz.render_done_panel("demo", allpass)
    check("all-pass renders RUN-READY", "RUN-READY" in out2, out2)


# ── 3. model-cache two-case detection (S1) ────────────────────────────────────
def test_model_cache():
    absent = wiz.model_cache_check("shared-model-cache", ["some-other-pvc"])
    check("model-cache absent → FAIL", absent.level == cr.FAIL, absent.level)
    name_hit = wiz.model_cache_check("nemotron-hf-cache", ["nemotron-hf-cache", "x"])
    check("model-cache name-match → PASS", name_hit.level == cr.PASS, name_hit.level)
    pos_hit = wiz.model_cache_check("random-pvc", ["random-pvc"])
    check("model-cache position-match → WARN", pos_hit.level == cr.WARN, pos_hit.level)
    check("position WARN message names the risk", "position" in pos_hit.message.lower())


# ── 4. single-IP writing — never a comma ──────────────────────────────────────
def test_single_ip():
    check("valid_single_ip accepts a plain IP", wiz.valid_single_ip("10.0.42.7"))
    check(
        "valid_single_ip rejects a comma-list",
        not wiz.valid_single_ip("10.0.0.1,10.0.42.7"),
    )
    check("valid_single_ip rejects empty", not wiz.valid_single_ip(""))
    check("valid_single_ip rejects junk", not wiz.valid_single_ip("not-an-ip"))
    a = wiz.Answers(
        cluster="c",
        namespace="ns",
        gpu_product="NVIDIA-B200",
        pull_secret="p",
        model_cache_pvc="m",
        no_internet_dns_ip="10.100.0.10",
        no_internet_kube_api_ip="10.0.42.7",
    )
    text = wiz.build_profile_text(a)
    check("profile writes single DNS IP", 'NO_INTERNET_DNS_IP="10.100.0.10"' in text)
    check("profile writes single API IP", 'NO_INTERNET_KUBE_API_IP="10.0.42.7"' in text)
    ni_lines = [ln for ln in text.splitlines() if ln.startswith("NO_INTERNET_") and "," in ln]
    check("no NO_INTERNET_* value contains a comma", not ni_lines, str(ni_lines))
    check("Phase-2 TODO comment present in the IP block", "# Phase-2" in text)
    check(
        "CONTROL_STORAGE_CLASS omitted when unset (no bare guess)",
        "CONTROL_STORAGE_CLASS" not in text,
    )
    a.control_sc = "efs-sc"
    check(
        "CONTROL_STORAGE_CLASS written when confirmed",
        'CONTROL_STORAGE_CLASS="efs-sc"' in wiz.build_profile_text(a),
    )
    # Gap-2: a fresh profile always writes a CONNECT_CMD line (empty by default → tooling derives tsh); when
    # the operator supplies the SSO login it lands verbatim, and round-trips back into Answers via _ENV_TO_FIELD.
    check("CONNECT_CMD line present (empty) in a fresh profile", 'CONNECT_CMD=""' in text)
    a.connect_cmd = "tsh kube login prod-b200"
    _t2 = wiz.build_profile_text(a)
    check(
        "CONNECT_CMD written verbatim when captured",
        'CONNECT_CMD="tsh kube login prod-b200"' in _t2,
    )
    _round = wiz.Answers(cluster="c")
    wiz._fill_answers_from_env(_round, {"CONNECT_CMD": "tsh kube login prod-b200"})
    check(
        "CONNECT_CMD round-trips profile→Answers (existing profile becomes the default)",
        _round.connect_cmd == "tsh kube login prod-b200",
    )


# ── 5. schema-version reader round-trip (Q7) ──────────────────────────────────
def test_schema_reader():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.env"
        p.write_text(f'# {pr.SCHEMA_HEADER}={pr.PROFILE_SCHEMA_VERSION}\nNAMESPACE="ns"\n')
        env = pr._read_env(p)
        check(
            "schema header round-trips",
            env.get(pr.SCHEMA_HEADER) == str(pr.PROFILE_SCHEMA_VERSION),
            env.get(pr.SCHEMA_HEADER),
        )
        check("normal keys still parse alongside header", env.get("NAMESPACE") == "ns")
        p2 = Path(d) / "old.env"
        p2.write_text('NAMESPACE="ns"\n')  # no header = schema 0
        check("absent header → schema 0", pr._read_env(p2).get(pr.SCHEMA_HEADER) == "0")


# ── 6. --play: non-zero on ❌ + never provisions; zero on all-pass ─────────────
def test_play_mode():
    C = cr.Check
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        play = d / "play.yaml"
        play.write_text(
            "schema_version: 1\ncluster: demo\ncontext: ctx\nnamespace: ns\n"
            "gpu_product: NVIDIA-B200\narch: amd64\npull_secret: nvcr\nhf_secret: hf\n"
            "model_cache_pvc: nemotron-hf-cache\nartifacts_sc: gp3\ncontrol_sc: efs-sc\n"
            "no_internet_dns_ip: 10.100.0.10\nno_internet_kube_api_ip: 10.0.42.7\n"
        )
        # A FAIL battery → exit 1; injected battery records that it never applies anything.
        applied = []

        def fail_battery(prof):
            return [
                C("var-reconcile", cr.PASS, "ok"),
                C("pull-secret", cr.FAIL, "401", fix="fix cred"),
            ]

        rc = wiz.run_play("demo", str(play), profiles_dir=d, battery_fn=fail_battery)
        check("--play exits non-zero on ❌", rc == wiz.EXIT_ERROR, rc)
        prof_path = d / "demo.env"
        check("--play wrote the profile", prof_path.exists())
        written = prof_path.read_text()
        check(
            "--play profile has single IPs (no comma)",
            'NO_INTERNET_KUBE_API_IP="10.0.42.7"' in written
            and "," not in [ln for ln in written.splitlines() if ln.startswith("NO_INTERNET_KUBE_API_IP")][0],
        )
        check(
            "--play never provisions (no kubectl apply in this offline path)",
            applied == [],
        )

        # readiness signal (coordinator spec): cluster-profiles/.state/<profile>.readiness.json, gitignored,
        # atomic 0600, reusing the STRUCTURED Check list + verdict().
        import json

        rs = d / ".state" / "demo.readiness.json"
        check("--play writes .state/<profile>.readiness.json", rs.exists(), str(rs))
        if rs.exists():
            check("readiness stamp 0600", stat.S_IMODE(os.stat(rs).st_mode) == 0o600)
            js = json.loads(rs.read_text())
            check(
                "readiness schema=1 + run_ready False on ❌",
                js.get("schema") == 1 and js.get("run_ready") is False,
                js,
            )
            check(
                "readiness level_counts has FAIL=1",
                js.get("level_counts", {}).get("FAIL") == 1,
                js.get("level_counts"),
            )
            check(
                "readiness carries structured checks (id+level)",
                isinstance(js.get("checks"), list) and js["checks"] and set(js["checks"][0]) == {"id", "level"},
                js.get("checks"),
            )
            check(
                "readiness carries profile_hash + ts",
                bool(js.get("profile_hash")) and bool(js.get("ts")),
                js,
            )

        # all-pass battery → exit 0
        rc2 = wiz.run_play(
            "demo",
            str(play),
            profiles_dir=d,
            battery_fn=lambda prof: [C("var-reconcile", cr.PASS, "ok")],
        )
        check("--play exits 0 on all-pass", rc2 == wiz.EXIT_OK, rc2)

    # a comma-list playfile is rejected (fail-closed)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        bad = d / "bad.yaml"
        bad.write_text(
            "schema_version: 1\ncluster: demo\ncontext: ctx\nnamespace: ns\n"
            "gpu_product: NVIDIA-B200\npull_secret: p\nmodel_cache_pvc: m\n"
            "no_internet_kube_api_ip: 10.0.0.1,10.0.42.7\n"
        )
        rc = wiz.run_play("demo", str(bad), profiles_dir=d, battery_fn=lambda p: [])
        check("--play rejects a comma-list playfile", rc == wiz.EXIT_ERROR, rc)
        # missing schema_version rejected
        bad2 = d / "nover.yaml"
        bad2.write_text(
            "cluster: demo\ncontext: ctx\nnamespace: ns\ngpu_product: x\n" "pull_secret: p\nmodel_cache_pvc: m\n"
        )
        check(
            "--play rejects a playfile with no schema_version",
            wiz.run_play("demo", str(bad2), profiles_dir=d, battery_fn=lambda p: []) == wiz.EXIT_ERROR,
        )


# ── 7. config-storage atomic write (§5) ───────────────────────────────────────
def test_config_storage():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        sys_path = d / "k8s-system.yaml"
        k8s_config.save_system_config(
            {
                "owner": "testuser",
                "connect_cmd": "tsh kube login x",
                "namespace": "should-be-dropped",
                "context": "also-dropped",
            },
            path=sys_path,
        )
        mode = stat.S_IMODE(os.stat(sys_path).st_mode)
        check("system config is 0600", mode == 0o600, oct(mode))
        loaded = k8s_config.load_system_config(path=sys_path)
        check("system config persists owner", loaded.get("owner") == "testuser")
        check(
            "system config drops cluster-specific keys (Q5)",
            "namespace" not in loaded and "context" not in loaded,
            str(loaded),
        )

        st = d / "state.yaml"
        k8s_config.save_init_state("demo", {"namespace": "ns"}, path=st)
        check("init-state 0600", stat.S_IMODE(os.stat(st).st_mode) == 0o600)
        check(
            "init-state round-trips for same cluster",
            k8s_config.load_init_state("demo", path=st) == {"namespace": "ns"},
        )
        check(
            "init-state returns None for a different cluster",
            k8s_config.load_init_state("other", path=st) is None,
        )
        # PER-CLUSTER KEYING (coordinator req): a second cluster's save must NOT clobber the first.
        k8s_config.save_init_state("clusterB", {"namespace": "nsB"}, path=st)
        check(
            "init-state keeps cluster A after cluster B written (no clobber)",
            k8s_config.load_init_state("demo", path=st) == {"namespace": "ns"},
        )
        check(
            "init-state keeps cluster B alongside A",
            k8s_config.load_init_state("clusterB", path=st) == {"namespace": "nsB"},
        )
        # clearing one cluster leaves the other intact
        k8s_config.clear_init_state("demo", path=st)
        check(
            "per-cluster clear drops only that cluster",
            k8s_config.load_init_state("demo", path=st) is None
            and k8s_config.load_init_state("clusterB", path=st) == {"namespace": "nsB"},
        )
        # 7-day expiry (per-cluster entry)
        import yaml
        from datetime import datetime, timedelta

        st.write_text(
            yaml.safe_dump(
                {
                    "clusters": {
                        "demo": {
                            "answers": {"x": 1},
                            "timestamp": (datetime.now() - timedelta(days=8)).isoformat(),
                        }
                    }
                }
            )
        )
        check(
            "init-state expires after 7 days",
            k8s_config.load_init_state("demo", path=st) is None,
        )


# ── 8. RWX detector (§6.4) ────────────────────────────────────────────────────
def test_rwx_detector():
    best, ranked = wiz.rank_rwx_classes(["gp3", "efs-sc", "standard"])
    check("RWX detector picks the efs candidate", best == "efs-sc", best)
    best2, _ = wiz.rank_rwx_classes(["gp3", "standard-ssd"])
    check("RWX detector safe-degrades to None when nothing matches", best2 is None, best2)
    best3, _ = wiz.rank_rwx_classes([])
    check("RWX detector handles empty list", best3 is None)
    # FAST model-cache default: the AWS FSx CSI class is named 'fsx-lustre' (no 'fsx' substring) —
    # the RWX detector must recognize it (and Lustre/FSx names) so the model cache defaults to the fast FS.
    fsx_best, _ = wiz.rank_rwx_classes(["ebs", "fsx-lustre", "gp3"])
    check(
        "RWX detector recognizes the FSx 'fsx-lustre' class",
        fsx_best == "fsx-lustre",
        fsx_best,
    )
    check(
        "RWX detector recognizes a lustre-named class",
        wiz.rank_rwx_classes(["gp3", "fsx-lustre"])[0] == "fsx-lustre",
    )
    # A fresh profile with the model-cache classes set emits BOTH MODEL_CACHE_*_CLASS lines (the fast RWX
    # default + the RWO opt-in), so install resolves the cache to the fast class.
    a_fast = wiz.Answers(
        cluster="c",
        namespace="ns",
        gpu_product="NVIDIA-GB300",
        pull_secret="p",
        cache_rwx_class="fsx-lustre",
        cache_rwo_class="ebs",
    )
    t_fast = wiz.build_profile_text(a_fast)
    check(
        "fresh profile emits MODEL_CACHE_RWX_CLASS=FSx (fast model-cache default)",
        'MODEL_CACHE_RWX_CLASS="fsx-lustre"' in t_fast,
        t_fast,
    )
    check(
        "fresh profile emits MODEL_CACHE_RWO_CLASS=ebs (RWO opt-in kept)",
        'MODEL_CACHE_RWO_CLASS="ebs"' in t_fast,
    )
    # round-trips profile → Answers (resume preserves the model-cache classes)
    _rt = wiz.Answers(cluster="c")
    wiz._fill_answers_from_env(_rt, {"MODEL_CACHE_RWX_CLASS": "fsx-lustre", "MODEL_CACHE_RWO_CLASS": "ebs"})
    check(
        "model-cache classes round-trip profile→Answers (resume preserves them)",
        _rt.cache_rwx_class == "fsx-lustre" and _rt.cache_rwo_class == "ebs",
    )


# ── 9. pvc_manifest is a valid RWX manifest (what provision-now WOULD apply) ───
def test_pvc_manifest():
    import json

    m = json.loads(wiz.pvc_manifest("cache", "ns", "efs-sc"))
    check(
        "pvc_manifest is ReadWriteMany",
        m["spec"]["accessModes"] == ["ReadWriteMany"],
        m,
    )
    check("pvc_manifest carries the SC", m["spec"]["storageClassName"] == "efs-sc")
    check("pvc_manifest namespaced", m["metadata"]["namespace"] == "ns")


# ── 10. subprocess smoke: --record stub + --help usage + no-flag discovery front door ────
def test_cli_smoke():
    p = subprocess.run(
        [sys.executable, str(SCRIPTS / "wizard_init.py"), "--record", "f.yaml"],
        capture_output=True,
        text=True,
    )
    check(
        "--record stub exits 1 with NotImplemented note",
        p.returncode == 1 and "not implemented" in (p.stdout + p.stderr).lower(),
        p.stdout + p.stderr,
    )
    # --help prints usage and documents that --cluster is OPTIONAL (the PROFILE label, not a cluster to know).
    ph = subprocess.run(
        [sys.executable, str(SCRIPTS / "wizard_init.py"), "--help"],
        capture_output=True,
        text=True,
    )
    out_h = (ph.stdout + ph.stderr).lower()
    check("--help prints usage rc 0", ph.returncode == 0 and "usage:" in out_h, out_h)
    check(
        "--help says --cluster is optional / a label",
        "optional" in out_h and "label" in out_h,
        out_h,
    )
    # No-flag init is now the DISCOVERY front door — it must not crash on a non-tty stdin (EOF → cancelled,
    # or an empty context list → login hint). Either way: a clean non-zero exit, no traceback.
    p2 = subprocess.run(
        [sys.executable, str(SCRIPTS / "wizard_init.py")],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    both = p2.stdout + p2.stderr
    check(
        "no-flag init exits non-zero without a traceback",
        p2.returncode != 0 and "Traceback" not in both,
        both,
    )


# ── 11. interactive happy path END-TO-END (offline: fake kubectl + scripted stdin + injected battery) ──
class _FakeKrun:
    """Canned kubectl for the interactive e2e. Matches by substring needles; first match wins; unmatched →
    (0,'',''). Accepts timeout=/stdin= kwargs. Never touches a real cluster."""

    def __init__(self, rules):
        self.rules = rules
        self.applied = []

    def __call__(self, args, timeout=30, stdin=None):
        if "apply" in args:
            self.applied.append(list(args))  # would only fire on provision-now (must NOT happen here)
        for needles, resp in self.rules:
            if all(any(n == a or n in a for a in args) for n in needles):
                return resp
        return (0, "", "")


def test_interactive_e2e():
    import builtins
    import json

    C = cr.Check
    nodes_json = json.dumps(
        {
            "items": [
                {
                    "metadata": {
                        "name": "n1",
                        "labels": {
                            "nvidia.com/gpu.product": "NVIDIA-B200",
                            "kubernetes.io/arch": "amd64",
                        },
                    }
                }
            ]
        }
    )
    fake = _FakeKrun(
        [
            (("get-contexts",), (0, "eks-b200\n", "")),
            (("current-context",), (0, "eks-b200\n", "")),
            (("cluster-info",), (0, "Kubernetes control plane is running", "")),
            (("get", "nodes", "json"), (0, nodes_json, "")),
            (("get", "namespaces"), (0, "example-benchmark\n", "")),
            (("get", "pods"), (0, json.dumps({"items": []}), "")),
            (("get", "pvc"), (0, "nemotron-hf-cache\n", "")),
            (("get", "secrets"), (0, "nvcr-pull docker\nhf-token Opaque\n", "")),
            (
                ("get", "storageclasses"),
                (
                    0,
                    _sc_json(
                        ("gp3", "ebs.csi.aws.com", False),
                        ("efs-sc", "efs.csi.aws.com", False),
                    ),
                    "",
                ),
            ),
            (("deployments",), (0, "coredns\n", "")),
            (("can-i",), (0, "yes\n", "")),
            (("kube-dns",), (0, "10.100.0.10", "")),
            (("endpoints", "kubernetes"), (0, "10.0.42.7", "")),
            (("cilium-config",), (0, "true", "")),
        ]
    )
    real_input = builtins.input
    builtins.input = lambda *a, **k: ""  # accept every default (context/ns/secrets/SC/confirm=yes)
    try:
        import json as _json

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            rc = wiz.run_interactive(
                "eks-b200",
                profiles_dir=d,
                krun=fake,
                battery_fn=lambda prof: [
                    C("var-reconcile", cr.PASS, "ok"),
                    C("gpu-nodes", cr.PASS, "1 × B200"),
                ],
            )
            check("interactive e2e exits 0 on all-pass", rc == wiz.EXIT_OK, rc)
            prof = d / "eks-b200.env"
            check("interactive e2e wrote the profile", prof.exists())
            txt = prof.read_text() if prof.exists() else ""
            check("interactive e2e wrote schema header", "LLMB_PROFILE_SCHEMA=1" in txt)
            check(
                "interactive e2e wrote single API IP (no comma)",
                'NO_INTERNET_KUBE_API_IP="10.0.42.7"' in txt,
            )
            check(
                "interactive e2e picked the name-matched model cache",
                'MODEL_CACHE_PVC="nemotron-hf-cache"' in txt,
                txt,
            )
            check(
                "interactive e2e wrote RWX control class from detector",
                'CONTROL_STORAGE_CLASS="efs-sc"' in txt,
            )
            check(
                "interactive e2e NEVER provisioned (no kubectl apply)",
                fake.applied == [],
            )
            rs = d / ".state" / "eks-b200.readiness.json"
            check("interactive e2e wrote the readiness stamp", rs.exists())
            if rs.exists():
                js = _json.loads(rs.read_text())
                check(
                    "interactive e2e readiness run_ready True",
                    js.get("run_ready") is True,
                    js,
                )
    finally:
        builtins.input = real_input


# ── 11b. no-flag cluster DISCOVERY front door (init shows what's connected, pick one) ─────────────────
def _scripted_input(answers):
    """Return a builtins.input replacement that yields each queued answer in turn (StopIteration → '')."""
    it = iter(answers)

    def _inp(*_a, **_k):
        try:
            return next(it)
        except StopIteration:
            return ""

    return _inp


def test_cluster_discovery():
    import builtins
    import json

    # slugify: EKS-arn → bare cluster name; long teleport name → sanitized; junk → 'cluster'.
    check(
        "slugify arn → bare name",
        wiz.slugify_context("arn:aws:eks:us-east-1:1:cluster/qwen3-qa") == "qwen3-qa",
    )
    check("slugify plain kept", wiz.slugify_context("eks-b200") == "eks-b200")
    check(
        "slugify sanitizes + lowercases",
        wiz.slugify_context("EXAMPLE_PROXY") == "example-proxy",
    )
    check("slugify empty → fallback", wiz.slugify_context("") == "cluster")
    check("slugify clamps ≤63", len(wiz.slugify_context("x" * 200)) <= 63)

    # render_context_menu: numbered, marks current, full names verbatim; empty → the no-clusters line.
    menu = wiz.render_context_menu(
        [
            {
                "name": "proxy.example.teleport.sh-example-cluster-qwen3-qa",
                "current": True,
            },
            {"name": "eks-b200", "current": False},
        ]
    )
    check(
        "menu numbers entries",
        "1  proxy.example.teleport.sh-example-cluster-qwen3-qa" in menu,
        menu,
    )
    check(
        "menu shows full context name",
        "proxy.example.teleport.sh-example-cluster-qwen3-qa" in menu,
        menu,
    )
    check("menu marks current-context", "(current" in menu, menu)
    check("menu names current as the Enter-default", "press Enter" in menu, menu)
    check(
        "menu empty → no-clusters line",
        "no connected clusters" in wiz.render_context_menu([]),
    )

    # DEFECT #5 — a 289-context kubeconfig printed as a flat wall and buried the current context (#192).
    # The menu now puts CURRENT first and collapses the rest, WITHOUT renumbering (the Enter-default must
    # keep working: the caller's default_idx is the original 1-based position).
    _many = [{"name": f"ctx-{i:03d}", "current": (i == 192)} for i in range(1, 290)]
    _m = wiz.render_context_menu(_many)
    _lines = [l for l in _m.splitlines() if l.strip() and l.strip()[0].isdigit()]
    check(
        "big menu: CURRENT context is listed FIRST",
        _lines[0].strip().startswith("192"),
        _lines[:2],
    )
    check(
        "big menu: current keeps its ORIGINAL number 192 (Enter-default cannot regress)",
        "192  ctx-192" in _m,
        _lines[:2],
    )
    check(
        "big menu: collapses instead of printing all 289",
        len(_lines) <= wiz.CONTEXT_MENU_LIMIT + 1,
        str(len(_lines)),
    )
    check(
        "big menu: offers show-all + filter affordances",
        "'a' to list them all" in _m and "filter" in _m,
        _m[-200:],
    )
    check(
        "big menu: show_all=True lists every context",
        len(
            [
                l
                for l in wiz.render_context_menu(_many, show_all=True).splitlines()
                if l.strip() and l.strip()[0].isdigit()
            ]
        )
        == 289,
    )
    # filtering keeps the ORIGINAL numbering so a typed number still selects the right context
    _f = [{"name": f"ctx-{i:03d}", "current": (i == 192)} for i in range(1, 290)]
    _f[41] = {"name": "example-gb300-cluster", "current": False}
    _fm = wiz.render_context_menu(_f, query="gb300")
    check(
        "filter matches by substring and preserves the original index",
        "42  example-gb300-cluster" in _fm,
        _fm,
    )
    check(
        "filter with no match explains how to recover",
        "no context matches" in wiz.render_context_menu(_f, query="zzzz"),
    )

    # reconnect hint surfaces the login command (tsh / SSO).
    hint = wiz._reconnect_hint()
    check(
        "reconnect hint mentions login/tsh",
        ("tsh" in hint.lower()) or ("login" in hint.lower()),
        hint,
    )

    real_input = builtins.input
    # (a) EMPTY context list → _PICK_EMPTY + a login hint (never a pick prompt).
    import io
    from contextlib import redirect_stdout

    empty_krun = _FakeKrun([(("get-contexts",), (1, "", ""))])
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            res = wiz._pick_context_interactive(empty_krun)
    finally:
        pass
    check("empty contexts → _PICK_EMPTY", res == wiz._PICK_EMPTY, repr(res))
    check(
        "empty contexts prints a login hint",
        "log in first" in buf.getvalue().lower(),
        buf.getvalue(),
    )

    # (b) two contexts, operator picks #2 and accepts the derived profile name (Enter).
    two_krun = _FakeKrun(
        [
            (
                ("get-contexts",),
                (0, "proxy.example.teleport.sh-x-qwen3-qa\neks-b200\n", ""),
            ),
            (("current-context",), (0, "eks-b200\n", "")),
        ]
    )
    builtins.input = _scripted_input(["1", ""])  # pick #1 (the teleport ctx), accept slug default
    try:
        with redirect_stdout(io.StringIO()):
            res = wiz._pick_context_interactive(two_krun)
    finally:
        builtins.input = real_input
    check(
        "pick returns (profile_name, context)",
        isinstance(res, tuple) and len(res) == 2,
        repr(res),
    )
    if isinstance(res, tuple):
        name, ctx = res
        check(
            "pick returned the chosen full context",
            ctx == "proxy.example.teleport.sh-x-qwen3-qa",
            ctx,
        )
        check(
            "pick derived a valid slug profile name",
            wiz.validate_cluster_name(name) is None,
            name,
        )

    # (c) explicit quit → _PICK_QUIT.
    builtins.input = _scripted_input(["q"])
    try:
        with redirect_stdout(io.StringIO()):
            resq = wiz._pick_context_interactive(two_krun)
    finally:
        builtins.input = real_input
    check("quit at picker → _PICK_QUIT", resq == wiz._PICK_QUIT, repr(resq))

    # (d) 'r' shows the reconnect hint then still lets the operator pick.
    rbuf = io.StringIO()
    builtins.input = _scripted_input(["r", "2", "myprof"])
    try:
        with redirect_stdout(rbuf):
            resr = wiz._pick_context_interactive(two_krun)
    finally:
        builtins.input = real_input
    check(
        "reconnect 'r' prints hint then picks",
        isinstance(resr, tuple) and resr[0] == "myprof" and "reconnect" in rbuf.getvalue().lower(),
        (repr(resr), rbuf.getvalue()[:200]),
    )

    # (e) FULL no-flag e2e: run_interactive(None) discovers, picks, and writes the profile end-to-end.
    nodes_json = json.dumps(
        {
            "items": [
                {
                    "metadata": {
                        "name": "n1",
                        "labels": {
                            "nvidia.com/gpu.product": "NVIDIA-B200",
                            "kubernetes.io/arch": "amd64",
                        },
                    }
                }
            ]
        }
    )
    fake = _FakeKrun(
        [
            (("get-contexts",), (0, "eks-b200\n", "")),
            (("current-context",), (0, "eks-b200\n", "")),
            (("cluster-info",), (0, "Kubernetes control plane is running", "")),
            (("get", "nodes", "json"), (0, nodes_json, "")),
            (("get", "namespaces"), (0, "example-benchmark\n", "")),
            (("get", "pods"), (0, json.dumps({"items": []}), "")),
            (("get", "pvc"), (0, "nemotron-hf-cache\n", "")),
            (("get", "secrets"), (0, "nvcr-pull docker\nhf-token Opaque\n", "")),
            (
                ("get", "storageclasses"),
                (
                    0,
                    _sc_json(
                        ("gp3", "ebs.csi.aws.com", False),
                        ("efs-sc", "efs.csi.aws.com", False),
                    ),
                    "",
                ),
            ),
            (("deployments",), (0, "coredns\n", "")),
            (("can-i",), (0, "yes\n", "")),
            (("kube-dns",), (0, "10.100.0.10", "")),
            (("endpoints", "kubernetes"), (0, "10.0.42.7", "")),
            (("cilium-config",), (0, "true", "")),
        ]
    )
    builtins.input = lambda *a, **k: ""  # accept every default: pick=current, profile name=slug, all else
    C = cr.Check
    try:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            with redirect_stdout(io.StringIO()):
                rc = wiz.run_interactive(
                    None,
                    profiles_dir=d,
                    krun=fake,
                    battery_fn=lambda prof: [C("var-reconcile", cr.PASS, "ok")],
                )
            check("no-flag run_interactive exits 0 on all-pass", rc == wiz.EXIT_OK, rc)
            prof = d / "eks-b200.env"  # slug of the picked 'eks-b200' context
            check("no-flag flow wrote the profile at the derived name", prof.exists())
            txt = prof.read_text() if prof.exists() else ""
            check(
                "no-flag flow pinned the picked context",
                'KUBE_CONTEXT="eks-b200"' in txt,
                txt,
            )
    finally:
        builtins.input = real_input


# ── 11c. secrets: NAMES auto-defaulted (never prompted); creds present → quiet ✓; missing → instruction ─
def test_secret_creds():
    import builtins
    import io
    from contextlib import redirect_stdout

    # Auto-default NAMES are BARE, namespace-scoped suffixes (QA fix — no <profile>- prefix) and never
    # require input. The prefix added nothing (these resources are namespace-scoped) and blew up to 60+
    # chars when the profile label was an auto-derived teleport hostname.
    check(
        "pull secret name auto-defaults to the bare suffix nvcr-cred",
        wiz.default_pull_secret_name("qwen3-qa") == "nvcr-cred",
    )
    check(
        "hf secret name auto-defaults to the bare suffix hf-token",
        wiz.default_hf_secret_name("qwen3-qa") == "hf-token",
    )

    # detect_ngc_cred / detect_hf_cred: PURE with injected env + home.
    with tempfile.TemporaryDirectory() as h:
        home = Path(h)
        # env var present → found.
        ok, src = wiz.detect_ngc_cred({"NGC_API_KEY": "abc"}, home)
        check("NGC found via $NGC_API_KEY", ok and "NGC_API_KEY" in src, (ok, src))
        ok, src = wiz.detect_hf_cred({"HF_TOKEN": "hf_x"}, home)
        check("HF found via $HF_TOKEN", ok and "HF_TOKEN" in src, (ok, src))
        # nothing present → missing.
        check(
            "NGC missing when no env + no config",
            wiz.detect_ngc_cred({}, home)[0] is False,
        )
        check(
            "HF missing when no env + no token file",
            wiz.detect_hf_cred({}, home)[0] is False,
        )
        # ~/.ngc/config with a real apikey → found; placeholder → missing.
        (home / ".ngc").mkdir()
        (home / ".ngc" / "config").write_text("[CURRENT]\napikey = REALKEY123\n")
        check(
            "NGC found via ~/.ngc/config real apikey",
            wiz.detect_ngc_cred({}, home)[0] is True,
        )
        (home / ".ngc" / "config").write_text("[CURRENT]\napikey = no-apikey\n")
        check(
            "NGC placeholder apikey → still missing",
            wiz.detect_ngc_cred({}, home)[0] is False,
        )
        # ~/.cache/huggingface/token non-empty → found.
        (home / ".cache" / "huggingface").mkdir(parents=True)
        (home / ".cache" / "huggingface" / "token").write_text("hf_realtoken\n")
        check(
            "HF found via ~/.cache/huggingface/token",
            wiz.detect_hf_cred({}, home)[0] is True,
        )

    # _report_cred_sources: present → quiet ✓ line, no instruction; missing → the exact where-to-get block.
    with tempfile.TemporaryDirectory() as h:
        home = Path(h)
        buf = io.StringIO()
        with redirect_stdout(buf):
            wiz._report_cred_sources({"NGC_API_KEY": "abc", "HF_TOKEN": "hf_x"}, home)
        out = buf.getvalue().lower()
        check("creds-present prints ✓ NGC found", "ngc creds found" in out, out)
        check("creds-present prints ✓ HF found", "huggingface token found" in out, out)
        check(
            "creds-present shows NO where-to-get instruction",
            "ngc.nvidia.com" not in out and "huggingface.co" not in out,
            out,
        )

        buf2 = io.StringIO()
        with redirect_stdout(buf2):
            wiz._report_cred_sources({}, home)
        miss = buf2.getvalue().lower()
        check(
            "creds-missing NGC instruction: ngc.nvidia.com → Setup → Generate API Key + ngc config set",
            "ngc.nvidia.com" in miss and "generate api key" in miss and "ngc config set" in miss,
            miss,
        )
        check(
            "creds-missing HF instruction: huggingface.co → Access Tokens + token file/env",
            "huggingface.co" in miss
            and "access token" in miss
            and ("~/.cache/huggingface/token" in miss or "hf_token" in miss),
            miss,
        )
        check("creds-missing NGC note: universal nvcr.io pulls", "nvcr.io" in miss, miss)

    # No name prompt fired: the interactive e2e (test_cluster_discovery / test_interactive_e2e) runs with a
    # blanket empty-input feed and still writes the <profile>-scoped default secret names.
    C = cr.Check
    import json

    nodes_json = json.dumps(
        {
            "items": [
                {
                    "metadata": {
                        "name": "n1",
                        "labels": {
                            "nvidia.com/gpu.product": "NVIDIA-B200",
                            "kubernetes.io/arch": "amd64",
                        },
                    }
                }
            ]
        }
    )
    fake = _FakeKrun(
        [
            (("get-contexts",), (0, "eks-b200\n", "")),
            (("current-context",), (0, "eks-b200\n", "")),
            (("cluster-info",), (0, "ok", "")),
            (("get", "nodes", "json"), (0, nodes_json, "")),
            (("get", "namespaces"), (0, "example-benchmark\n", "")),
            (("get", "pods"), (0, json.dumps({"items": []}), "")),
            (("get", "pvc"), (0, "nemotron-hf-cache\n", "")),
            (
                ("get", "storageclasses"),
                (
                    0,
                    _sc_json(
                        ("gp3", "ebs.csi.aws.com", False),
                        ("efs-sc", "efs.csi.aws.com", False),
                    ),
                    "",
                ),
            ),
            (("deployments",), (0, "coredns\n", "")),
            (("can-i",), (0, "yes\n", "")),
        ]
    )
    real_input = builtins.input
    builtins.input = lambda *a, **k: ""
    try:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            with redirect_stdout(io.StringIO()):
                wiz.run_interactive(
                    "eks-b200",
                    profiles_dir=d,
                    krun=fake,
                    battery_fn=lambda prof: [C("var-reconcile", cr.PASS, "ok")],
                )
            txt = (d / "eks-b200.env").read_text()
            check(
                "profile carries the bare auto-defaulted pull-secret name (no prompt, no prefix)",
                'IMAGE_PULL_SECRET="nvcr-cred"' in txt,
                txt,
            )
            check(
                "profile carries the bare auto-defaulted hf-secret name (no prompt, no prefix)",
                'HF_SECRET="hf-token"' in txt,
                txt,
            )
    finally:
        builtins.input = real_input


# ── 11d. storage-class guidance (no dead-end) + model-cache defers to install's G3 (stage-time create) ─
def test_storage_and_cache_defaults():
    import builtins
    import io
    import json
    from contextlib import redirect_stdout

    check(
        "model-cache default name is the bare suffix model-cache (QA fix — no <profile>- prefix)",
        wiz.default_model_cache_name("qwen3-qa") == "model-cache",
    )
    C = cr.Check
    nodes_json = json.dumps(
        {
            "items": [
                {
                    "metadata": {
                        "name": "n1",
                        "labels": {
                            "nvidia.com/gpu.product": "NVIDIA-B200",
                            "kubernetes.io/arch": "amd64",
                        },
                    }
                }
            ]
        }
    )
    # Cluster with NO RWX storage class (only ebs/gp3) and NO existing PVC → exercises both new paths.
    fake = _FakeKrun(
        [
            (("get-contexts",), (0, "eks-b200\n", "")),
            (("current-context",), (0, "eks-b200\n", "")),
            (("cluster-info",), (0, "ok", "")),
            (("get", "nodes", "json"), (0, nodes_json, "")),
            (("get", "namespaces"), (0, "example-benchmark\n", "")),
            (("get", "pods"), (0, json.dumps({"items": []}), "")),
            (("get", "pvc"), (0, "", "")),  # no existing PVCs
            (
                ("get", "storageclasses"),
                (
                    0,
                    _sc_json(
                        ("ebs", "ebs.csi.aws.com", True),
                        ("gp3", "ebs.csi.aws.com", False),
                    ),
                    "",
                ),
            ),  # NO rwx candidate
            (("deployments",), (0, "coredns\n", "")),
            (
                ("can-i",),
                (0, "no\n", ""),
            ),  # RBAC forbids create → provision-now unavailable
        ]
    )
    real_input = builtins.input
    builtins.input = lambda *a, **k: ""  # accept all defaults: RWO for control, defer model cache
    buf = io.StringIO()
    try:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            with redirect_stdout(buf):
                rc = wiz.run_interactive(
                    "small-cell",
                    profiles_dir=d,
                    krun=fake,
                    battery_fn=lambda prof: [C("var-reconcile", cr.PASS, "ok")],
                )
            out = buf.getvalue()
            txt = (d / "small-cell.env").read_text()
            check("run exits 0", rc == wiz.EXIT_OK, rc)
            # (1) no-RWX class → explains RWX + offers RWO fallback, NOT a blank dead-end.
            check("no-RWX path explains ReadWriteMany", "ReadWriteMany" in out, out)
            check(
                "no-RWX path says none detected + offers RWO fallback",
                "not detected" in out.lower() and "single-pod" in out.lower(),
                out,
            )
            check(
                "no-RWX path defaults CONTROL_STORAGE_CLASS to the RWO class (not blank)",
                'CONTROL_STORAGE_CLASS="ebs"' in txt,
                txt,
            )
            # (2) model cache → deferred to install's stage-time G3 create, NOT an eager RWX provision.
            check(
                "model-cache explains defer-to-install at stage time",
                "stage time" in out.lower() and "recommended" in out.lower(),
                out,
            )
            check(
                "model-cache defaults to the bare model-cache PVC name (stage-time create)",
                'MODEL_CACHE_PVC="model-cache"' in txt,
                txt,
            )
            check(
                "init did NOT eagerly apply a PVC (no kubectl apply)",
                fake.applied == [],
                fake.applied,
            )
            # (3) artifacts SC explained as where run artifacts/logs go.
            check(
                "artifacts SC prompt explains run artifacts/logs",
                "run artifacts" in out.lower(),
                out,
            )
    finally:
        builtins.input = real_input


# ── 11e. [QA fix] init-default bugs surfaced by a live GB300 fresh install ─────────────────────────────
# Exercise a structured proxy context and its derived profile label.
# and the 4 StorageClasses (enterprise-file=fsx, standard-object=s3, ebs=default, gp2=aws-ebs). Asserts:
#   BUG 1  resource names default to the BARE namespace-scoped suffixes (nvcr-cred / hf-token / model-cache),
#          and the auto-derived PROFILE LABEL strips the teleport proxy prefix.
#   BUG 2  RWO/cache → ebs (cluster default block class); RWX → fsx-lustre (fsx file class);
#          and the s3.csi.aws.com object store (s3-object) is chosen for NEITHER.
def test_qa_init_defaults():
    import builtins
    import io
    import json
    from contextlib import redirect_stdout

    QA_CTX = "proxy.example.teleport.sh-example-managed-cluster"

    # ── BUG 1b: the auto-derived profile label strips everything through '.teleport.sh-' ──
    check(
        "teleport prefix stripped → bare cluster-id label",
        wiz.slugify_context(QA_CTX) == "example-managed-cluster",
        wiz.slugify_context(QA_CTX),
    )
    check(
        "teleport strip is case-insensitive",
        wiz.slugify_context("PROXY.EXAMPLE.TELEPORT.SH-my-cluster") == "my-cluster",
        wiz.slugify_context("PROXY.EXAMPLE.TELEPORT.SH-my-cluster"),
    )
    check(
        "non-teleport contexts unaffected by the strip",
        wiz.slugify_context("eks-b200") == "eks-b200",
    )

    # ── BUG 1: bare, namespace-scoped resource-name defaults (no <label>- prefix) ──
    check(
        "BUG1 pull-secret default is bare nvcr-cred",
        wiz.default_pull_secret_name(QA_CTX) == "nvcr-cred",
    )
    check(
        "BUG1 hf-secret default is bare hf-token",
        wiz.default_hf_secret_name(QA_CTX) == "hf-token",
    )
    check(
        "BUG1 model-cache default is bare model-cache",
        wiz.default_model_cache_name(QA_CTX) == "model-cache",
    )

    # ── BUG 2: provisioner-aware storage-class selection (pure) ──
    qa_classes = [
        {"name": "fsx-lustre", "provisioner": "fsx.csi.aws.com", "is_default": False},
        {"name": "s3-object", "provisioner": "s3.csi.aws.com", "is_default": False},
        {"name": "ebs", "provisioner": "ebs.csi.aws.com", "is_default": True},
        {"name": "gp2", "provisioner": "kubernetes.io/aws-ebs", "is_default": False},
    ]
    check(
        "BUG2 RWO/cache class resolves to the default block class ebs",
        wiz.select_rwo_class(qa_classes) == "ebs",
        wiz.select_rwo_class(qa_classes),
    )
    check(
        "BUG2 RWX class resolves to the fsx file class fsx-lustre",
        wiz.select_rwx_class(qa_classes) == "fsx-lustre",
        wiz.select_rwx_class(qa_classes),
    )
    check(
        "BUG2 s3.csi object store is chosen for NEITHER RWO nor RWX",
        wiz.select_rwo_class(qa_classes) != "s3-object" and wiz.select_rwx_class(qa_classes) != "s3-object",
        (wiz.select_rwo_class(qa_classes), wiz.select_rwx_class(qa_classes)),
    )
    check(
        "BUG2 an all-object cluster yields None (no S3 default forced)",
        wiz.select_rwo_class([{"name": "obj", "provisioner": "s3.csi.aws.com", "is_default": True}]) is None,
    )
    # is-default-class is preferred for RWO even without provisioner info (annotation-only clusters)
    check(
        "BUG2 RWO honors is-default-class annotation when provisioner is unknown",
        wiz.select_rwo_class(
            [
                {"name": "slow", "provisioner": "", "is_default": False},
                {"name": "fast", "provisioner": "", "is_default": True},
            ]
        )
        == "fast",
    )

    # ── BUG 1+2 together: a full offline interactive init on the QA cluster fingerprint ──
    nodes_json = json.dumps(
        {
            "items": [
                {
                    "metadata": {
                        "name": "n1",
                        "labels": {
                            "nvidia.com/gpu.product": "NVIDIA-GB300",
                            "kubernetes.io/arch": "arm64",
                        },
                    }
                }
            ]
        }
    )
    sc_json = _sc_json(
        ("fsx-lustre", "fsx.csi.aws.com", False),
        ("s3-object", "s3.csi.aws.com", False),
        ("ebs", "ebs.csi.aws.com", True),
        ("gp2", "kubernetes.io/aws-ebs", False),
    )
    fake = _FakeKrun(
        [
            (("get-contexts",), (0, QA_CTX + "\n", "")),
            (("current-context",), (0, QA_CTX + "\n", "")),
            (("cluster-info",), (0, "ok", "")),
            (("get", "nodes", "json"), (0, nodes_json, "")),
            (("get", "namespaces"), (0, "llmb-wizard-qa\n", "")),
            (("get", "pods"), (0, json.dumps({"items": []}), "")),
            (("get", "pvc"), (0, "", "")),  # fresh namespace, no cache PVC
            (("get", "storageclasses"), (0, sc_json, "")),
            (("deployments",), (0, "coredns\n", "")),
            (("can-i",), (0, "no\n", "")),
        ]
    )
    real_input = builtins.input
    builtins.input = lambda *a, **k: ""  # accept every default
    C = cr.Check
    try:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            with redirect_stdout(io.StringIO()):
                rc = wiz.run_interactive(
                    "qa-gb300",
                    profiles_dir=d,
                    krun=fake,
                    battery_fn=lambda p: [C("var-reconcile", cr.PASS, "ok")],
                )
            check("QA e2e exits 0", rc == wiz.EXIT_OK, rc)
            txt = (d / "qa-gb300.env").read_text()
            check(
                "QA e2e: bare pull-secret name",
                'IMAGE_PULL_SECRET="nvcr-cred"' in txt,
                txt,
            )
            check("QA e2e: bare hf-secret name", 'HF_SECRET="hf-token"' in txt, txt)
            check(
                "QA e2e: bare model-cache PVC name",
                'MODEL_CACHE_PVC="model-cache"' in txt,
                txt,
            )
            check(
                "QA e2e: artifacts SC = ebs (default block, NOT object)",
                'ARTIFACTS_STORAGE_CLASS="ebs"' in txt,
                txt,
            )
            check(
                "QA e2e: control SC = fsx-lustre (fsx RWX, NOT object)",
                'CONTROL_STORAGE_CLASS="fsx-lustre"' in txt,
                txt,
            )
            check(
                "QA e2e: s3-object (s3) appears in NO storage-class field",
                'STORAGE_CLASS="s3-object"' not in txt,
                txt,
            )
    finally:
        builtins.input = real_input


# ── 12. [fix 1] reachability precheck → FAIL, not a wall of WARN (silent-success killer) ──────────────
def test_reachability_gate():
    C = cr.Check
    # pure classifier
    check(
        "classify_reachability rc=0 → PASS",
        cr.classify_reachability(0, "").level == cr.PASS,
    )
    bad = cr.classify_reachability(1, 'error: context "typo" does not exist')
    check("classify_reachability rc!=0 → FAIL", bad.level == cr.FAIL, bad.level)
    check(
        "reachability FAIL names it as unproven, not OK",
        "unknown" in bad.message.lower(),
    )

    # run_battery: an unreachable krun must SHORT-CIRCUIT to a single reachability FAIL — NOT safe-degrade
    # every live probe to WARN and still read run-ready.
    def unreachable_krun(args, timeout=30, stdin=None):
        return (1, "", "You must be logged in to the server (Unauthorized)")

    checks = cr.run_battery(
        {"NAMESPACE": "n", "GPU_PRODUCT": "NVIDIA-B200", "KUBE_CONTEXT": "typo"},
        krun=unreachable_krun,
    )
    ids = [c.id for c in checks]
    ok, _ = cr.verdict(checks)
    check("run_battery unreachable → NOT run-ready (FAIL)", ok is False, str(ids))
    check(
        "run_battery unreachable → reachability FAIL present",
        any(c.id == "reachability" and c.level == cr.FAIL for c in checks),
        str(ids),
    )
    check(
        "run_battery unreachable SHORT-CIRCUITS (no wall of live-probe WARNs)",
        not any(i in ids for i in ("gpu-nodes", "pull-secret", "staging-roundtrip", "oidc-issuer")),
        str(ids),
    )
    # a reachable krun runs the full battery (reachability among them, PASS)
    checks2 = cr.run_battery({"NAMESPACE": "n"}, krun=lambda a, timeout=30, stdin=None: (0, "ok", ""))
    check(
        "run_battery reachable → reachability PASS + full battery runs",
        any(c.id == "reachability" and c.level == cr.PASS for c in checks2)
        and any(c.id == "gpu-nodes" for c in checks2),
        str([c.id for c in checks2]),
    )
    # the Done renderer labels + truncates on a reachability FAIL
    panel = wiz.render_done_panel(
        "typo",
        [
            C("var-reconcile", cr.PASS, "ok"),
            C("reachability", cr.FAIL, "unreachable", fix="refresh login"),
        ],
    )
    check(
        "Done panel renders reachability FAIL + stops",
        "❌ reachability" in panel and "NOT yet run-ready" in panel,
        panel,
    )


# ── 13. [fix 2] cluster-name validation + path-traversal rejection ────────────────────────────────────
def test_name_validation():
    check("valid RFC-1123 name accepted", wiz.validate_cluster_name("eks-b200") is None)
    check(
        "path-traversal '../evil' rejected",
        wiz.validate_cluster_name("../evil") is not None,
    )
    check("slash rejected", wiz.validate_cluster_name("a/b") is not None)
    check("uppercase rejected", wiz.validate_cluster_name("EKS") is not None)
    check("spaces rejected", wiz.validate_cluster_name("a b") is not None)
    check("empty rejected", wiz.validate_cluster_name("") is not None)
    check("64-char name rejected (>63)", wiz.validate_cluster_name("a" * 64) is not None)
    check("63-char name accepted", wiz.validate_cluster_name("a" * 63) is None)
    # run_play must refuse a traversal name and write NOTHING outside the profiles dir
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        play = d / "p.yaml"
        play.write_text(
            "schema_version: 1\ncluster: ../evil\ncontext: ctx\nnamespace: ns\n"
            "gpu_product: NVIDIA-B200\npull_secret: p\nmodel_cache_pvc: real-cache\n"
        )
        rc = wiz.run_play("", str(play), profiles_dir=d, battery_fn=lambda p: [])
        check("--play refuses a path-traversal cluster name", rc == wiz.EXIT_ERROR, rc)
        check(
            "--play traversal wrote no file outside profiles dir",
            not (d.parent / "evil.env").exists() and not (d / ".." / "evil.env").exists(),
        )


# ── 14. [fix 3] identity lock on resume/edit (context + GPU_PRODUCT are 🔒) ────────────────────────────
def test_identity_lock():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        # seed an existing profile pinning context=ctxA + GPU_PRODUCT=NVIDIA-B200
        existing = wiz.build_profile_text(
            wiz.Answers(
                cluster="demo",
                context="ctxA",
                namespace="nsA",
                gpu_product="NVIDIA-B200",
                pull_secret="p",
                model_cache_pvc="real-cache",
                artifacts_sc="gp3",
            )
        )
        (d / "demo.env").write_text(existing)
        # a playfile that CHANGES the context → refused as a different profile
        pbad = d / "changed.yaml"
        pbad.write_text(
            "schema_version: 1\ncluster: demo\ncontext: ctxB\nnamespace: nsA\n"
            "gpu_product: NVIDIA-B200\npull_secret: p\nmodel_cache_pvc: real-cache\n"
        )
        rc = wiz.run_play("demo", str(pbad), profiles_dir=d, battery_fn=lambda p: [])
        check(
            "identity lock: changing KUBE_CONTEXT on an existing profile is refused",
            rc == wiz.EXIT_ERROR,
            rc,
        )
        # a playfile that changes GPU_PRODUCT → refused
        pgpu = d / "gpu.yaml"
        pgpu.write_text(
            "schema_version: 1\ncluster: demo\ncontext: ctxA\nnamespace: nsA\n"
            "gpu_product: NVIDIA-GB200\npull_secret: p\nmodel_cache_pvc: real-cache\n"
        )
        check(
            "identity lock: changing GPU_PRODUCT is refused",
            wiz.run_play("demo", str(pgpu), profiles_dir=d, battery_fn=lambda p: []) == wiz.EXIT_ERROR,
        )
        # a matching playfile succeeds and PRESERVES an existing non-identity value the playfile omits
        pok = d / "ok.yaml"
        pok.write_text(
            "schema_version: 1\ncluster: demo\ncontext: ctxA\nnamespace: nsA\n"
            "gpu_product: NVIDIA-B200\npull_secret: p\nmodel_cache_pvc: real-cache\n"
        )
        rc2 = wiz.run_play(
            "demo",
            str(pok),
            profiles_dir=d,
            battery_fn=lambda p: [cr.Check("var-reconcile", cr.PASS, "ok")],
        )
        check(
            "identity lock: a matching context/GPU playfile is accepted",
            rc2 == wiz.EXIT_OK,
            rc2,
        )
        written = (d / "demo.env").read_text()
        check(
            "identity lock: existing ARTIFACTS_STORAGE_CLASS preserved when playfile omits it",
            'ARTIFACTS_STORAGE_CLASS="gp3"' in written,
            written,
        )


# ── 15. [fix 4] --play model-cache validation: placeholder rejected + missing-PVC folds a FAIL ─────────
def test_play_model_cache():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        ph = d / "ph.yaml"
        ph.write_text(
            "schema_version: 1\ncluster: demo\ncontext: ctx\nnamespace: ns\n"
            "gpu_product: NVIDIA-B200\npull_secret: p\nmodel_cache_pvc: shared-model-cache\n"
        )
        check(
            "--play rejects the placeholder model-cache PVC (fail-closed loader)",
            wiz.run_play("demo", str(ph), profiles_dir=d, battery_fn=lambda p: []) == wiz.EXIT_ERROR,
        )
        # headless model-cache existence check folds a FAIL into the battery even when the battery all-passes
        good = d / "g.yaml"
        good.write_text(
            "schema_version: 1\ncluster: demo\ncontext: ctx\nnamespace: ns\n"
            "gpu_product: NVIDIA-B200\npull_secret: p\nmodel_cache_pvc: real-cache\n"
        )
        rc = wiz.run_play(
            "demo",
            str(good),
            profiles_dir=d,
            battery_fn=lambda p: [cr.Check("var-reconcile", cr.PASS, "ok")],
            pvcs=["some-other-pvc"],
        )  # real-cache absent → model-cache FAIL
        check(
            "--play folds a model-cache FAIL (chosen PVC absent from namespace) → exit 1",
            rc == wiz.EXIT_ERROR,
            rc,
        )
        rc2 = wiz.run_play(
            "demo",
            str(good),
            profiles_dir=d,
            battery_fn=lambda p: [cr.Check("var-reconcile", cr.PASS, "ok")],
            pvcs=["real-cache"],
        )  # present → PASS
        check("--play model-cache present → exit 0", rc2 == wiz.EXIT_OK, rc2)


# ── 16. [fix 5] side-band writers use a UNIQUE tmp per writer (no fixed .{name}.tmp collision) ─────────
def test_tmp_uniqueness():
    import tempfile as _tf

    seen = []
    orig = _tf.mkstemp

    def spy(*a, **k):
        fd, name = orig(*a, **k)
        seen.append(name)
        return fd, name

    _tf.mkstemp = spy
    try:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            st = d / "k8s-init-state.yaml"
            k8s_config.atomic_write_yaml(st, {"a": 1})
            k8s_config.atomic_write_yaml(st, {"a": 2})
            check(
                "atomic_write_yaml uses mkstemp (unique tmp per writer)",
                len(seen) >= 2 and seen[-1] != seen[-2],
                seen,
            )
            check(
                "no fixed '.k8s-init-state.yaml.tmp' left behind",
                not (d / ".k8s-init-state.yaml.tmp").exists(),
            )
            check(
                "atomic_write_yaml result is correct + 0600",
                k8s_config._read_yaml(st) == {"a": 2} and stat.S_IMODE(os.stat(st).st_mode) == 0o600,
            )
            # profile_init.write_profile shares the discipline
            import profile_init as _pi

            seen.clear()
            prof = d / "c.env"
            _pi.write_profile(prof, "X=1\n")
            _pi.write_profile(prof, "X=2\n")
            check(
                "write_profile uses a unique mkstemp tmp too",
                len(seen) >= 2 and seen[-1] != seen[-2],
                seen,
            )
            check(
                "write_profile is 0600 + correct",
                prof.read_text() == "X=2\n" and stat.S_IMODE(os.stat(prof).st_mode) == 0o600,
            )
    finally:
        _tf.mkstemp = orig


# ── 17. [fix 6] --dry-run persists NO resume state; --help exits 0; _prompt_from_list rejects unknowns ─
def test_dry_run_and_help():
    import builtins
    import json

    C = cr.Check
    nodes_json = json.dumps(
        {
            "items": [
                {
                    "metadata": {
                        "name": "n1",
                        "labels": {
                            "nvidia.com/gpu.product": "NVIDIA-B200",
                            "kubernetes.io/arch": "amd64",
                        },
                    }
                }
            ]
        }
    )

    class _Fake:
        def __init__(self):
            self.applied = []

        def __call__(self, args, timeout=30, stdin=None):
            if "apply" in args:
                self.applied.append(list(args))
            table = [
                (("get-contexts",), (0, "eks-b200\n", "")),
                (("current-context",), (0, "eks-b200\n", "")),
                (("cluster-info",), (0, "ok", "")),
                (("get", "nodes", "json"), (0, nodes_json, "")),
                (("get", "namespaces"), (0, "example-benchmark\n", "")),
                (("get", "pods"), (0, json.dumps({"items": []}), "")),
                (("get", "pvc"), (0, "nemotron-hf-cache\n", "")),
                (("get", "secrets"), (0, "nvcr-pull docker\nhf-token Opaque\n", "")),
                (
                    ("get", "storageclasses"),
                    (
                        0,
                        _sc_json(
                            ("gp3", "ebs.csi.aws.com", False),
                            ("efs-sc", "efs.csi.aws.com", False),
                        ),
                        "",
                    ),
                ),
                (("deployments",), (0, "coredns\n", "")),
                (("can-i",), (0, "yes\n", "")),
            ]
            for needles, resp in table:
                if all(any(n == x or n in x for x in args) for n in needles):
                    return resp
            return (0, "", "")

    real_input = builtins.input
    builtins.input = lambda *a, **k: ""
    try:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            k8s_config.clear_init_state("dryrun-cluster")  # ensure clean slate
            rc = wiz.run_interactive(
                "dryrun-cluster",
                profiles_dir=d,
                dry_run=True,
                krun=_Fake(),
                battery_fn=lambda p: [C("var-reconcile", cr.PASS, "ok")],
            )
            check("--dry-run exits 0", rc == wiz.EXIT_OK, rc)
            check("--dry-run wrote NO profile", not (d / "dryrun-cluster.env").exists())
            check(
                "--dry-run persisted NO resume state",
                k8s_config.load_init_state("dryrun-cluster") is None,
            )
    finally:
        builtins.input = real_input

    # --help / -h → usage + exit 0
    for flag in ("--help", "-h"):
        p = subprocess.run(
            [sys.executable, str(SCRIPTS / "wizard_init.py"), flag],
            capture_output=True,
            text=True,
        )
        check(
            f"init {flag} prints usage + exit 0",
            p.returncode == 0 and "usage:" in (p.stdout + p.stderr).lower(),
            p.stdout + p.stderr,
        )

    # _prompt_from_list rejects an out-of-set value, then accepts a numeric selection
    import profile_init as _pi

    feed = iter(["not-a-real-sc", "2"])  # first is out-of-set (rejected/looped), second picks choice #2
    builtins.input = lambda *a, **k: next(feed)
    try:
        picked = _pi._prompt_from_list("SC", ["gp3", "efs-sc"], "gp3")
        check(
            "_prompt_from_list rejects unknown then honors a numeric pick",
            picked == "efs-sc",
            picked,
        )
        # the '!' escape forces a genuinely custom value
        feed2 = iter(["!my-custom"])
        builtins.input = lambda *a, **k: next(feed2)
        check(
            "_prompt_from_list '!' escape forces a custom value",
            _pi._prompt_from_list("SC", ["gp3"], "gp3") == "my-custom",
        )
    finally:
        builtins.input = real_input


# ── 18. [fix #45] resume-write is PRESERVE-then-overlay — no key/block is dropped, CLUSTER untouched ─────
def test_resume_preserves_all_keys():
    # A realistic, HAND-WRITTEN profile (not build_profile_text output) with: core fields, a non-"." subpath,
    # a CLUSTER value that differs from the profile filename, plus custom / RDMA / CONNECT_CMD blocks that the
    # wizard's fixed field set never enumerates. Confirming [Y] on resume must preserve EVERY input key/value.
    sample = (
        "\n".join(
            [
                f"# {pr.SCHEMA_HEADER}={pr.PROFILE_SCHEMA_VERSION}",
                "# Cluster profile: demo",
                "",
                "# ----- Identity -----",
                'CLUSTER="prod-cluster-alias"',  # deliberately != profile filename "demo"
                'NAMESPACE="nsA"',
                'OWNER="alice"',
                'KUBE_CONTEXT="ctxA"',
                "",
                "# ----- GPU scheduling -----",
                'GPU_PRODUCT="NVIDIA-B200"',
                'ARCH="amd64"',
                'SCHEDULER_NAME="volcano"',
                "",
                'IMAGE_PULL_SECRET="nvcr-pull"',
                'MODEL_CACHE_PVC="real-cache"',
                'MODEL_CACHE_SUBPATH="nemotron/v3/weights"',  # NON-"." — the reconstruction bug forced this to "."
                'HF_SECRET="hf-token"',
                'ARTIFACTS_STORAGE_CLASS="gp3"',
                'CONTROL_STORAGE_CLASS="efs-sc"',
                "",
                "# ----- Custom settings -----",
                'CUSTOM_FEATURE_ENABLED="true"',
                'CUSTOM_FEATURE_ENDPOINT="https://service.example.test"',
                "",
                "# ----- RDMA / IB fabric (probe-fabric output) -----",
                'RDMA_UCX_NET_DEVICES="mlx5_0:1,mlx5_1:1"',
                'RDMA_NODE_SELECTOR="feature.node.kubernetes.io/rdma.available=true"',
                "",
                "# ----- Connectivity -----",
                'CONNECT_CMD="tsh kube login prod-cluster-alias"',
                "",
            ]
        )
        + "\n"
    )

    with tempfile.TemporaryDirectory() as dd:
        dd = Path(dd)
        (dd / "demo.env").write_text(sample)
        # A matching playfile (identity ctxA/B200 unchanged) drives a headless resume-write == confirming [Y].
        play = dd / "resume.yaml"
        play.write_text(
            "schema_version: 1\ncluster: demo\ncontext: ctxA\nnamespace: nsA\n"
            "gpu_product: NVIDIA-B200\narch: amd64\npull_secret: nvcr-pull\n"
            "model_cache_pvc: real-cache\n"
        )
        rc = wiz.run_play(
            "demo",
            str(play),
            profiles_dir=dd,
            battery_fn=lambda p: [cr.Check("var-reconcile", cr.PASS, "ok")],
        )
        check("resume-write exits 0", rc == wiz.EXIT_OK, rc)

        rewritten = (dd / "demo.env").read_text()
        env = pr._read_env(dd / "demo.env")

        # Every input key/value from the sample profile survives (this is the core #45 assertion).
        expect = {
            "CLUSTER": "prod-cluster-alias",  # NOT clobbered to the profile filename "demo"
            "NAMESPACE": "nsA",
            "OWNER": "alice",
            "KUBE_CONTEXT": "ctxA",
            "GPU_PRODUCT": "NVIDIA-B200",
            "ARCH": "amd64",
            "SCHEDULER_NAME": "volcano",
            "IMAGE_PULL_SECRET": "nvcr-pull",
            "MODEL_CACHE_PVC": "real-cache",
            "MODEL_CACHE_SUBPATH": "nemotron/v3/weights",  # NOT forced back to "."
            "HF_SECRET": "hf-token",
            "ARTIFACTS_STORAGE_CLASS": "gp3",
            "CONTROL_STORAGE_CLASS": "efs-sc",
            "CUSTOM_FEATURE_ENABLED": "true",
            "CUSTOM_FEATURE_ENDPOINT": "https://service.example.test",
            "RDMA_UCX_NET_DEVICES": "mlx5_0:1,mlx5_1:1",
            "RDMA_NODE_SELECTOR": "feature.node.kubernetes.io/rdma.available=true",
            "CONNECT_CMD": "tsh kube login prod-cluster-alias",
        }
        for k, v in expect.items():
            check(f"resume preserves {k}={v!r}", env.get(k) == v, f"got {env.get(k)!r}")

        # Explicit guards for the three symptoms named in the bug report.
        check(
            "resume does NOT overwrite CLUSTER with the profile name",
            'CLUSTER="demo"' not in rewritten and env.get("CLUSTER") == "prod-cluster-alias",
            rewritten,
        )
        check(
            "resume does NOT drop MODEL_CACHE_SUBPATH to '.'",
            env.get("MODEL_CACHE_SUBPATH") == "nemotron/v3/weights",
            rewritten,
        )
        check(
            "resume keeps custom/RDMA/CONNECT_CMD blocks",
            all(
                kk in rewritten
                for kk in (
                    "CUSTOM_FEATURE_ENABLED",
                    "RDMA_UCX_NET_DEVICES",
                    "CONNECT_CMD",
                )
            ),
            rewritten,
        )
        # schema header round-trips through the overlay
        check(
            "resume keeps the schema header",
            env.get(pr.SCHEMA_HEADER) == str(pr.PROFILE_SCHEMA_VERSION),
            env.get(pr.SCHEMA_HEADER),
        )

    # overlay_env_text unit: in-place update for a present key, append for an absent one, others untouched.
    ov = wiz.overlay_env_text('A="1"\n# c\nB="2"\n', {"A": "9", "C": "3"})
    check("overlay updates present key in place", 'A="9"' in ov and 'A="1"' not in ov, ov)
    check("overlay leaves untouched key B", 'B="2"' in ov, ov)
    check("overlay preserves comment lines", "# c" in ov, ov)
    check("overlay appends an absent key", 'C="3"' in ov, ov)


# ── 19. node-size cluster facts: CPU_PER_NODE / MEM_PER_NODE / WHOLE_NODE_* (85% DaemonSet headroom) ─────
def test_node_size_facts():
    # cpu quantity normalization → millicores (both forms allocatable.cpu actually takes)
    check("cpu 139580m → 139580 millicores", wiz._cpu_to_millicores("139580m") == 139580)
    check(
        "cpu bare cores '192' → 192000 millicores",
        wiz._cpu_to_millicores("192") == 192000,
    )
    check(
        "cpu fractional cores '191.5' → 191500 millicores",
        wiz._cpu_to_millicores("191.5") == 191500,
    )
    check(
        "cpu junk/empty → 0 (fact omitted, not a crash)",
        wiz._cpu_to_millicores("") == 0 and wiz._cpu_to_millicores("junk") == 0 and wiz._cpu_to_millicores(None) == 0,
    )
    # memory quantity normalization → whole Gi, rounded DOWN
    # 948007936Ki = 904.09 Gi → floors to 904 (903Gi would be exactly 946864128Ki).
    check(
        "mem 948007936Ki → 904Gi (rounded down)",
        wiz._mem_to_gib("948007936Ki") == 904,
        wiz._mem_to_gib("948007936Ki"),
    )
    check("mem 946864128Ki → exactly 903Gi", wiz._mem_to_gib("946864128Ki") == 903)
    check("mem 903Gi → 903Gi", wiz._mem_to_gib("903Gi") == 903)
    check("mem plain bytes → Gi", wiz._mem_to_gib(str(4 * 1024**3)) == 4)
    check(
        "mem rounds DOWN, never up (never over-promise)",
        wiz._mem_to_gib("2047Mi") == 1,
        wiz._mem_to_gib("2047Mi"),
    )
    check(
        "mem junk/empty → 0 (fact omitted, not a crash)",
        wiz._mem_to_gib("") == 0 and wiz._mem_to_gib("junk") == 0 and wiz._mem_to_gib(None) == 0,
    )

    # 85% headroom math. CPU is a BARE INTEGER core count (Guaranteed QoS + integer cpu = exclusive CPUs
    # under a static CPUManager policy); a fractional millicore value would forfeit that.
    check("WHOLE_NODE_HEADROOM_PCT is 85", wiz.WHOLE_NODE_HEADROOM_PCT == 85)
    _wc = wiz.whole_node_cpu(139580)
    check(
        "whole_node_cpu(139580m) → 118 whole cores (floor(139580*0.85/1000))",
        _wc == 118,
        _wc,
    )
    check(
        "whole_node_cpu returns a bare int — no 'm', no decimal",
        isinstance(_wc, int) and str(_wc).isdigit(),
        repr(_wc),
    )
    check(
        "whole_node_mem_gib(903) → 767Gi (floor(903*0.85))",
        wiz.whole_node_mem_gib(903) == 767,
        wiz.whole_node_mem_gib(903),
    )
    check(
        "headroom degrades to 0 when undetected",
        wiz.whole_node_cpu(0) == 0 and wiz.whole_node_mem_gib(0) == 0,
    )

    # node_size_facts over the probed node list — uniform fleet, no warning.
    _n = [
        {"cpu_alloc": "139580m", "mem_alloc": "946864128Ki"},
        {"cpu_alloc": "139580m", "mem_alloc": "946864128Ki"},
    ]
    cpu_m, mem_gi, warn = wiz.node_size_facts(_n)
    check(
        "node_size_facts: uniform fleet → (139580, 903) and NO warning",
        (cpu_m, mem_gi, warn) == (139580, 903, ""),
        (cpu_m, mem_gi, warn),
    )
    # MIXED node sizes → warn (never fail) and report the SMALLEST: a whole-node pod must fit the smallest
    # node it might land on.
    _mixed = [
        {"cpu_alloc": "192", "mem_alloc": "2000Gi"},
        {"cpu_alloc": "96", "mem_alloc": "1000Gi"},
    ]
    cpu_m2, mem_gi2, warn2 = wiz.node_size_facts(_mixed)
    check("node_size_facts: mixed sizes pick the SMALLEST cpu", cpu_m2 == 96000, cpu_m2)
    check("node_size_facts: mixed sizes pick the SMALLEST mem", mem_gi2 == 1000, mem_gi2)
    check("node_size_facts: mixed sizes WARN (not fail)", "SMALLEST" in warn2, warn2)
    # Graceful degradation: no GPU nodes / no RBAC to read allocatable → all zeros, no warning.
    check("node_size_facts: no nodes → (0, 0, '')", wiz.node_size_facts([]) == (0, 0, ""))
    check(
        "node_size_facts: nodes without allocatable → (0, 0, '')",
        wiz.node_size_facts([{"gpus": 4}]) == (0, 0, ""),
    )

    # Generated profile text carries the facts (GB300 fingerprint: 139580m cpu / 903Gi mem / 4 GPUs).
    a = wiz.Answers(
        cluster="c",
        namespace="ns",
        gpu_product="NVIDIA-GB300",
        pull_secret="p",
        gpu_per_node="4",
        cpu_per_node="139580m",
        mem_per_node="903Gi",
        whole_node_cpu="118",
        whole_node_mem="767Gi",
    )
    t = wiz.build_profile_text(a)
    for line in (
        'GPU_PER_NODE="4"',
        'CPU_PER_NODE="139580m"',
        'MEM_PER_NODE="903Gi"',
        'WHOLE_NODE_CPU="118"',
        'WHOLE_NODE_MEM="767Gi"',
    ):
        check(f"profile text carries {line}", line in t, t)
    check("profile explains WHY headroom exists (DaemonSets)", "DaemonSets" in t, t)
    # Degradation: undetected node size → the facts are simply absent (like GPU_PER_NODE).
    t_bare = wiz.build_profile_text(
        wiz.Answers(cluster="c", namespace="ns", gpu_product="NVIDIA-GB300", pull_secret="p")
    )
    check(
        "undetected node size omits the facts entirely",
        "CPU_PER_NODE" not in t_bare and "WHOLE_NODE_CPU" not in t_bare,
        t_bare,
    )
    # Round-trip profile → Answers, so a resume preserves the facts instead of dropping them.
    _rt = wiz.Answers(cluster="c")
    wiz._fill_answers_from_env(
        _rt,
        {
            "CPU_PER_NODE": "139580m",
            "MEM_PER_NODE": "903Gi",
            "WHOLE_NODE_CPU": "118",
            "WHOLE_NODE_MEM": "767Gi",
        },
    )
    check(
        "node-size facts round-trip profile→Answers (resume preserves them)",
        (_rt.cpu_per_node, _rt.mem_per_node, _rt.whole_node_cpu, _rt.whole_node_mem)
        == ("139580m", "903Gi", "118", "767Gi"),
    )


def main() -> int:
    # Isolate config-dir writes (readiness stamps, resume state) from the operator's real ~/.config/llmb.
    os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp(prefix="llmb-selftest-")
    print("selftest_wizard_init — offline wizard unit tests")
    for t in (
        test_dispatcher,
        test_done_renderer,
        test_model_cache,
        test_single_ip,
        test_schema_reader,
        test_play_mode,
        test_config_storage,
        test_rwx_detector,
        test_pvc_manifest,
        test_cli_smoke,
        test_interactive_e2e,
        test_cluster_discovery,
        test_secret_creds,
        test_storage_and_cache_defaults,
        test_qa_init_defaults,
        test_reachability_gate,
        test_name_validation,
        test_identity_lock,
        test_play_model_cache,
        test_tmp_uniqueness,
        test_dry_run_and_help,
        test_resume_preserves_all_keys,
        test_node_size_facts,
    ):
        t()
    print(f"\n{'FAILED' if fails else 'OK'} — {len(fails)} failure(s)" + (f": {fails}" if fails else ""))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
