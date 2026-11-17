"""Tests for the research pipelines of the Analysis tab (run headless)."""

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from amazeing.app import pipelines as pl   # noqa: E402  (no Qt needed)

ENV_ALL = {"R": True, "pymc": True, "tdt": True, "statsmodels": True, "plotly": True}


def _source(p):
    with open(pl.script_path(p), encoding="utf-8") as f:
        return f.read()


def _make_root(tmp_path):
    """A data root with an Experiment 1 day, a grammar day and a photometry tank folder."""
    root = tmp_path / "AuditoryExperiments"
    (root / "8_arms_w_voc" / "w1_d1").mkdir(parents=True)
    (root / "grammar" / "day_1").mkdir(parents=True)
    (root / "dopamine recordings" / "dopaminergic recordings").mkdir(parents=True)
    return root


# -- the registry ----------------------------------------------------------------
class TestRegistry:

    def test_scripts_found_from_source(self):
        assert pl.scripts_dir() is not None
        assert os.path.isdir(pl.scripts_dir())

    def test_every_script_exists(self):
        missing = [p.script for p in pl.PIPELINES if not os.path.isfile(pl.script_path(p))]
        assert missing == []

    def test_ids_unique_and_groups_known(self):
        ids = [p.id for p in pl.PIPELINES]
        assert len(ids) == len(set(ids))
        assert {p.group for p in pl.PIPELINES} <= set(pl.GROUPS)

    def test_dependencies_exist_and_no_cycles(self):
        for p in pl.PIPELINES:
            for d in p.after:
                assert d in pl.BY_ID, f"{p.id} depends on unknown {d}"
        assert pl.registry_problems() == []

    def test_needs_are_known(self):
        for p in pl.PIPELINES:
            for n in p.needs + tuple(x for o in p.options for x in o.needs):
                assert n in pl.NEED_LABELS

    def test_flags_match_each_scripts_argparse(self):
        for p in pl.PIPELINES:
            src = _source(p)
            if p.shuffled:
                assert "--shuffled-only" in src, p.id
            if p.thesis:
                assert "--thesis-figures" in src, p.id
            for o in p.options:
                for a in o.args:
                    if a.startswith("--"):
                        assert a in src, f"{p.id}: {a}"
            for t in p.args:
                if t.startswith("--"):
                    assert t in src, f"{p.id}: {t}"

    def test_the_three_shuffled_scripts(self):
        assert {p.id for p in pl.PIPELINES if p.shuffled} == {
            "exp1_stimulus_pi", "exp1_complexity", "exp1_lmm"}

    def test_scripts_calling_r_need_it(self):
        for p in pl.PIPELINES:
            src = _source(p)
            if "rscript()" in src and p.id != "pilot_four_arm":
                assert "R" in p.needs, p.id

    def test_r_models_present(self):
        base = pl.scripts_dir()
        for name in ("exp1_lmm", "exp1_approach", "exp1_individuals", "exp2_lmm",
                     "fisher_z_check", "pilot_lmm"):
            assert os.path.isfile(os.path.join(base, name + ".R")), name

    def test_legacy_pipeline_left_out(self):
        scripts = {p.script for p in pl.PIPELINES}
        for legacy in ("01_preference_analysis.py", "02_within_trial_preference.py",
                       "03_completers_lmm.py", "04_random_slopes_presentation.py",
                       "run_batch_preference.py", "aesthetic_value_model_4D.py"):
            assert legacy not in scripts
        assert not any(s.endswith(".ipynb") for s in scripts)

    def test_bundle_files_cover_every_script(self):
        base = pl.scripts_dir()
        files = {os.path.normpath(f) for f in pl.bundle_files(base)}
        for p in pl.PIPELINES:
            assert pl.script_path(p, base) in files
        assert any(f.endswith(".R") for f in files)
        assert not any(os.sep + "tests" + os.sep in f for f in files)


# -- the launcher -----------------------------------------------------------------
class TestScriptLauncher:

    def test_command_from_source(self, monkeypatch):
        from amazeing.app import launcher
        monkeypatch.delattr(sys, "frozen", raising=False)
        path = os.path.join("some", "dir", "exp1_lmm.py")
        cmd = launcher.command_for_script(path, ["data", "--shuffled-only"])
        assert cmd == [sys.executable, "-u", os.path.abspath(path), "data", "--shuffled-only"]
        assert launcher.script_cwd(path) == os.path.abspath(os.path.join("some", "dir"))

    def test_command_when_frozen(self, monkeypatch):
        from amazeing.app import launcher
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        path = os.path.abspath("x.py")
        cmd = launcher.command_for_script(path, ["d"])
        assert cmd == [sys.executable, "--script", path, "d"]

    def test_scripts_dir_when_frozen(self, monkeypatch, tmp_path):
        (tmp_path / "analysis" / "auditory").mkdir(parents=True)
        monkeypatch.delenv("AMAZEING_ANALYSIS_DIR", raising=False)
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
        assert pl.scripts_dir() == str(tmp_path / "analysis" / "auditory")

    def test_run_script_imports_siblings(self, monkeypatch, tmp_path, capsys):
        from amazeing.app import launcher
        (tmp_path / "sibling.py").write_text("VALUE = 41\n")
        (tmp_path / "main_script.py").write_text(
            "import sys\nimport sibling\n"
            "if __name__ == '__main__':\n"
            "    print('got', sibling.VALUE + 1, sys.argv[1:])\n")
        monkeypatch.setattr(sys, "argv", list(sys.argv))
        monkeypatch.setattr(sys, "path", list(sys.path))
        monkeypatch.delitem(sys.modules, "sibling", raising=False)
        assert launcher.run_script(str(tmp_path / "main_script.py"), ["a", "--b"]) == 0
        assert "got 42 ['a', '--b']" in capsys.readouterr().out

    def test_main_dispatches_script(self, monkeypatch, tmp_path):
        from amazeing import app
        from amazeing.app import launcher
        seen = []
        monkeypatch.setattr(launcher, "run_script", lambda p, a: seen.append((p, a)) or 0)
        assert app.main(["--script", "s.py", "x"]) == 0
        assert seen == [("s.py", ["x"])]
        assert app.main(["--script"]) == 2

    def test_missing_script(self, tmp_path):
        from amazeing.app import launcher
        assert launcher.run_script(str(tmp_path / "nope.py"), []) == 2


# -- layout, arguments, requirements --------------------------------------------------
class TestLayout:

    def test_root(self, tmp_path):
        root = _make_root(tmp_path)
        lay = pl.resolve_layout(str(root))
        assert lay.kind == "root"
        assert lay.folders["exp1"] == str(root / "8_arms_w_voc")
        assert lay.folders["grammar"] == str(root / "grammar")
        assert lay.folders["photometry"] == str(root / "dopamine recordings")
        assert lay.folders["survey"] == str(tmp_path / "AuditoryExperiments_survey")

    def test_experiment_folder_finds_siblings(self, tmp_path):
        root = _make_root(tmp_path)
        lay = pl.resolve_layout(str(root / "8_arms_w_voc"))
        assert lay.kind == "exp1"
        assert lay.folders["grammar"] == str(root / "grammar")
        assert pl.resolve_layout(str(root / "grammar")).kind == "grammar"
        assert pl.resolve_layout(str(root / "dopamine recordings")).kind == "photometry"
        batch = root / "8_arms_w_voc" / "BATCH_ANALYSIS"
        batch.mkdir()
        assert pl.resolve_layout(str(batch)).kind == "exp1"

    def test_lone_experiment_folder(self, tmp_path):
        exp = tmp_path / "my_copy"
        (exp / "w2_sequences").mkdir(parents=True)
        lay = pl.resolve_layout(str(exp))
        assert lay.kind == "exp1"
        assert set(lay.folders) == {"exp1"}

    def test_unknown_and_empty(self, tmp_path):
        assert pl.resolve_layout(str(tmp_path)).kind == "unknown"
        assert pl.resolve_layout("").kind == "unknown"
        assert pl.resolve_layout(str(tmp_path / "absent")).folders == {}

    def test_build_args(self, tmp_path):
        f = {"exp1": str(tmp_path / "e1"), "grammar": str(tmp_path / "g")}
        p = pl.BY_ID["exp1_complexity"]
        assert pl.build_args(p, f) == [f["exp1"]]
        assert pl.build_args(p, f, shuffled=True, thesis_dir="T") == [
            f["exp1"], "--shuffled-only", "--thesis-figures", "T"]
        # shuffled is ignored by scripts that do not take it
        assert pl.build_args(pl.BY_ID["exp1_bd_model"], f, shuffled=True) == [f["exp1"]]
        mv = pl.build_args(pl.BY_ID["model_validation"], f, options=["phase2"])
        assert mv[:4] == ["--results_dir", f["grammar"], "--out_dir",
                          os.path.join(f["grammar"], "model_validation_output")]
        assert mv[-1] == "--phase2"
        with pytest.raises(KeyError):
            pl.build_args(pl.BY_ID["exp2_figures"], f)          # thesis folder required
        with pytest.raises(KeyError):
            pl.build_args(pl.BY_ID["fibpho_arms"], f)           # no photometry folder

    def test_outputs_and_done(self, tmp_path):
        root = _make_root(tmp_path)
        f = pl.resolve_layout(str(root)).folders
        p = pl.BY_ID["exp1_stimulus_pi"]
        out = os.path.join(f["exp1"], "BATCH_ANALYSIS", "stimulus_pi")
        assert pl.output_dir(p, f) == out
        assert pl.output_dir(p, f, shuffled=True) == out + "_shuffled"
        assert not pl.is_done(p, f)
        os.makedirs(out)
        assert pl.is_done(p, f) and not pl.is_done(p, f, shuffled=True)
        assert pl.output_dir(pl.BY_ID["fibpho_arms"], f) == os.path.join(
            f["photometry"], "FIBPHO_ANALYSIS", "arms")
        assert not pl.is_done(pl.BY_ID["stimulus_complexity"], f)

    def test_missing_requirements(self, tmp_path):
        f = {"exp1": str(tmp_path)}
        no_r = dict(ENV_ALL, R=False)
        reasons = pl.missing(pl.BY_ID["exp1_lmm"], f, no_r)
        assert len(reasons) == 1 and reasons[0].startswith("needs R")
        assert pl.missing(pl.BY_ID["exp1_lmm"], f, ENV_ALL) == []
        assert "needs the Experiment 2 folder (grammar)" in pl.missing(
            pl.BY_ID["exp2_lmm"], f, ENV_ALL)[0]
        # PyMC only matters when the Bayesian option is on
        g = {"grammar": str(tmp_path)}
        no_pymc = dict(ENV_ALL, pymc=False)
        assert pl.missing(pl.BY_ID["model_validation"], g, no_pymc) == []
        assert "PyMC" in pl.missing(pl.BY_ID["model_validation"], g, no_pymc, ["phase2"])[0]
        assert "PyMC" in pl.missing(pl.BY_ID["grain_comparison"], g, no_pymc)[0]

    def test_install_hints_name_a_source_install_when_frozen(self):
        assert "source install" in pl.install_hint("pymc", frozen=True)
        # tdt is bundled in the packaged app, so it is never sent to a source install
        assert "source install" not in pl.install_hint("tdt", frozen=True)
        assert pl.install_hint("pymc", frozen=False).startswith("pip install")
        assert "lme4" in pl.install_hint("R")


# -- run plans --------------------------------------------------------------------------
class TestPlan:

    @staticmethod
    def ids(steps):
        return [(s.pipeline.id, s.shuffled) for s in steps]

    def test_dependencies_first_in_order(self):
        steps = pl.plan(["exp1_bd_model"])
        assert self.ids(steps) == [("exp1_stimulus_pi", False), ("exp1_complexity", False),
                                   ("exp1_lmm", False), ("exp1_bd_model", False)]
        assert steps[0].reason == "needed by exp1_bd_model"
        assert steps[-1].reason == "selected"

    def test_every_plan_is_topological(self):
        for p in pl.PIPELINES:
            seen = set()
            for s in pl.plan([p.id]):
                assert set(s.pipeline.after) <= seen, s.pipeline.id
                seen.add(s.pipeline.id)

    def test_done_dependencies_skipped(self):
        steps = pl.plan(["exp1_bd_model"], done=lambda p, s: p.id != "exp1_lmm")
        assert self.ids(steps) == [("exp1_lmm", False), ("exp1_bd_model", False)]
        steps = pl.plan(["exp1_bd_model"], done=lambda p, s: True, rerun_dependencies=True)
        assert len(steps) == 4

    def test_transitive_dependency_through_a_done_one_is_not_added(self):
        # exp1_complexity done, so its own dependency is not pulled in through it
        steps = pl.plan(["exp1_bd_model"],
                        done=lambda p, s: p.id in ("exp1_complexity", "exp1_stimulus_pi"))
        assert self.ids(steps) == [("exp1_lmm", False), ("exp1_bd_model", False)]

    def test_shuffled_variants(self):
        steps = pl.plan(["exp1_complexity", "exp1_bd_model"], shuffled=True)
        ids = self.ids(steps)
        assert ("exp1_complexity", True) in ids and ("exp1_stimulus_pi", True) in ids
        # exp1_bd_model reads the full-block outputs
        assert ("exp1_complexity", False) in ids and ("exp1_stimulus_pi", False) in ids
        assert ("exp1_bd_model", False) in ids
        assert ids.index(("exp1_stimulus_pi", True)) < ids.index(("exp1_complexity", True))

    def test_selected_dependency_counts_as_selected(self):
        steps = pl.plan(["exp1_complexity", "exp1_stimulus_pi"])
        assert [s.reason for s in steps] == ["selected", "selected"]

    def test_cycle_detected(self, monkeypatch):
        a = pl.Pipeline("a", "a.py", "A", "", "Experiment 1", (), None, after=("b",))
        b = pl.Pipeline("b", "b.py", "B", "", "Experiment 1", (), None, after=("a",))
        monkeypatch.setattr(pl, "BY_ID", {"a": a, "b": b})
        monkeypatch.setattr(pl, "ORDER", {"a": 0, "b": 1})
        with pytest.raises(ValueError, match="cycle"):
            pl.plan(["a"])


# -- the queue and the tab (Qt) -------------------------------------------------------------
def _wait_for(signal, timeout_ms=30000):
    from PySide6.QtCore import QEventLoop, QTimer
    loop = QEventLoop()
    got = []
    signal.connect(lambda *a: (got.append(a), loop.quit()))
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()
    return got


class TestQueue:

    def _py(self, code):
        return [sys.executable, "-c", code]

    def test_runs_all_in_order(self, qapp):
        pytest.importorskip("PySide6")
        from amazeing.app.process_panel import CommandQueue, ProcessPanel
        panel = ProcessPanel(show_stdin=False)
        q = CommandQueue(panel)
        codes = []
        q.step_finished.connect(lambda i, c: codes.append((i, c)))
        q.run([("one", self._py("print('first')"), None),
               ("two", self._py("print('second')"), None)])
        assert _wait_for(q.all_finished) == [(True,)]
        assert codes == [(0, 0), (1, 0)]
        text = panel.console.toPlainText()
        assert text.index("first") < text.index("second")
        panel.deleteLater()

    def test_stops_at_first_failure(self, qapp):
        pytest.importorskip("PySide6")
        from amazeing.app.process_panel import CommandQueue, ProcessPanel
        panel = ProcessPanel(show_stdin=False)
        q = CommandQueue(panel)
        codes = []
        q.step_finished.connect(lambda i, c: codes.append((i, c)))
        q.run([("ok", self._py("pass"), None),
               ("fails", self._py("import sys; sys.exit(3)"), None),
               ("never", self._py("print('should not run')"), None)])
        assert _wait_for(q.all_finished) == [(False,)]
        assert codes == [(0, 0), (1, 3)]
        text = panel.console.toPlainText()
        assert "=== 3 of 3" not in text and "should not run" not in text.split("$")[-1]
        assert "1 later step(s) not run" in text
        assert not q.is_running()
        panel.deleteLater()

    def test_program_that_cannot_start_stops_the_run(self, qapp, tmp_path):
        pytest.importorskip("PySide6")
        from amazeing.app.process_panel import CommandQueue, ProcessPanel
        panel = ProcessPanel(show_stdin=False)
        q = CommandQueue(panel)
        ended = []
        q.all_finished.connect(ended.append)     # the failure is reported inside run()
        q.run([("missing", [str(tmp_path / "no_such_program.exe")], None),
               ("never", self._py("pass"), None)])
        if not ended:
            _wait_for(q.all_finished, 10000)
        assert ended == [False]
        assert not q.is_running() and panel.status.text() == "Error"
        assert "=== 2 of 2" not in panel.console.toPlainText()
        panel.deleteLater()


class TestResearchTab:

    def test_analysis_tab_builds(self, qapp):
        pytest.importorskip("PySide6")
        from amazeing.app.analysis_tab import AnalysisTab
        tab = AnalysisTab()
        assert tab.pages.count() == 2
        assert tab.pages.tabText(1) == "Research figures"
        assert set(tab.research.items) == set(pl.BY_ID)
        tab.deleteLater()

    def test_status_and_unavailable(self, qapp, tmp_path):
        pytest.importorskip("PySide6")
        from PySide6.QtCore import Qt
        from amazeing.app.research_panel import ResearchFigures
        root = _make_root(tmp_path)
        os.makedirs(root / "8_arms_w_voc" / "BATCH_ANALYSIS" / "stimulus_pi")
        w = ResearchFigures(env=dict(ENV_ALL, R=False, pymc=False))
        w.folder.setText(str(root))
        assert w.layout_info.kind == "root"
        assert w.items["exp1_stimulus_pi"].text(1) == "done"
        assert w.items["exp1_complexity"].text(1) == "not run"
        lmm = w.items["exp1_lmm"]
        assert lmm.text(1) == "cannot run"
        assert not (lmm.flags() & Qt.ItemIsUserCheckable)
        assert "lme4" in lmm.toolTip(0)
        # ticking the group ticks only what can run
        group = w._groups["Experiment 1"]
        group.setCheckState(0, Qt.Checked)
        sel = w.selected()
        assert "exp1_stimulus_pi" in sel and "exp1_lmm" not in sel
        assert group.checkState(0) == Qt.Checked
        # the shuffled variant has no output yet
        w.shuffled.setChecked(True)
        assert w.items["exp1_stimulus_pi"].text(1) == "not run"
        w.deleteLater()

    def test_build_run_expands_and_refuses(self, qapp, tmp_path):
        pytest.importorskip("PySide6")
        from amazeing.app.research_panel import ResearchFigures
        root = _make_root(tmp_path)
        w = ResearchFigures(env=ENV_ALL)
        w.folder.setText(str(root / "8_arms_w_voc"))
        w.set_selected(["exp1_complexity"])
        items, problems = w.build_run()
        assert problems == []
        assert [label for label, _c, _w in items] == [
            "exp1_stimulus_pi, needed by exp1_complexity", "exp1_complexity"]
        cmd, cwd = items[1][1], items[1][2]
        assert cmd[-1] == str(root / "8_arms_w_voc")
        assert cwd == pl.scripts_dir()
        # a dependency that cannot run blocks the run
        w.env["R"] = False
        w.set_selected(["exp1_bd_model"])
        items, problems = w.build_run()
        assert any(p.startswith("exp1_lmm") for p in problems)
        # a thesis-only script asks for the thesis folder
        w.env["R"] = True
        w.folder.setText(str(root))
        w.set_selected(["exp2_figures"])
        _items, problems = w.build_run()
        assert any("thesis" in p for p in problems)
        w.deleteLater()

    def test_runs_a_script_end_to_end(self, qapp, tmp_path, monkeypatch):
        """bayes_principle_figure reads no data, so it runs on an empty grammar folder."""
        pytest.importorskip("PySide6")
        pytest.importorskip("matplotlib")
        from amazeing.app.research_panel import ResearchFigures
        grammar = tmp_path / "grammar"
        (grammar / "day_1").mkdir(parents=True)
        w = ResearchFigures(env=ENV_ALL)
        w.folder.setText(str(grammar))
        w.set_selected(["bayes_principle_figure"])
        items, problems = w.build_run()
        assert problems == []
        w.queue.run(items)
        assert _wait_for(w.queue.all_finished, 120000) == [(True,)], \
            w.panel.console.toPlainText()
        out = grammar / "BATCH_ANALYSIS" / "bayes_principle"
        assert any(f.suffix == ".png" for f in out.iterdir())
        w.refresh()
        assert w.items["bayes_principle_figure"].text(1) == "done"
        w.tree.setCurrentItem(w.items["bayes_principle_figure"])
        assert w.fig_list.count() > 0
        w.deleteLater()
