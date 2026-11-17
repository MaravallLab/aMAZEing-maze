"""The thesis research pipelines in ``analysis/auditory/``, as the app sees them.

The scripts are standalone research code, cited by path in the thesis, so
they are never moved or edited from here. This module only records, for each
one, what it answers, which folder it reads, which scripts must run before it,
what it needs beyond Python and where it writes; and it turns a selection into
an ordered run plan and a command line per script. Nothing here imports Qt, so
the packaging spec and the tests can use it as well.

The source of truth is ``docs/analysis/auditory.md`` and each script's
argparse. The earlier batch pipeline (``01_*`` to ``04_*``,
``run_batch_preference``, ``aesthetic_value_model_4D``) and the notebooks are
left out on purpose: the thesis does not use them.
"""

from __future__ import annotations

import glob
import importlib.util
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Tuple

# -- folders ------------------------------------------------------------------
# A layout maps these keys to absolute folders. Script arguments and outputs
# are templates over them, so the folders a script needs are the keys its
# argument templates name.
FOLDER_NAMES = {
    "root": "the data root",
    "exp1": "the Experiment 1 folder (8_arms_w_voc)",
    "grammar": "the Experiment 2 folder (grammar)",
    "photometry": "the photometry folder (dopamine recordings)",
    "survey": "the pilot survey folder (<data root>_survey)",
    "thesis": "a thesis figures folder",
}

EXP1_NAME = "8_arms_w_voc"
GRAMMAR_NAME = "grammar"
PHOTOMETRY_NAME = "dopamine recordings"
TANK_FOLDER = "dopaminergic recordings"     # fibpho_alignment.TANK_FOLDER

GROUPS = ("Experiment 1", "Experiment 2", "Four-arm pilot", "Photometry", "Stimulus measures")

# -- requirements ---------------------------------------------------------------
# Each need maps to a check in detect_environment().
NEED_LABELS = {
    "R": "R",
    "pymc": "PyMC",
    "tdt": "the tdt package",
    "statsmodels": "statsmodels",
    "plotly": "plotly",
}

PYMC_PIP = 'pip install "pymc>=5.15,<5.20" "arviz>=0.18,<0.20"'


@dataclass(frozen=True)
class Option:
    """A switch specific to one script, shown when that script is selected."""
    id: str
    label: str
    args: Tuple[str, ...]
    needs: Tuple[str, ...] = ()
    tip: str = ""


@dataclass(frozen=True)
class Pipeline:
    id: str
    script: str                     # path under analysis/auditory, forward slashes
    title: str
    question: str
    group: str
    args: Tuple[str, ...]           # argument templates over the layout keys
    output: Optional[str]           # folder its results go to (template), None if thesis only
    after: Tuple[str, ...] = ()     # ids that must have run first
    needs: Tuple[str, ...] = ()
    marker: Optional[str] = None    # file whose presence means "done" (default: output)
    shuffled: bool = False          # takes --shuffled-only (output gains _shuffled)
    thesis: bool = False            # takes --thesis-figures DIR
    thesis_required: bool = False
    options: Tuple[Option, ...] = ()
    note: str = ""

    @property
    def folders(self) -> Tuple[str, ...]:
        """Layout keys this script cannot run without, in first-use order."""
        keys: List[str] = []
        for t in self.args:
            for k in _placeholders(t):
                if k not in keys:
                    keys.append(k)
        return tuple(keys)


def _placeholders(template: str) -> List[str]:
    return re.findall(r"{(\w+)}", template)


E1 = "{exp1}/BATCH_ANALYSIS/"
E2 = "{grammar}/BATCH_ANALYSIS/"
FP = "{photometry}/FIBPHO_ANALYSIS"
MV = "{grammar}/model_validation_output"

PIPELINES: Tuple[Pipeline, ...] = (
    # ---------------------------------------------------------------- Experiment 1
    Pipeline(
        "exp1_vocalisation_pi", "exp1_vocalisation_pi.py", "Vocalisation preference",
        "Do mice prefer the vocalisation to silence and to the synthetic sounds? Per day, "
        "capped and uncapped.",
        "Experiment 1", ("{exp1}",), E1 + "vocalisation_pi", thesis=True),
    Pipeline(
        "exp1_stimulus_pi", "exp1_stimulus_pi.py", "Stimulus preference indices",
        "Each stimulus against the silent arm; consonant against dissonant; rough against "
        "smooth envelopes.",
        "Experiment 1", ("{exp1}",), E1 + "stimulus_pi", needs=("statsmodels",),
        shuffled=True, thesis=True),
    Pipeline(
        "exp1_complexity", "exp1_complexity.py", "Preference and complexity",
        "Does preference follow spectral or temporal complexity, in a line or an inverted U?",
        "Experiment 1", ("{exp1}",), E1 + "complexity", after=("exp1_stimulus_pi",),
        needs=("statsmodels",), shuffled=True, thesis=True),
    Pipeline(
        "exp1_lmm", "exp1_lmm.py", "Mixed models of visit length",
        "Class by set, consonance, roughness and modulation; writes the arm by block table "
        "later scripts read.",
        "Experiment 1", ("{exp1}",), E1 + "lmm", needs=("R",), shuffled=True),
    Pipeline(
        "exp1_order_place", "exp1_order_place.py", "Day order and place",
        "Day order against habituation, and place bias: indices block by block, arm shares, "
        "the usual silent arm, first entries.",
        "Experiment 1", ("{exp1}",), E1 + "order_place", needs=("statsmodels",), thesis=True),
    Pipeline(
        "exp1_approach", "exp1_approach.py", "Entering against staying",
        "Entries per arm and mean visit length, with a Poisson mixed model of the entries.",
        "Experiment 1", ("{exp1}",), E1 + "approach",
        after=("exp1_lmm", "exp1_order_place"), needs=("R",), thesis=True),
    Pipeline(
        "exp1_exploration", "exp1_exploration.py", "Exploration and engagement",
        "Does exploratory drive in the silent habituation block go with engagement with the "
        "sounds?",
        "Experiment 1", ("{exp1}",), E1 + "exploration", needs=("statsmodels",), thesis=True),
    Pipeline(
        "exp1_individuals", "exp1_individuals.py", "Individual mice",
        "How the averages divide among the mice: indices per mouse, variance components, "
        "principal components.",
        "Experiment 1", ("{exp1}",), E1 + "individuals",
        after=("exp1_stimulus_pi", "exp1_lmm"), needs=("R",), thesis=True),
    Pipeline(
        "exp1_consistency_checks", "exp1_consistency_checks.py", "Consonance and place checks",
        "Is each mouse's consonance index consistent across the interval sets, and does "
        "place explain the sound against silence contrast?",
        "Experiment 1", ("{exp1}",), E1 + "consistency", after=("exp1_stimulus_pi",),
        thesis=True),
    Pipeline(
        "exp1_bd_model", "exp1_bd_model.py", "Brielmann and Dayan model",
        "Does the model, with each stimulus described by its measured complexity, account "
        "for every index on every day?",
        "Experiment 1", ("{exp1}",), E1 + "bd_model",
        after=("exp1_stimulus_pi", "exp1_complexity", "exp1_lmm"), thesis=True),
    Pipeline(
        "pi_fisher_figure", "pi_fisher_figure.py", "Preference index and Fisher z",
        "The figure of Section 2.6: the index against its Fisher z, with the range of the "
        "mice of Experiment 1.",
        "Experiment 1", ("{exp1}",), E1 + "pi_fisher",
        after=("exp1_stimulus_pi", "exp1_lmm"), thesis=True),

    # ---------------------------------------------------------------- Experiment 2
    Pipeline(
        "exp2_lmm", "exp2_lmm.py", "Session-level preference",
        "The preference for the grammar of the enriched cage over the standard cage, block "
        "by block and per mouse, with place controlled.",
        "Experiment 2", ("{grammar}",), E2 + "exp2", needs=("R",), thesis=True),
    Pipeline(
        "model_validation", "model_validation/run_validation.py", "Value model validation",
        "Cohort, sanity checks, model-free design analysis, latent regressors and parameter "
        "recovery (Phase 1); the Bayesian fits on request.",
        "Experiment 2",
        ("--results_dir", "{grammar}", "--out_dir", MV, "--day", "both", "--figures"),
        MV, marker=MV + "/results.json",
        options=(Option("phase2", "Bayesian fits (Phase 2, PyMC, slow)", ("--phase2",),
                        needs=("pymc",),
                        tip="Without this, results.json has no Phase 2 section, which "
                            "the thesis figure of the model reads."),),
        note="Writes to model_validation_output beside the grammar data."),
    Pipeline(
        "grain_comparison", "model_validation/grain_comparison.py", "Grains of observation",
        "The Bayesian fit at three grains of observation.",
        "Experiment 2", ("{grammar}", MV), MV, marker=MV + "/grain_results.json",
        needs=("pymc",)),
    Pipeline(
        "exp2_model_check", "exp2_model_check.py", "Place-only model and ladder",
        "Could place alone produce the semantic weight? And the ladder of models, "
        "cross-validated.",
        "Experiment 2", ("{grammar}",), E2 + "exp2_model",
        marker=E2 + "exp2_model/ladder.csv"),
    Pipeline(
        "exp2_model_compare", "exp2_model_compare.py", "Information criteria",
        "AIC and BIC for the four nested value models, at the grain of the session.",
        "Experiment 2", ("{grammar}",), E2 + "exp2_model",
        marker=E2 + "exp2_model/information_criteria.csv"),
    Pipeline(
        "exp2_model_explainer", "exp2_model_explainer.py", "How the model is fitted",
        "An explanatory figure of the fit, step by step.",
        "Experiment 2", ("{grammar}",), E2 + "exp2_model",
        marker=E2 + "exp2_model/fig_model_explainer.png",
        after=("model_validation", "exp2_model_compare"), thesis=True),
    Pipeline(
        "exp2_power", "exp2_power.py", "Sample size and power",
        "The smallest effect 32 mice detect, and the mice a replication needs.",
        "Experiment 2", ("{grammar}",), E2 + "exp2", marker=E2 + "exp2/power.txt",
        after=("exp2_lmm",), needs=("statsmodels",)),
    Pipeline(
        "exp2_figures", "exp2_figures.py", "Thesis figures of Experiment 2",
        "The stimuli, the session-level preference and the model, as the thesis prints them.",
        "Experiment 2", ("{grammar}",), None,
        after=("exp2_lmm", "model_validation", "grain_comparison"), needs=("statsmodels",),
        thesis=True, thesis_required=True,
        note="Writes thesis figures only, so it needs the thesis figures folder."),
    Pipeline(
        "bayes_principle_figure", "bayes_principle_figure.py", "Bayesian fitting, the principle",
        "The figure of Section 2.7.2; schematic, it reads no data.",
        "Experiment 2", (E2 + "bayes_principle",), E2 + "bayes_principle", thesis=True),

    # ---------------------------------------------------------------- pilot
    Pipeline(
        "survey_standalone_sessions", "survey_standalone_sessions.py", "Survey of pilot sessions",
        "Reads every loose pilot session at the top of the data root on its own terms.",
        "Four-arm pilot", ("--root", "{root}", "--out", "{survey}"), "{survey}",
        marker="{survey}/blocks.csv"),
    Pipeline(
        "pilot_four_arm", "pilot_four_arm.py", "Four-arm pilot",
        "The three pilot cohorts, every mouse shown, tested as Experiment 1 is.",
        "Four-arm pilot", ("--survey", "{survey}"), "{survey}/pilot",
        after=("survey_standalone_sessions",), needs=("statsmodels",), thesis=True,
        note="Without R its mixed models of entries and visit length are skipped."),

    # ---------------------------------------------------------------- Experiment 1, after the pilot
    Pipeline(
        "fisher_z_check", "fisher_z_check.py", "Fisher z check",
        "Does analysing the index on its Fisher z scale change any result?",
        "Experiment 1", ("{exp1}", "--survey", "{survey}"), E1 + "fisher_z_check",
        after=("exp1_stimulus_pi", "exp1_complexity", "exp1_order_place", "exp1_exploration",
               "exp1_individuals", "pilot_four_arm"),
        needs=("R", "statsmodels"),
        options=(Option("skip_model", "Leave out the slow refit of the model", ("--skip-model",),
                        tip="Skips the refit of the model of Brielmann and Dayan."),),
        note="Also reads the pilot tables, so it needs the pilot survey folder."),

    # ---------------------------------------------------------------- photometry
    Pipeline(
        "fibpho_alignment", "fibpho_alignment.py", "Align photometry with sessions",
        "Aligns every recording with its session; delta F/F; overview, block and peri-event "
        "figures.",
        "Photometry", ("{photometry}",), FP, marker=FP + "/alignment_summary.csv",
        needs=("tdt",)),
    Pipeline(
        "fibpho_checks", "fibpho_checks.py", "Photometry checks",
        "Entries against random moments; each channel on its own; is 405 nm a clean motion "
        "reference?",
        "Photometry", ("{photometry}",), FP + "/checks", after=("fibpho_alignment",),
        needs=("tdt",)),
    Pipeline(
        "fibpho_arms", "fibpho_arms.py", "Dopamine by arm",
        "Dopamine while the mouse is in each arm, against the behavioural preference for it.",
        "Photometry", (FP,), FP + "/arms", after=("fibpho_alignment",)),
    Pipeline(
        "fibpho_dynamics", "fibpho_dynamics.py", "Dopamine dynamics",
        "How the response changes within blocks, across blocks and from arm to arm.",
        "Photometry", (FP,), FP + "/dynamics", after=("fibpho_alignment",)),
    Pipeline(
        "fibpho_behaviour", "fibpho_behaviour.py", "Dopamine and behaviour",
        "Dopamine by where the mouse was; the response to each stimulus from first to last "
        "exposure.",
        "Photometry", (FP,), FP + "/behaviour", after=("fibpho_alignment",), thesis=True),
    Pipeline(
        "fibpho_dwell", "fibpho_dwell.py", "Dopamine and dwell",
        "Dopamine before, during and after each visit, against how long the mouse stayed.",
        "Photometry", (FP,), FP + "/dwell", after=("fibpho_alignment",),
        needs=("statsmodels",), thesis=True),

    # ---------------------------------------------------------------- stimulus measures
    Pipeline(
        "stimulus_complexity", "stimulus_complexity.py", "Complexity measures",
        "Prints a worked table of the spectral and temporal complexity measures.",
        "Stimulus measures", (), None),
    Pipeline(
        "complexity_3d", "complexity_3d.py", "Complexity in three dimensions",
        "Figures 3.4 and 4.12 as pages you can turn with the mouse, and the numbers the "
        "thesis quotes from them.",
        "Stimulus measures", ("--survey", "{survey}", "--exp1", "{exp1}"),
        E1 + "complexity", marker=E1 + "complexity/exp1_complexity_3d.html",
        after=("exp1_stimulus_pi", "exp1_complexity", "exp1_lmm", "pilot_four_arm"),
        needs=("plotly",), thesis=True,
        note="The thesis still images need kaleido and an installed Chrome."),
)

BY_ID: Dict[str, Pipeline] = {p.id: p for p in PIPELINES}
ORDER: Dict[str, int] = {p.id: i for i, p in enumerate(PIPELINES)}


def in_group(group: str) -> List[Pipeline]:
    return [p for p in PIPELINES if p.group == group]


# -- locating the scripts ---------------------------------------------------------
def scripts_dir() -> Optional[str]:
    """The ``analysis/auditory`` folder, or None when it cannot be found.

    Frozen, the spec bundles it as data inside the application folder. From
    source it sits at the top of the repository. ``AMAZEING_ANALYSIS_DIR``
    overrides both.
    """
    env = os.environ.get("AMAZEING_ANALYSIS_DIR")
    if env:
        return env if os.path.isdir(env) else None
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
        cand = os.path.join(base, "analysis", "auditory")
    else:
        here = os.path.dirname(os.path.abspath(__file__))           # src/amazeing/app
        cand = os.path.normpath(os.path.join(here, "..", "..", "..", "analysis", "auditory"))
    return cand if os.path.isdir(cand) else None


def script_path(p: Pipeline, base: Optional[str] = None) -> Optional[str]:
    base = base or scripts_dir()
    if base is None:
        return None
    return os.path.normpath(os.path.join(base, *p.script.split("/")))


# -- the environment -------------------------------------------------------------
def find_rscript() -> Optional[str]:
    """Rscript as exp1_lmm.rscript() finds it: the PATH, then the newest R in Program Files."""
    found = shutil.which("Rscript")
    if found:
        return found
    candidates = sorted(glob.glob(r"C:\Program Files\R\R-*\bin\Rscript.exe"))
    return candidates[-1] if candidates else None


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def detect_environment() -> Dict[str, bool]:
    return {
        "R": find_rscript() is not None,
        "pymc": _has_module("pymc") and _has_module("arviz"),
        "tdt": _has_module("tdt"),
        "statsmodels": _has_module("statsmodels"),
        "plotly": _has_module("plotly"),
    }


def install_hint(need: str, frozen: Optional[bool] = None) -> str:
    """What to install for ``need``, worded for the packaged app or a source install."""
    frozen = getattr(sys, "frozen", False) if frozen is None else frozen
    if need == "R":
        return ("install R 4 with the packages lme4, lmerTest and emmeans; Rscript must be "
                "on the PATH or under C:/Program Files/R")
    if need == "pymc":
        if frozen:
            return ("the packaged app does not include PyMC: run the app from a source "
                    "install, after " + PYMC_PIP)
        return PYMC_PIP
    if need == "tdt":
        # Bundled in the packaged app, and part of the analysis extra.
        return "pip install tdt"
    return 'pip install -e ".[analysis]"'


# -- folders and layout -----------------------------------------------------------
@dataclass
class Layout:
    """Where each kind of folder is, guessed from the one folder the user picked."""
    picked: str = ""
    kind: str = "unknown"
    folders: Dict[str, str] = field(default_factory=dict)

    def describe(self) -> str:
        names = {"exp1": "Experiment 1", "grammar": "Experiment 2", "photometry": "photometry",
                 "survey": "pilot survey", "root": "data root"}
        found = [f"{names[k]}: {self.folders[k]}"
                 for k in ("root", "exp1", "grammar", "photometry", "survey")
                 if k in self.folders]
        return "\n".join(found) if found else "No experiment folder recognised here."


def _isdir(*parts) -> bool:
    return os.path.isdir(os.path.join(*parts))


def _looks_like_exp1(d: str) -> bool:
    return any(os.path.isdir(x) for x in glob.glob(os.path.join(glob.escape(d), "w[0-9]_*")))


def _looks_like_grammar(d: str) -> bool:
    return (os.path.basename(d).lower() == GRAMMAR_NAME
            or _isdir(d, "habituation") or _isdir(d, "day_1"))


def _looks_like_photometry(d: str) -> bool:
    return os.path.basename(d).lower() == PHOTOMETRY_NAME or _isdir(d, TANK_FOLDER)


def _looks_like_root(d: str) -> bool:
    return _isdir(d, EXP1_NAME) or _isdir(d, GRAMMAR_NAME) or _isdir(d, PHOTOMETRY_NAME)


def _root_folders(root: str) -> Dict[str, str]:
    out = {"root": root}
    for key, name in (("exp1", EXP1_NAME), ("grammar", GRAMMAR_NAME),
                      ("photometry", PHOTOMETRY_NAME)):
        if _isdir(root, name):
            out[key] = os.path.join(root, name)
    # survey_standalone_sessions.py writes beside the data root
    out["survey"] = os.path.join(os.path.dirname(root), os.path.basename(root) + "_survey")
    return out


def resolve_layout(folder: str) -> Layout:
    """Guess which experiment ``folder`` holds, and find its sibling folders.

    The data root (holding ``8_arms_w_voc``, ``grammar``, ``dopamine
    recordings``) gives every folder. One experiment folder gives that
    experiment, plus its siblings when its parent looks like the data root.
    A ``BATCH_ANALYSIS`` or ``FIBPHO_ANALYSIS`` folder is read as the folder
    above it.
    """
    lay = Layout(picked=folder or "")
    if not folder or not os.path.isdir(folder):
        return lay
    folder = os.path.normpath(os.path.abspath(folder))
    if os.path.basename(folder).upper() in ("BATCH_ANALYSIS", "FIBPHO_ANALYSIS"):
        folder = os.path.dirname(folder)
    if _looks_like_root(folder) and not _looks_like_exp1(folder):
        lay.kind = "root"
        lay.folders = _root_folders(folder)
        return lay
    if _looks_like_exp1(folder):
        key = "exp1"
    elif _looks_like_photometry(folder):
        key = "photometry"
    elif _looks_like_grammar(folder):
        key = "grammar"
    else:
        return lay
    lay.kind = key
    parent = os.path.dirname(folder)
    if parent and parent != folder and _looks_like_root(parent):
        lay.folders = _root_folders(parent)
    lay.folders[key] = folder
    return lay


# -- templates, status and requirements -------------------------------------------
def fill(template: str, folders: Mapping[str, str]) -> str:
    """Fill one template; raises KeyError naming the first missing folder."""
    keys = _placeholders(template)
    if not keys:
        return template
    for k in keys:
        if not folders.get(k):
            raise KeyError(k)
    return os.path.normpath(template.format(**{k: folders[k] for k in keys}))


def output_dir(p: Pipeline, folders: Mapping[str, str], shuffled: bool = False) -> Optional[str]:
    """The folder this script writes to, or None when it is not known."""
    if p.output is None:
        return folders.get("thesis") if p.thesis_required else None
    try:
        out = fill(p.output, folders)
    except KeyError:
        return None
    return out + "_shuffled" if (shuffled and p.shuffled) else out


def marker_path(p: Pipeline, folders: Mapping[str, str], shuffled: bool = False) -> Optional[str]:
    if p.output is None:
        return None
    if p.marker is None or (shuffled and p.shuffled):
        return output_dir(p, folders, shuffled)
    try:
        return fill(p.marker, folders)
    except KeyError:
        return None


def is_done(p: Pipeline, folders: Mapping[str, str], shuffled: bool = False) -> bool:
    """True when the script's output is on disk (always False for thesis-only scripts)."""
    path = marker_path(p, folders, shuffled)
    return bool(path) and os.path.exists(path)


def missing(p: Pipeline, folders: Mapping[str, str], env: Mapping[str, bool],
            options: Iterable[str] = ()) -> List[str]:
    """Reasons ``p`` cannot run now; empty when it can."""
    reasons = []
    for k in p.folders:
        if not folders.get(k):
            reasons.append(f"needs {FOLDER_NAMES[k]}")
    needs = list(p.needs)
    chosen = set(options)
    for o in p.options:
        if o.id in chosen:
            needs += [n for n in o.needs if n not in needs]
    for n in needs:
        if not env.get(n, False):
            reasons.append(f"needs {NEED_LABELS[n]}: {install_hint(n)}")
    return reasons


def build_args(p: Pipeline, folders: Mapping[str, str], shuffled: bool = False,
               thesis_dir: Optional[str] = None, options: Iterable[str] = ()) -> List[str]:
    """The script's command-line arguments. KeyError names a missing folder."""
    args = [fill(t, folders) for t in p.args]
    if shuffled and p.shuffled:
        args.append("--shuffled-only")
    if p.thesis and thesis_dir:
        args += ["--thesis-figures", thesis_dir]
    elif p.thesis_required:
        raise KeyError("thesis")
    chosen = set(options)
    for o in p.options:
        if o.id in chosen:
            args += list(o.args)
    return args


# -- run plans --------------------------------------------------------------------
@dataclass(frozen=True)
class Step:
    pipeline: Pipeline
    shuffled: bool
    reason: str          # "selected", or "needed by <id>"

    def label(self) -> str:
        s = self.pipeline.id + (" (blocks 2 to 4)" if self.shuffled else "")
        return s if self.reason == "selected" else f"{s}, {self.reason}"


def plan(selected: Iterable[str], shuffled: bool = False,
         done: Optional[Callable[[Pipeline, bool], bool]] = None,
         rerun_dependencies: bool = False) -> List[Step]:
    """Expand a selection with the scripts it depends on, in dependency order.

    A dependency is added only when ``done`` says its output is missing,
    unless ``rerun_dependencies``. "Blocks 2 to 4 only" applies to the scripts
    that take ``--shuffled-only``; a dependency runs in the variant its
    dependent reads, so a shuffled exp1_complexity pulls in the shuffled
    exp1_stimulus_pi while exp1_bd_model pulls in the full one.
    """
    done = done or (lambda p, s: False)
    steps: Dict[Tuple[str, bool], Step] = {}

    def visit(p: Pipeline, shuf: bool, reason: str, chain: Tuple[str, ...]) -> None:
        if p.id in chain:
            raise ValueError("dependency cycle: " + " -> ".join(chain + (p.id,)))
        key = (p.id, shuf)
        if key in steps:
            if reason == "selected":
                steps[key] = Step(p, shuf, reason)
            return
        steps[key] = Step(p, shuf, reason)
        for dep_id in p.after:
            if dep_id in chain + (p.id,):
                raise ValueError("dependency cycle: " + " -> ".join(chain + (p.id, dep_id)))
            dep = BY_ID[dep_id]
            dep_shuf = shuf and dep.shuffled
            if (dep_id, dep_shuf) in steps:
                continue
            if rerun_dependencies or not done(dep, dep_shuf):
                visit(dep, dep_shuf, f"needed by {p.id}", chain + (p.id,))

    for pid in selected:
        p = BY_ID[pid]
        visit(p, shuffled and p.shuffled, "selected", ())
    return sorted(steps.values(), key=lambda s: (ORDER[s.pipeline.id], s.shuffled))


def registry_problems() -> List[str]:
    """Dependencies that are unknown, cyclic, or listed after their dependent."""
    problems = []
    for p in PIPELINES:
        for d in p.after:
            if d not in BY_ID:
                problems.append(f"{p.id}: unknown dependency {d}")
            elif ORDER[d] > ORDER[p.id]:
                problems.append(f"{p.id}: dependency {d} is listed after it")
    if problems:
        return problems
    for p in PIPELINES:
        try:
            plan([p.id])
        except ValueError as e:
            problems.append(str(e))
    return problems


# -- packaging --------------------------------------------------------------------
def bundle_files(base: str) -> List[str]:
    """Files under ``base`` (analysis/auditory) the packaged app carries as data."""
    out: List[str] = []
    for pattern in ("*.py", "*.R", os.path.join("model_validation", "*.py")):
        out += glob.glob(os.path.join(base, pattern))
    return sorted(out)
