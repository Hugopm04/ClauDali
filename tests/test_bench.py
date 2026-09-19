"""Tests for the benchmark harness.

The benchmark generates 1,816 specs from a handful of YAML files, and it is
meant to run unattended for weeks. Both facts mean the failures worth catching
here are the quiet ones: an id that shifts and orphans a finished item's state,
a wave that no longer covers every sub-case, a decode path left on `auto` so
that two items stop being comparable, a reconcile that re-renders work already
on disk.

Compiling all 1,816 specs is the real gate and it lives in
``python -m bench plan --check``, because it takes the better part of a minute.
What runs here is a deterministic sample of it, plus the structural properties
that no amount of compiling would notice.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import catalog, report, runner  # noqa: E402
from bench.model import Calibration, Item, Record, State  # noqa: E402
from bench.progress import ProgressBar, RunProgress, human_duration  # noqa: E402


@pytest.fixture(scope="module")
def plan() -> list[Item]:
    return catalog.build_plan()


# ---------------------------------------------------------------------------
# The catalogue
# ---------------------------------------------------------------------------


def test_the_topic_catalogue_has_the_shape_it_claims():
    families, defaults = catalog.load_topics()
    subcases = [s for family in families.values() for s in family["subcases"].values()]
    prompts = [p for s in subcases for p in s["prompts"]]
    assert len(families) == 18
    assert len(subcases) == 90
    assert len(prompts) == 270
    assert all(s["prompts"] for s in subcases), "a sub-case with no prompts renders nothing"
    assert defaults["variations"] == 4


def test_every_family_and_subcase_says_what_it_is_testing():
    """Without it, a survey this size is a thousand images and no verdict."""
    families, _ = catalog.load_topics()
    for name, family in families.items():
        assert family.get("why", "").strip(), name
        for subcase_name, subcase in family["subcases"].items():
            assert subcase.get("why", "").strip(), f"{name}/{subcase_name}"


def test_item_ids_are_unique(plan):
    ids = [item.id for item in plan]
    assert len(ids) == len(set(ids))


def test_the_plan_is_the_size_the_readme_promises(plan):
    counts = {phase: sum(1 for i in plan if i.phase == phase) for i in plan for phase in [i.phase]}
    assert counts["survey"] == 90 * 3 * 2 * 3, "90 sub-cases x 3 prompts x 2 variants x 3 models"
    assert counts["settings"] == 153
    assert len(plan) == 1816


def test_phases_run_cheapest_and_riskiest_first(plan):
    """Features before settings before accuracy before survey.

    Features are cheap and they are what fails, so they belong in the first
    hour. Settings decide the options the survey then holds still, so running
    the survey first would risk 1,460 hours on a setting later shown wrong.
    """
    order = [item.phase for item in plan]
    first = {phase: order.index(phase) for phase in set(order)}
    last = {phase: len(order) - 1 - order[::-1].index(phase) for phase in set(order)}
    sequence = ["features", "settings", "accuracy", "survey"]
    for earlier, later in zip(sequence, sequence[1:]):
        assert last[earlier] < first[later], f"{earlier} must finish before {later} starts"


def test_the_smoke_pass_comes_before_the_full_one(plan):
    """A path that cannot run at all should say so in the first hour."""
    features = [item for item in plan if item.phase == "features"]
    scales = [item.tags["scale"] for item in features]
    assert scales == sorted(scales, key=lambda s: 0 if s == "smoke" else 1)


def test_the_settings_phase_is_ordered_shortest_first(plan):
    calibration = Calibration()
    estimates = [
        calibration.estimate(item) for item in plan if item.phase == "settings"
    ]
    assert estimates == sorted(estimates)


def test_the_init_source_is_rendered_before_anything_needs_it(plan):
    """img2img and inpainting load an image an earlier item produced."""
    for scale in ("smoke", "full"):
        items = [i for i in plan if i.phase == "features" and i.tags.get("scale") == scale]
        provider = next(i for i in items if i.tags.get("provides") == "init-source")
        consumers = [i for i in items if i.tags.get("needs") == "init-source"]
        assert consumers, scale
        for consumer in consumers:
            assert items.index(provider) < items.index(consumer), consumer.id


# ---------------------------------------------------------------------------
# The waves
# ---------------------------------------------------------------------------


def test_each_wave_covers_every_subcase_exactly_once_per_prompt(plan):
    """The property that makes stopping between waves safe.

    A wave that skipped a sub-case, or covered one twice, would mean stopping
    after it left a survey with a hole in it -- which is precisely what the
    wave ordering exists to prevent.
    """
    _families, _ = catalog.load_topics()
    survey = [item for item in plan if item.phase == "survey"]
    for wave in sorted({item.wave for item in survey}):
        items = [item for item in survey if item.wave == wave]
        keys = [(i.tags["family"], i.tags["subcase"], i.tags["prompt_index"]) for i in items]
        assert len(keys) == len(set(keys)), f"{wave} renders the same prompt twice"
        subcases = {(family, subcase) for family, subcase, _index in keys}
        assert len(subcases) == 90, f"{wave} misses a sub-case"


def test_the_first_wave_uses_each_topics_own_model(plan):
    """Wave 1 shows every topic on the checkpoint its intent would have picked."""
    for item in plan:
        if item.wave != "w01":
            continue
        assert item.model == catalog.INTENT_MODEL[item.tags["intent"]], item.id


def test_every_prompt_is_rendered_both_rich_and_minimal(plan):
    """The control that separates a weak topic from a badly chosen style key."""
    survey = [item for item in plan if item.phase == "survey"]
    pairs: dict[tuple, set[str]] = {}
    for item in survey:
        key = (item.tags["family"], item.tags["subcase"],
               item.tags["prompt_index"], item.tags["model"])
        pairs.setdefault(key, set()).add(item.tags["variant"])
    assert pairs, "no survey items"
    assert all(variants == {"rich", "minimal"} for variants in pairs.values())


def test_a_minimal_spec_carries_nothing_but_the_subject_and_intent(plan):
    minimal = next(i for i in plan if i.tags.get("variant") == "minimal")
    assert set(minimal.spec) == {"intent", "subject", "render", "composition"}
    assert set(minimal.spec["subject"]) == {"primary"}


def test_rich_null_removes_a_key_the_family_set():
    """A painterly sub-case has to be able to take the family's lens off again."""
    spec = {"camera": {"lens": "85mm", "shot": "scene"}}
    catalog._apply_rich(spec, {"lens": None})
    assert spec["camera"] == {"shot": "scene"}


def test_an_unknown_rich_key_is_refused_rather_than_ignored():
    with pytest.raises(KeyError):
        catalog._apply_rich({}, {"lightning": "neon"})


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------


def test_the_decode_path_is_pinned_everywhere_but_the_axis_that_tests_it(plan):
    """`auto` would make two items rendered on different evenings incomparable.

    It picks between a CPU fp32 untiled decode and a GPU tiled one by reading
    free RAM at that moment, so with `auto` one of the two silently gets the
    better decoder. Measured while building this: at 640x640 with 8.45 GB free,
    `auto` chose the CPU, drove free RAM to 0.0 GB and took 686 s against 100 s.
    """
    for item in plan:
        decode = item.spec["render"].get("vae_decode")
        if item.tags.get("axis") == "vae_decode":
            assert decode == item.tags["value"], item.id
        else:
            assert decode == catalog.PINNED_VAE_DECODE, item.id


def test_every_survey_item_shares_one_seed(plan):
    seeds = {
        item.spec["render"]["seed"]
        for item in plan
        if item.phase in {"survey", "accuracy"}
    }
    assert seeds == {catalog.SURVEY_SEED}, "comparisons need the same seeds everywhere"


def test_a_deterministic_sample_of_specs_compiles(plan):
    """The full sweep is `bench plan --check`; this is the fast regression lock."""
    from claudali.compiler import compile_spec
    from claudali.spec import load_spec

    sample = plan[::97]
    assert len(sample) > 15
    for item in sample:
        compiled = compile_spec(load_spec(item.spec))
        assert compiled.prompt, item.id
        unexpected = [
            warning
            for warning in compiled.warnings
            if "CLIP chunks" not in warning and "refiner was trained" not in warning
        ]
        assert not unexpected, f"{item.id}: {unexpected}"


def test_the_accuracy_battery_pairs_its_controls(plan):
    """Each text test has a composited twin that is correct by construction."""
    ids = {item.id for item in plan if item.phase == "accuracy"}
    for name in ("text-short", "text-long"):
        assert f"accuracy/{name}" in ids
        assert f"accuracy/{name}-overlay" in ids
    assert "accuracy/small-subject-plain" in ids
    assert "accuracy/small-subject-anchored" in ids


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------


def test_work_units_count_the_extra_pixels_a_hires_pass_samples():
    plain = Item(id="a", phase="p", wave="w", title="", why="", model="m", group="g",
                 spec={"render": {"steps": 10, "width": 1024, "height": 1024, "variations": 1}})
    hires = Item(id="b", phase="p", wave="w", title="", why="", model="m", group="g",
                 spec={"render": {"steps": 10, "width": 1024, "height": 1024, "variations": 1},
                       "hires": {"enabled": True, "scale": 2.0, "strength": 0.5}})
    assert plain.work_units() == pytest.approx(10 * 1.048576)
    # 5 hi-res steps at four times the pixels.
    assert hires.work_units() == pytest.approx(plain.work_units() + 5 * 1.048576 * 4)


def test_calibration_prefers_what_it_measured_over_what_it_assumed():
    calibration = Calibration()
    guessed = calibration.step_seconds(1024, 1024, "fp16")
    for _ in range(4):
        calibration.observe_step(1024, 1024, "fp16", 42.0)
    assert calibration.step_seconds(1024, 1024, "fp16") == 42.0
    assert guessed != 42.0


def test_the_cost_model_fits_an_intercept_once_it_has_two_sizes():
    """A sampler step has a fixed cost, and ignoring it wrecks the ETA.

    Extrapolating proportionally from one small measurement overestimates a
    large one badly: the first 640x640 reading alone projected the whole
    benchmark at 92 days against an expected 64. With two sizes the fit
    separates the per-step overhead from the per-pixel cost, and predicting
    1344x768 from 640x640 and 1024x1024 then lands inside the 23-26 s that
    CLAUDE.md measured for it.
    """
    calibration = Calibration()
    calibration.observe_step(640, 640, "fp16", 13.78)
    assert calibration.fit() == (0.0, pytest.approx(23.0)), "one size cannot fit an intercept"

    calibration.observe_step(1024, 1024, "fp16", 24.5)
    intercept, slope = calibration.fit()
    assert intercept > 0 and slope > 0
    assert 23.0 <= calibration.step_seconds(1344, 768, "fp16") <= 26.0


def test_a_negative_slope_is_never_believed():
    """A big render that happened to run on an idle machine must not invert the model."""
    calibration = Calibration()
    calibration.observe_step(640, 640, "fp16", 40.0)
    calibration.observe_step(1024, 1024, "fp16", 10.0)
    assert calibration.fit() == (0.0, pytest.approx(23.0))
    assert calibration.step_seconds(1344, 768, "fp16") > 0


def test_calibration_uses_a_median_so_one_resumed_first_step_cannot_skew_it():
    calibration = Calibration()
    for _ in range(9):
        calibration.observe_step(768, 768, "fp16", 12.0)
    calibration.observe_step(768, 768, "fp16", 900.0)  # a resume's first step
    assert calibration.step_seconds(768, 768, "fp16") == 12.0


def test_calibration_survives_a_corrupt_file(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text("{not json", encoding="utf-8")
    assert Calibration.load(path).per_size == {}


# ---------------------------------------------------------------------------
# State, grouping and resuming
# ---------------------------------------------------------------------------


def test_settings_items_run_alone_and_survey_items_group_by_model():
    def make(phase, model, index):
        return Item(id=f"{phase}/{index}", phase=phase, wave="w", title="", why="",
                    model=model, group="g", spec={"render": {"model": model}})

    settings = [make("settings", "juggernaut-xl", i) for i in range(3)]
    assert [len(batch) for batch in runner.group(settings)] == [1, 1, 1]

    survey = [make("survey", "juggernaut-xl", i) for i in range(3)]
    survey += [make("survey", "sdxl-base", i + 3) for i in range(2)]
    batches = list(runner.group(survey))
    assert [len(batch) for batch in batches] == [3, 2]
    assert [batch[0].model for batch in batches] == ["juggernaut-xl", "sdxl-base"]


def test_a_group_is_capped_so_no_worker_runs_forever():
    items = [
        Item(id=f"survey/{i}", phase="survey", wave="w", title="", why="",
             model="m", group="g", spec={})
        for i in range(runner.MAX_GROUP * 2 + 1)
    ]
    assert all(len(batch) <= runner.MAX_GROUP for batch in runner.group(items))


def test_reconcile_believes_the_bundle_over_the_state_file(tmp_path, monkeypatch):
    """A killed runner leaves items marked running whose work is on disk.

    Re-rendering those would repeat hours of work, and over a run this long it
    happens more than once.
    """
    monkeypatch.setattr("bench.model.RUNS_DIR", tmp_path)
    item = Item(id="features/smoke/x", phase="features", wave="w", title="", why="",
                model="m", group="g", spec={})
    monkeypatch.setattr(type(item), "bundle_dir", property(lambda self: tmp_path / self.id))
    item.bundle_dir.mkdir(parents=True)
    (item.bundle_dir / "result.json").write_text(
        json.dumps({
            "status": "done", "duration_s": 61.5,
            "variations": [{"decode": {"device": "gpu", "tiled": True}}],
            "timing": {"stages": {"base": {"s_per_step": 11.5}}},
            "memory": {"ram_free_start_gb": 9.1, "vram_peak_reserved_gb": 5.8},
        }),
        encoding="utf-8",
    )

    state = State({item.id: Record(id=item.id, status="running")}, path=tmp_path / "state.json")
    assert runner.reconcile([item], state) == 1
    record = state.records[item.id]
    assert record.status == "done"
    assert record.s_per_step == 11.5
    assert record.decode == "gpu/tiled"
    assert record.vram_peak_gb == 5.8
    assert state.is_finished(item.id)


def test_reconcile_leaves_a_finished_record_alone(tmp_path, monkeypatch):
    item = Item(id="x", phase="features", wave="w", title="", why="", model="m",
                group="g", spec={})
    monkeypatch.setattr(type(item), "bundle_dir", property(lambda self: tmp_path / self.id))
    state = State(
        {item.id: Record(id=item.id, status="failed", error="boom")},
        path=tmp_path / "state.json",
    )
    assert runner.reconcile([item], state) == 0
    assert state.records[item.id].error == "boom"


def test_a_state_saves_back_to_where_it_was_loaded(tmp_path):
    """Not a convenience. A default path here overwrites the real run's books.

    The runner saves state from several places. When ``save()`` defaulted to
    the module-level path, a State built anywhere else -- a test, a script,
    a report -- clobbered ``bench/state.json`` for whatever run was going on
    at the time. That happened, which is why this test exists.
    """
    path = tmp_path / "elsewhere.json"
    state = State(path=path)
    state.get("a").status = "done"
    state.save()
    assert path.exists()
    assert State.load(path).status("a") == "done"


def test_state_survives_a_round_trip(tmp_path):
    path = tmp_path / "state.json"
    state = State()
    state.get("a").status = "done"
    state.get("b").status = "paused"
    state.save(path)
    loaded = State.load(path)
    assert loaded.status("a") == "done"
    assert loaded.is_finished("a") and not loaded.is_finished("b")


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------


def test_the_percentage_has_enough_decimals_to_visibly_move():
    """One sampler step is ~0.0005% of this benchmark.

    At two decimals the bar would sit still for ten minutes at a time and look
    like a hang, which is the whole reason for the extra digits.
    """
    progress = RunProgress(total_units=200_000.0, done_units=46_837.0, total_items=1816)
    line = ProgressBar(enabled=False).render(progress)[0]
    assert "23.4185%" in line


def test_progress_never_goes_backwards_when_an_item_finishes():
    progress = RunProgress(total_units=100.0, total_items=2)
    progress.item_units, progress.item_fraction = 40.0, 0.9
    before = progress.fraction
    progress.done_units += 40.0
    progress.item_units, progress.item_fraction = 0.0, 0.0
    assert progress.fraction >= before


def test_the_eta_ignores_work_finished_in_an_earlier_session():
    """A run resumed at 60% must not claim credit for that 60% when estimating."""
    progress = RunProgress(total_units=100.0, done_units=60.0, resumed_units=60.0)
    assert progress.rate() is None
    progress.done_units = 70.0
    progress.started -= 100.0
    assert progress.rate() == pytest.approx(0.1, rel=0.05)
    assert progress.remaining_seconds() == pytest.approx(300.0, rel=0.05)


def test_human_duration_is_readable_at_every_scale():
    assert human_duration(45) == "45s"
    assert human_duration(3 * 60 + 7) == "3m 07s"
    assert human_duration(5 * 3600 + 120) == "5h 02m"
    assert human_duration(63 * 86400) == "63d 00h 00m"
    assert human_duration(None) == "unknown"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def test_the_report_writes_a_row_per_finished_item(tmp_path, plan):
    items = plan[:4]
    state = State({item.id: Record(id=item.id, status="done", duration_s=12.0) for item in items})
    path = report.write_csv(items, state, tmp_path / "measurements.csv")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(items) + 1
    assert lines[0].split(",")[:3] == ["id", "phase", "wave"]


def test_the_report_skips_items_that_have_not_run(tmp_path, plan):
    path = report.write_csv(plan[:4], State(), tmp_path / "m.csv")
    assert path.read_text(encoding="utf-8").strip().count("\n") == 0, "header only"


# ---------------------------------------------------------------------------
# Stopping and resuming
# ---------------------------------------------------------------------------


class _FakeProcess:
    """A worker that emits the given lines, optionally raising Ctrl+C partway."""

    def __init__(self, lines, interrupt_after=None):
        self._lines = list(lines)
        self._interrupt_after = interrupt_after
        self.stderr = None
        self.waited = False
        self.killed = False

    @property
    def stdout(self):
        def generate():
            for index, line in enumerate(self._lines):
                if self._interrupt_after is not None and index == self._interrupt_after:
                    self._interrupt_after = None
                    raise KeyboardInterrupt
                yield line
        return generate()

    def wait(self):
        self.waited = True

    def kill(self):  # pragma: no cover - the runner must never call this
        self.killed = True


def _events(item_id, *extra):
    return [json.dumps({"event": "start", "item": item_id, "title": "t", "model": "m"})] + [
        json.dumps(event) for event in extra
    ]


def _runner_over(monkeypatch, tmp_path, process, items):
    state = State(path=tmp_path / "state.json")
    calibration = Calibration()
    monkeypatch.setattr(Calibration, "save", lambda self, path=None: None)
    run = runner.Runner(state, calibration, ProgressBar(enabled=False))
    monkeypatch.setattr(run, "_spawn", lambda batch: process)
    plan = runner.Plan(
        all_items=items,
        outstanding=items,
        total_units=sum(i.work_units() for i in items),
        done_units=0.0,
    )
    return run, state, plan


def _item(item_id, phase="settings"):
    return Item(
        id=item_id, phase=phase, wave="w", title="t", why="", model="m", group="g",
        spec={"render": {"steps": 4, "width": 512, "height": 512, "variations": 1}},
    )


def test_ctrl_c_leaves_the_item_paused_and_the_worker_alone(monkeypatch, tmp_path):
    """The runner must not kill a worker that is writing its checkpoint.

    Ctrl+C in a console reaches the worker too, so by the time the runner sees
    KeyboardInterrupt the worker is already pausing. Killing it there would
    throw away the checkpoint mid-write, which is the one thing that must not
    happen: that checkpoint is the whole reason a multi-week run can be stopped.
    """
    item = _item("settings/a")
    lines = _events(
        item.id,
        {"event": "pausing"},
        {"event": "paused", "item": item.id, "bundle": str(tmp_path / "b"), "images": 2},
        {"event": "interrupted"},
    )
    process = _FakeProcess(lines, interrupt_after=1)
    run, state, plan = _runner_over(monkeypatch, tmp_path, process, [item])

    run.run(plan)

    assert run.interrupted
    assert not process.killed, "the worker must be left to finish its checkpoint"
    assert process.waited, "the runner must wait for it"
    assert state.status(item.id) == "paused"
    assert state.records[item.id].images == 2


def test_a_paused_item_is_outstanding_again_next_session(monkeypatch, tmp_path):
    items = [_item("settings/a"), _item("settings/b")]
    state = State(path=tmp_path / "state.json")
    state.get("settings/a").status = "paused"
    monkeypatch.setattr(runner, "reconcile", lambda items, state, save=True: 0)
    plan = runner.prepare(items, state)
    assert [item.id for item in plan.outstanding] == ["settings/a", "settings/b"]
    assert plan.done_units == 0.0


def test_a_failed_item_does_not_stop_the_run(monkeypatch, tmp_path):
    """One bad item out of 1,816 must not end a four-week run."""
    items = [_item("settings/a"), _item("settings/b")]
    processes = [
        _FakeProcess(_events("settings/a", {"event": "failed", "item": "settings/a",
                                            "error": "OutOfMemoryError: boom"})),
        _FakeProcess(_events("settings/b", {"event": "done", "item": "settings/b",
                                            "images": 1, "duration_s": 12.0,
                                            "timing": {}, "memory": {}})),
    ]
    state = State(path=tmp_path / "state.json")
    monkeypatch.setattr(Calibration, "save", lambda self, path=None: None)
    run = runner.Runner(state, Calibration(), ProgressBar(enabled=False))
    monkeypatch.setattr(run, "_spawn", lambda batch: processes.pop(0))
    plan = runner.Plan(all_items=items, outstanding=items,
                       total_units=sum(i.work_units() for i in items), done_units=0.0)

    progress = run.run(plan)

    assert state.status("settings/a") == "failed"
    assert "OutOfMemoryError" in state.records["settings/a"].error
    assert state.status("settings/b") == "done"
    assert progress.failed_items == 1 and progress.done_items == 1


def test_a_finished_item_advances_the_whole_run_bar(monkeypatch, tmp_path):
    item = _item("settings/a")
    process = _FakeProcess(_events(
        item.id,
        {"event": "step", "item": item.id, "stage": "base", "step": 2, "steps": 4,
         "variation": 1, "variations": 1, "done": 2.0, "total": 4.0},
        {"event": "done", "item": item.id, "images": 1, "duration_s": 40.0,
         "timing": {"stages": {"base": {"s_per_step": 9.5}},
                    "steps": {"base": {"mean_s": 9.5}},
                    "spans": [{"name": "load", "seconds": 42.0}]},
         "memory": {"ram_free_start_gb": 9.0, "vram_peak_reserved_gb": 5.4},
         "decode": {"device": "gpu", "tiled": True}},
    ))
    run, state, plan = _runner_over(monkeypatch, tmp_path, process, [item])

    progress = run.run(plan)

    assert progress.fraction == pytest.approx(1.0)
    record = state.records[item.id]
    assert record.status == "done" and record.s_per_step == 9.5
    assert record.decode == "gpu/tiled" and record.vram_peak_gb == 5.4
    # The item's measurements reached the cost model.
    assert run.calibration.step_seconds(512, 512, "fp16") == 9.5
    assert run.calibration.load_seconds() == 42.0
