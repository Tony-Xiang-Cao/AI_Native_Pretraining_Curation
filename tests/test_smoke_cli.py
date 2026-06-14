import importlib

import autocurate


def test_public_api_exports():
    for name in ("Document", "CurationLoop", "hillclimb", "evaluate_gate",
                 "StreamMonitor", "CrawlMonitorRoutine", "text_features",
                 "build_report", "HTMLGate", "OCRGate", "JSONGate"):
        assert hasattr(autocurate, name)


def test_all_submodules_import():
    for m in ("schema", "utils", "lexicons", "datagen", "metrics", "report",
              "pipeline", "judge", "cli", "svgplot", "experiments",
              "profile.heuristics", "profile.outliers",
              "extract.html_gate", "extract.ocr_gate", "extract.json_gate",
              "verify.corruptions", "verify.harness",
              "hillclimb.base", "hillclimb.offline", "hillclimb.agentic",
              "agentloop.spc", "agentloop.ledger", "agentloop.routine"):
        importlib.import_module(f"autocurate.{m}")


def test_cli_profile_runs(capsys):
    from autocurate.cli import main
    main(["profile", "The study measured the effect across three trials."])
    out = capsys.readouterr().out
    assert "gzip_ratio" in out
