PYTHON ?= python3
TECTONIC ?= tectonic

.PHONY: test demo tables paper paper-short perf benchmark-versions leaderboard

test:
	$(PYTHON) -m pytest -q

demo:
	$(PYTHON) -m evidencescene.cli demo

perf:
	$(PYTHON) benchmarks/performance/bench_suite.py
	$(PYTHON) benchmarks/performance/bench_ablation.py

benchmark-versions:
	./benchmarks/performance/run_versions.sh 04a1b6b HEAD

leaderboard:
	$(PYTHON) tools/render_performance_leaderboard.py

# Regenerate the checked-in table fragments from the vendored receipts.
# Receipt hashes are verified by the generators before rendering.
tables:
	cd benchmarks && $(PYTHON) generate_table.py > ../papers/evidencescene/pose-table.tex
	cd benchmarks && $(PYTHON) generate_generated_mask_table.py > ../papers/evidencescene/generated-mask-table.tex
	cd benchmarks && $(PYTHON) sequence_summary.py > sequence-summary.json

paper-short:
	cd papers/evidencescene && $(TECTONIC) main.tex
	mv papers/evidencescene/main.pdf papers/evidencescene/evidencescene.pdf
	cp papers/evidencescene/evidencescene.pdf docs/assets/papers/evidencescene.pdf

paper: paper-short
