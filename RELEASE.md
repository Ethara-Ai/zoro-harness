# RetailBench Public Release Notes

This file records the repository-level release checklist for publishing
RetailBench as a public benchmark repository.

## Public Entry Points

- Project README: [`README.md`](README.md)
- Benchmark homepage: [`docs/index.html`](docs/index.html)
- Paper: [arXiv:2603.16453](https://arxiv.org/abs/2603.16453)
- Citation metadata: [`CITATION.cff`](CITATION.cff)
- Paper-facing metric artifacts: [`paper_submit_data/outputs`](paper_submit_data/outputs)

## GitHub Pages

The static site is contained entirely under `docs/`. The workflow
[`pages.yml`](.github/workflows/pages.yml) deploys that directory when changes
land on `main` or `master`.

Local preview:

```bash
python3 -m http.server 8000 --directory docs
```

Open `http://localhost:8000`.

## Reproducibility Scope

The public repository should include:

- simulator and agent runners;
- curated paper-facing CSV/JSON outputs;
- four-stage diagnostic figures and reports;
- data-access and reconstruction notes.

The public repository should not include:

- local API credentials, bot tokens, or user-specific connector settings;
- raw private LLM rollout logs;
- local runtime databases;
- historical manuscript zip archives;
- local build caches, Python bytecode, and temporary files.

## Release Hygiene Applied

The following local-only files were removed from Git tracking with
`git rm --cached` and are ignored going forward:

- `.claude/settings.local.json`
- `claude.toml`
- `test.py`
- `log1`
- `log2`
- `paper.zip`
- `paper/*.zip`
- `.skill_build/`
- `script/test.py`
- the accidental root file whose filename is a single space

The files remain in the local checkout if they existed before cleanup, but they
should not be committed or bundled into a manual source archive.

## Verification Commands

```bash
node --check docs/assets/site.js
python3 -m http.server 8000 --directory docs
```

For paper-facing figures:

```bash
python3 paper_submit_data/render_four_stage_report.py
```
