# Architecture reference review

Inspected on 2026-11-17, before project files were created. Both repositories
were read through the GitHub API/raw files; no repository code was copied.

| Reference | Inspected `main` revision | Material inspected |
| --- | --- | --- |
| [Rxd3/NALTRA](https://github.com/Rxd3/NALTRA/tree/bc0c40fdb2267445ffccc8060ec684ca0627b286) | `bc0c40fdb2267445ffccc8060ec684ca0627b286` | Repository tree, README, CONTRIBUTING, pyproject, docs/architecture.md |
| [Rxd3/RECAP](https://github.com/Rxd3/RECAP/tree/8edf76eaad8238989524c677426647cee965563a) | `8edf76eaad8238989524c677426647cee965563a` | Repository tree, README, CONTRIBUTING, TEAM_TASKS, pyproject |

NALTRA informs separation of data preparation, model work, inference/decoding,
offline evaluation, reporting, shared schemas, and provenance utilities. TURAS
adopts one prediction contract, reusable saved predictions, configuration outside
code, and compatibility-aware artifacts. It does not copy NALTRA's text
classification, taxonomy, translation, ensemble implementations, or results.

RECAP informs six numbered workstreams with branches, owned paths, task
checklists, deliverables, dependencies, objective completion criteria, and
handoffs. TURAS also adopts independent PR review, coordinated shared-contract
changes, individual evidence records, merge commits, and unchecked work until
evidence exists. It does not copy RECAP's punctuation/casing models, completed
checkboxes, project-specific tests, personal details, or experiment outcomes.

The new `main`, README, CONTRIBUTING, TEAM_TASKS, contributions, docs, configs,
src package, scripts, data, experiments, results, tests, presentation, pyproject,
and ignore rules follow the user-confirmed repository conventions. The explicit
ASR/RDL scope and package name take precedence over either reference's layout.
