# Historical v7 source fixture

`deep_v7_sources.zip` contains exact source bytes from commit
`1fc57983d2e0e2b3c8b3921d8a3339ab870f86ab`, before the v8/v9 changes.
Its `provenance.json` records each SHA256. These match the accepted planner X
manifest and deep-search baseline; none of their hashes were repinned.

The fixture keeps historical replay validation independent of changing
production files and of Git history availability in shallow CI checkouts.
Replay loads only the accepted historical planner. CLI contract tests extract
the archived source files into their temporary directory to exercise strict
manifest checks. The archived simulator is used only with mocked dispatch.

Production continues to use the shared current rules engine. This archive
does not enable another production engine or restart normalization/clipping
experiments. Tests separately assert that current ranking, normalization and
safety arbitration preserve the accepted baseline's function ASTs and behavior.
