# Building the application

The Windows build bundles Python and every library, so the result runs on a machine with nothing installed. The exact library versions used for a build are pinned in `packaging/requirements-lock.txt` (generated with `pip freeze` on a machine where the test suite passes); the build script installs those versions first, so two builds from the same commit are identical.

```powershell
python -m venv .venv-build
.\.venv-build\Scripts\Activate.ps1
.\packaging\build_windows.ps1
```

This runs the tests, builds `dist\amazeing-app\amazeing-app.exe` with PyInstaller, smoke-tests the executable, and zips the result as `dist\amazeing-app.zip`, which is the file name the README and the installation page tell people to download. The same executable runs the command-line tools (for example `amazeing-app.exe --entry auditory --config session.yaml`), which is how the application launches sessions on a machine without Python.

## Starting the application on this machine

`dist\` holds two launchers, so there is one folder to look in rather than two places to remember. The build script copies them in from `packaging/dist_launchers/`, and they are not part of the release zip.

| Double-click | What it starts |
|---|---|
| `Run from source.cmd` | The code in `src/` as it stands now, through your Python install. Use this while working on the code: an edit shows up the next time you start it. The console window it leaves open is where a failure to start is reported. |
| `Run the packaged app.cmd` | The build above, with Python and every library bundled inside it. It is a snapshot, so a code change does not reach it until the build script is run again. |

To edit what the launchers do, change the files in `packaging/dist_launchers/` and run the build script, or copy them into `dist\` by hand.

Refreshing the pins after upgrading a library: install the new version, run the tests, then `pip freeze --exclude-editable > packaging/requirements-lock.txt`.

---
