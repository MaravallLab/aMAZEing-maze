# Building the application

The Windows build bundles Python and every library, so the result runs on a machine with nothing installed. The exact library versions used for a build are pinned in `packaging/requirements-lock.txt` (generated with `pip freeze` on a machine where the test suite passes); the build script installs those versions first, so two builds from the same commit are identical.

```powershell
python -m venv .venv-build
.\.venv-build\Scripts\Activate.ps1
.\packaging\build_windows.ps1
```

This runs the tests, builds `dist\amazeing-app\amazeing-app.exe` with PyInstaller, and smoke-tests the executable. Zip the `dist\amazeing-app` folder to distribute it. The same executable runs the command-line tools (for example `amazeing-app.exe --entry auditory --config session.yaml`), which is how the application launches sessions on a machine without Python.

Refreshing the pins after upgrading a library: install the new version, run the tests, then `pip freeze --exclude-editable > packaging/requirements-lock.txt`.

---
