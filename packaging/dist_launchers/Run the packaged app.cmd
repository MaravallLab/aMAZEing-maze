@echo off
rem Launch the packaged application: the same build a colleague gets when they
rem unzip a release, with Python and every library bundled inside it.
rem
rem It is a snapshot. Changes to the code do not reach it until it is rebuilt
rem with  packaging\build_windows.ps1

if not exist "%~dp0amazeing-app\amazeing-app.exe" (
    echo The packaged application has not been built yet.
    echo Build it from the repository root with:  .\packaging\build_windows.ps1
    pause
    exit /b 1
)
start "" "%~dp0amazeing-app\amazeing-app.exe"
