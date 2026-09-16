@echo off
rem Launch the application from the repository source.
rem
rem This is the one to use while the code is being worked on: it runs whatever
rem is in src\ right now, so an edit shows up the next time you start it. It
rem needs Python and an editable install:  pip install -e ".[gui,analysis]"
rem
rem The console window stays open behind the application on purpose. If the
rem application fails to start, the reason is printed there.

cd /d "%~dp0.."
python -m amazeing.app
if errorlevel 1 (
    echo.
    echo The application exited with an error. The message above says why.
    pause
)
