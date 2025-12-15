@echo off
setlocal EnableExtensions DisableDelayedExpansion
echo.
echo === Protected areas 30x30 environment setup (Windows) ===

REM ---- 1. CONFIGURATION ----
REM Determine project folder name
for %%I in ("%~dp0.") do set PROJECT_NAME=%%~nxI
set ENV_NAME=%PROJECT_NAME%_env

REM Set relative path to the sibling dependency
REM %~dp0 is the script dir. We want one level up, then seals_dev
set "PA3030_DIR_NAME=pa_dev"
set "PA3030_REPO_URL=git@github.com:gianluca-kaufmann/master_thesis.git"

REM ---- 2. CHECK TOOLING ----
REM Find mamba / micromamba
set "MAMBA="
where mamba >nul 2>nul
if errorlevel 1 goto try_micromamba
set "MAMBA=mamba"
goto have_mamba

:try_micromamba
where micromamba >nul 2>nul
if errorlevel 1 goto no_mamba
set "MAMBA=micromamba"

:have_mamba
REM Require git
where git >nul 2>nul
if errorlevel 1 goto no_git

REM ---- 3. SETUP PATHS ----
pushd "%~dp0"
set "SCRIPT_DIR=%CD%"
set "ENV_YML=%SCRIPT_DIR%\environment.yml"
REM Calculate full path to sibling directory for checking existence
cd ..
set "PARENT_DIR=%CD%"
set "PA3030_FULL_PATH=%PARENT_DIR%\%PA3030_DIR_NAME%"
REM Return to script dir
cd "%SCRIPT_DIR%"

echo Script directory: %SCRIPT_DIR%
echo Sibling Project:  %PA3030_FULL_PATH%
echo Conda Env Name:   %ENV_NAME%

REM ---- 4. CLONE SIBLING REPO (The "Missing Link") ----
echo.
echo Step 1/3: Checking local dependencies...

if exist "%PA3030_FULL_PATH%\" (
    echo   [OK] '%PA3030_DIR_NAME%' already exists locally.
) else (
    echo   [UPDATE] '%PA3030_DIR_NAME%' not found. Cloning from GitHub...
    git clone %PA3030_REPO_URL% "%PA3030_FULL_PATH%"
    if errorlevel 1 goto clone_fail
    echo   [OK] Clone successful.
)

REM ---- 5. CONDA ENVIRONMENT ----
echo.
echo Step 2/3: Creating or updating conda environment...

if not exist "%ENV_YML%" goto no_yml

REM Check if env exists
"%MAMBA%" env list | findstr /I /C:"%ENV_NAME%" >nul 2>nul
if errorlevel 1 goto create_from_yml
goto update_from_yml

:create_from_yml
echo   Env not found; creating...
"%MAMBA%" env create -n "%ENV_NAME%" -f "%ENV_YML%"
if errorlevel 1 goto env_fail
goto vs_code_setup

:update_from_yml
echo   Env exists; updating...
"%MAMBA%" env update -n "%ENV_NAME%" -f "%ENV_YML%" --prune
if errorlevel 1 goto env_fail
goto vs_code_setup

REM ---- 6. VS CODE INTROSPECTION HELPER ----
:vs_code_setup
echo.
echo Step 3/3: Configuring VS Code helper...
REM Create a .env file to help VS Code resolve paths even if env isn't active yet
(
echo PYTHONPATH=../%PA3030_DIR_NAME%
) > .env
echo   [OK] Created .env file pointing to %PA3030_DIR_NAME%

goto done

:done
echo.
echo ==========================================================
echo ✅ SETUP COMPLETE
echo ==========================================================
echo.
echo 1. To activate the environment:
echo    conda activate %ENV_NAME%
echo.
echo 2. VS Code / Pylance Setup:
echo    - Open Command Palette (Ctrl+Shift+P)
echo    - Type: "Python: Select Interpreter"
echo    - Select: "%ENV_NAME%"
echo    - Pylance will now automatically find 'seals' in ../%PA3030_DIR_NAME%
echo.
echo ==========================================================

popd
endlocal
exit /b 0

REM ---- ERROR HANDLERS ----
:no_yml
echo.
echo ERROR: environment.yml not found at: %ENV_YML%
popd
endlocal
exit /b 1

:clone_fail
echo.
echo ERROR: Failed to clone pa3030_dev. Check internet or permissions.
popd
endlocal
exit /b 1

:env_fail
echo.
echo ERROR: Conda environment creation/update failed. See output above.
popd
endlocal
exit /b 1

:no_mamba
echo.
echo ERROR: mamba (or micromamba) not found on PATH.
endlocal
exit /b 1

:no_git
echo.
echo ERROR: git not found on PATH.
endlocal
exit /b 1