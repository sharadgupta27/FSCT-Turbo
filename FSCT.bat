@echo off
setlocal EnableDelayedExpansion

REM ===================================================================
REM  FSCT - single entry point
REM
REM  Just double-click this file.
REM
REM  First run  : sets everything up by itself - conda environment,
REM               PyTorch, FSCT dependencies, the browser UI packages
REM               and LAStools. No questions asked.
REM  Later runs : asks only what you want to launch.
REM
REM  Advanced (not shown in the menu):
REM      FSCT.bat gui          launch the desktop app
REM      FSCT.bat web          launch the browser UI
REM      FSCT.bat setup        re-run setup, keeping the environment
REM      FSCT.bat setup /force rebuild the environment from scratch
REM      FSCT.bat verify       run the installation checks
REM      FSCT.bat lastools     re-download LAStools
REM      FSCT.bat version      print the version and exit
REM      FSCT.bat help         command list
REM
REM  Version matrix (pinned - the torch / torch-geometric /
REM  torch-cluster triple must agree or the model will not load):
REM      Python 3.11, PyTorch 2.5.1,
REM      torch-geometric 2.6.1, torch-cluster 1.6.3
REM ===================================================================

set "ENV_NAME=lidar"
set "PY_VERSION=3.11"
set "TORCH_VERSION=2.5.1"
set "PYG_VERSION=2.6.1"

set "HERE=%~dp0"

REM The app version lives in version.py so there is one source of truth.
REM Parsed with findstr rather than python, because this runs before the
REM conda environment is known to exist.
set "APP_VERSION=unknown"
for /f "tokens=2 delims==" %%v in ('findstr /b /c:"__version__" "%HERE%version.py" 2^>nul') do (
    for /f "tokens=* delims= " %%w in ("%%v") do set "APP_VERSION=%%~w"
)
set "CONDA="
set "RECREATE="
set "MENUTRIES=0"

REM Bumped whenever the package set changes, so an existing install is
REM refreshed rather than silently left on the old stack.
set "SETUP_VERSION=2"
set "STAMP=%HERE%.fsct_setup"

if /i "%~2"=="/force"    set "RECREATE=1"
if /i "%~2"=="/recreate" set "RECREATE=1"

if /i "%~1"=="gui"       goto cmd_gui
if /i "%~1"=="web"       goto cmd_web
if /i "%~1"=="setup"     goto forced_setup
if /i "%~1"=="install"   goto forced_setup
if /i "%~1"=="verify"    goto cmd_verify
if /i "%~1"=="lastools"  goto cmd_lastools
if /i "%~1"=="version"   goto cmd_version
if /i "%~1"=="--version" goto cmd_version
if /i "%~1"=="help"      goto cmd_help
if /i "%~1"=="/?"        goto cmd_help
if not "%~1"=="" (
    echo.
    echo Unknown command: %~1
    goto cmd_help
)


REM ===================================================================
REM  Default path: make sure everything is installed, then show the menu
REM ===================================================================
:auto
call :find_conda
if errorlevel 1 goto fail

REM Setup is needed if the environment is missing or the stamp does not
REM match this script's SETUP_VERSION. The stamp keeps startup instant on
REM every subsequent launch instead of importing torch to probe the env.
call :env_exists
if errorlevel 1 goto run_setup
if not exist "%STAMP%" goto run_setup
set "STAMPED="
for /f "usebackq delims=" %%V in ("%STAMP%") do set "STAMPED=%%V"
if not "!STAMPED!"=="%SETUP_VERSION%" goto run_setup
goto menu

:forced_setup
call :find_conda
if errorlevel 1 goto fail

:run_setup
echo.
echo ========================================
echo   FSCT - first-time setup
echo ========================================
echo.
echo Setting everything up. Nothing is required from you.
echo.
echo   - Python %PY_VERSION% environment ('%ENV_NAME%')
echo   - PyTorch %TORCH_VERSION% and PyTorch Geometric %PYG_VERSION%
echo   - FSCT dependencies and both user interfaces
echo   - LAStools ^(downloaded and configured automatically^)
echo.
echo This takes 15-25 minutes the first time and needs about 6 GB.
echo Please leave this window open.
echo.

echo [1/6] Checking your hardware...
"%CONDA%" --version
if errorlevel 1 (
    echo ERROR: conda was found but will not run.
    goto fail
)
set "CUDA_TAG=cpu"
nvidia-smi >nul 2>&1
if not errorlevel 1 (
    set "CUDA_TAG=cu121"
    echo   Nvidia GPU detected - using the CUDA 12.1 build.
) else (
    echo   No Nvidia GPU detected - using the CPU build.
    echo   FSCT works on CPU, but segmentation will be much slower.
)
set "TORCH_INDEX=https://download.pytorch.org/whl/!CUDA_TAG!"
set "PYG_WHEELS=https://data.pyg.org/whl/torch-%TORCH_VERSION%+!CUDA_TAG!.html"
echo.

echo [2/6] Preparing the '%ENV_NAME%' environment...
call :env_exists
if errorlevel 1 goto create_env
if not defined RECREATE (
    echo   Reusing the existing environment and refreshing packages.
    goto env_ready
)
echo   Removing the old environment...
"%CONDA%" env remove -n %ENV_NAME% -y
if errorlevel 1 (
    echo.
    echo ERROR: Could not remove the existing environment.
    echo Close anything using it, including running FSCT windows, and retry.
    goto fail
)

:create_env
echo   Creating '%ENV_NAME%' with Python %PY_VERSION%...
"%CONDA%" create -n %ENV_NAME% python=%PY_VERSION% pip -y -c conda-forge
if errorlevel 1 (
    echo.
    echo ERROR: Failed to create the environment.
    goto fail
)

:env_ready
call :set_runners
echo.

echo [3/6] Installing the scientific stack from conda-forge...
echo       numpy, scipy, scikit-learn, laspy, hdbscan and friends
echo.
REM libblas=*=*openblas forces the OpenBLAS build of the BLAS/LAPACK stack.
REM conda-forge otherwise pulls Intel MKL, and MKL 2026.1.0 fails to resolve a
REM delay-loaded export on this platform: numpy.linalg.svd / lstsq kill the
REM process outright with 0xC06D007F, no Python traceback. That takes out
REM cylinder fitting completely, since fit_cylinder uses svd and skimage's
REM CircleModel uses lstsq. OpenBLAS is marginally slower for big matrices but
REM the matrices here are tiny, and it actually works.
"%CONDA%" install -n %ENV_NAME% -y -c conda-forge --override-channels ^
    "libblas=*=*openblas" ^
    "numpy>=1.26" pandas scipy scikit-learn scikit-image matplotlib ^
    networkx laspy lazrs-python tqdm joblib hdbscan jinja2 rdflib pillow ^
    pywavelets fsspec requests
if errorlevel 1 (
    echo.
    echo ERROR: conda-forge package installation failed. See above.
    goto fail
)
echo.

REM PyTorch comes from pip, NOT conda. Mixing the 'pytorch' channel with
REM conda-forge lets the solver pair a CPU-only pytorch with a CUDA
REM torchaudio - it "succeeds", then fails at runtime.
REM
REM torchvision is deliberately NOT installed. Nothing in FSCT imports it and
REM torch-geometric does not require it, but its bundled image DLLs shadow the
REM ones conda-forge's Pillow links against, which breaks _imaging with
REM "DLL load failed ... The operating system cannot run %%1" for every
REM package that touches PIL.
echo [4/6] Installing PyTorch %TORCH_VERSION% (!CUDA_TAG!)...
echo       This is the big one - several GB. Please be patient.
echo.
!PIP! torch==%TORCH_VERSION% --index-url !TORCH_INDEX!
if errorlevel 1 (
    echo.
    echo ERROR: PyTorch installation failed.
    echo If the download timed out, simply run this again.
    goto fail
)
!PIP! torch-geometric==%PYG_VERSION%
if errorlevel 1 (
    echo.
    echo ERROR: torch-geometric installation failed.
    goto fail
)
REM torch-cluster supplies the fps/radius/knn operators scripts\model.py
REM needs. Prebuilt Windows wheels, so no Visual Studio required.
!PIP! torch-cluster -f !PYG_WHEELS!
if errorlevel 1 (
    echo.
    echo ERROR: torch-cluster installation failed.
    echo No prebuilt wheel matched, so pip tried to build from source.
    echo Check that !PYG_WHEELS! lists a cp311 win_amd64 wheel.
    goto fail
)
echo.

echo [5/6] Installing FSCT dependencies and both interfaces...
REM Everything else comes from requirements.txt - the single requirements
REM file. Both interfaces are installed up front so the menu can offer
REM either one without a separate step.
REM
REM By this point conda-forge has installed the scientific stack and pip has
REM installed the torch triple, so this pass only adds what is still missing:
REM mdutils/markdown for scripts\report_writer.py, python-louvain (imported
REM as 'community') and scikit-spatial for scripts\measure.py, customtkinter
REM for the desktop app, and streamlit/plotly/pydeck/statsmodels for the
REM browser UI.
REM
REM Every bound in requirements.txt is open at the top precisely so this runs
REM as a no-op against the conda-forge builds. The file previously capped
REM numpy<2.2 and pandas<3.0 while this installer produced numpy 2.4 and
REM pandas 3.0, so pointing pip at it here would have pulled PyPI wheels over
REM the OpenBLAS builds. If you tighten a bound, re-check it with:
REM     pip install -r requirements.txt --dry-run
REM which must report nothing to install on a freshly built environment.
if not exist "%HERE%requirements.txt" (
    echo.
    echo ERROR: requirements.txt is missing.
    goto fail
)
!PIP! -r "%HERE%requirements.txt" -f !PYG_WHEELS!
if errorlevel 1 (
    echo.
    echo ERROR: pip package installation failed.
    goto fail
)
echo.

echo [6/6] Setting up LAStools and verifying...
REM Downloads to third_party\LAStools and records the path in
REM gui_config.json, so neither interface has to ask for it.
!RUN! python "%HERE%setup_lastools.py"
if errorlevel 1 (
    echo.
    echo   Warning: LAStools could not be downloaded automatically.
    echo   FSCT still runs - only the external 3D viewer is affected.
)
echo.
!RUN! python "%HERE%test_installation.py"
if errorlevel 1 (
    echo.
    echo ========================================
    echo   Setup finished WITH PROBLEMS
    echo ========================================
    echo.
    echo Some checks failed. Run 'FSCT.bat setup /force' to rebuild
    echo the environment from scratch.
    goto fail
)

REM Only stamp after the checks pass, so a broken install retries next time.
> "%STAMP%" echo %SETUP_VERSION%

echo.
echo ========================================
echo   Setup complete
echo ========================================
echo.
pause
goto menu


REM ===================================================================
REM  Menu - the only thing regular runs show
REM ===================================================================
:menu
set /a "MENUTRIES+=1"
REM With redirected or closed stdin, set /p leaves PICK empty and this loop
REM would spin forever. Bail out after a few dead reads.
if !MENUTRIES! GTR 12 (
    echo.
    echo No input received - exiting.
    exit /b 1
)
cls
echo ========================================
echo   FSCT - Forest Structural Complexity Tool  v%APP_VERSION%
echo ========================================
echo.
echo   1.  Desktop app
echo   2.  Browser UI
echo.
echo   0.  Exit
echo.
set "PICK="
set /p "PICK=  Choose [0-2]: "
if not "!PICK!"=="" set "MENUTRIES=0"

if "!PICK!"=="1" goto cmd_gui
if "!PICK!"=="2" goto cmd_web
if "!PICK!"=="0" exit /b 0
goto menu


REM ===================================================================
REM  Launchers
REM ===================================================================
:cmd_gui
call :require_env
if errorlevel 1 goto fail
echo.
echo Starting the FSCT desktop app...
echo.
REM Run from the project directory so model\model.pth, gui_config.json and
REM scripts\ resolve no matter where this was launched from.
pushd "%HERE%"
"%CONDA%" run -n %ENV_NAME% --no-capture-output python fsct_desktop.py
set "EXITCODE=!ERRORLEVEL!"
popd
if not "!EXITCODE!"=="0" (
    echo.
    echo ERROR: The application exited with code !EXITCODE!.
    echo If you see missing-module errors, run:  FSCT.bat setup /force
    goto fail
)
exit /b 0

:cmd_web
call :require_env
if errorlevel 1 goto fail
echo.
echo The browser UI will open in your default browser.
echo To stop the server, press Ctrl+C in this window.
echo.
pushd "%HERE%"
"%CONDA%" run -n %ENV_NAME% --no-capture-output streamlit run fsct_web.py
set "EXITCODE=!ERRORLEVEL!"
popd
if not "!EXITCODE!"=="0" (
    echo.
    echo ERROR: Streamlit exited with code !EXITCODE!.
    goto fail
)
exit /b 0


REM ===================================================================
REM  Maintenance commands
REM ===================================================================
:cmd_verify
call :require_env
if errorlevel 1 goto fail
call :set_runners
!RUN! python "%HERE%test_installation.py"
if errorlevel 1 goto fail
goto done

:cmd_lastools
call :require_env
if errorlevel 1 goto fail
call :set_runners
echo Downloading LAStools...
echo.
!RUN! python "%HERE%setup_lastools.py" --force
if errorlevel 1 (
    echo.
    echo ERROR: LAStools could not be downloaded.
    echo Install it manually from https://rapidlasso.de/lastools/ and set
    echo the path in the app's Settings tab.
    goto fail
)
goto done

:cmd_version
echo FSCT %APP_VERSION%
goto done

:cmd_help
echo.
echo FSCT - Forest Structural Complexity Tool  v%APP_VERSION%
echo.
echo   FSCT.bat                  set up if needed, then choose what to launch
echo   FSCT.bat gui              launch the desktop app
echo   FSCT.bat web              launch the browser UI
echo   FSCT.bat setup            re-run setup, keeping the environment
echo   FSCT.bat setup /force     rebuild the environment from scratch
echo   FSCT.bat verify           run the installation checks
echo   FSCT.bat lastools         re-download LAStools
echo   FSCT.bat version          print the version and exit
echo   FSCT.bat help             this list
echo.
echo See USAGE.md for details.
goto done


REM ===================================================================
REM  Subroutines
REM ===================================================================

REM Sets CONDA to the path of conda.exe. Returns 1 if not found.
:find_conda
set "CONDA="
call :probe "%CONDA_EXE%\..\.."
if defined CONDA exit /b 0
call :probe "C:\ProgramData\miniforge3"
if defined CONDA exit /b 0
call :probe "%USERPROFILE%\miniforge3"
if defined CONDA exit /b 0
call :probe "%LOCALAPPDATA%\miniforge3"
if defined CONDA exit /b 0
call :probe "C:\ProgramData\miniconda3"
if defined CONDA exit /b 0
call :probe "%USERPROFILE%\miniconda3"
if defined CONDA exit /b 0
call :probe "C:\ProgramData\Anaconda3"
if defined CONDA exit /b 0
call :probe "%USERPROFILE%\Anaconda3"
if defined CONDA exit /b 0
for /f "delims=" %%C in ('where conda.exe 2^>nul') do (
    set "CONDA=%%C"
    exit /b 0
)
echo.
echo ERROR: Could not find a conda installation.
echo.
echo FSCT needs Miniforge. Install it, then run this file again:
echo   https://github.com/conda-forge/miniforge/releases/latest
echo.
echo Searched:
echo   C:\ProgramData\miniforge3      %USERPROFILE%\miniforge3
echo   %LOCALAPPDATA%\miniforge3
echo   C:\ProgramData\miniconda3      %USERPROFILE%\miniconda3
echo   C:\ProgramData\Anaconda3       %USERPROFILE%\Anaconda3
echo   and your PATH
exit /b 1

:probe
if "%~1"=="" exit /b 0
if exist "%~1\Scripts\conda.exe" set "CONDA=%~1\Scripts\conda.exe"
exit /b 0

REM Returns 0 if the environment exists, 1 otherwise.
REM 'conda env list' prints the name at the start of the line; anchoring with
REM /B and requiring the trailing space stops '%ENV_NAME%_old' from matching.
:env_exists
"%CONDA%" env list | findstr /B /C:"%ENV_NAME% " >nul 2>&1
exit /b %ERRORLEVEL%

REM find_conda + env_exists, falling into setup when the env is missing.
:require_env
if not defined CONDA call :find_conda
if errorlevel 1 exit /b 1
call :env_exists
if errorlevel 1 (
    echo The '%ENV_NAME%' environment is missing - running setup first.
    goto run_setup
)
exit /b 0

REM RUN/PIP prefixes. The quotes around %CONDA% are embedded on purpose -
REM conda may live under a path containing spaces and these are expanded
REM unquoted at the call sites.
:set_runners
set RUN="%CONDA%" run -n %ENV_NAME% --no-capture-output
set PIP=%RUN% python -m pip install --disable-pip-version-check
exit /b 0


REM ===================================================================
REM  Exits
REM ===================================================================
:fail
echo.
pause
exit /b 1

:done
echo.
pause
exit /b 0
