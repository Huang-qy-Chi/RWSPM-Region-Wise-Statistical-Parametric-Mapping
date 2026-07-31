@echo off
REM ============================================================
REM Compile cball_ext.dll for Windows (MinGW-w64 / MSYS2)
REM Usage: double-click or run from command prompt
REM Requires: MinGW-w64 (gcc) in PATH
REM
REM NOTE: If you have libgomp (OpenMP), add -fopenmp to both
REM compile and link commands for parallel speedup.
REM ============================================================

echo Compiling cball_ext.dll ...

gcc -shared -O3 ^
    cball_wrapper.c bdd_matrix.c utilities.c median.c ^
    -I. -lm ^
    -o cball_ext.dll

if %ERRORLEVEL% EQU 0 (
    echo SUCCESS: cball_ext.dll created
    dir cball_ext.dll
) else (
    echo FAILED: compilation error
)
pause
