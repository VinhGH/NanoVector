@echo off
git config core.hooksPath .githooks

if errorlevel 1 (
    echo Failed to configure Git hooks.
    exit /b 1
)

echo Git hooks configured successfully.
