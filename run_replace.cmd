@echo off
setlocal
if "%~1"=="" (
  echo Usage: run_replace.cmd "person_image" "background_image"
  exit /b 1
)
if "%~2"=="" (
  echo Usage: run_replace.cmd "person_image" "background_image"
  exit /b 1
)
cd /d "%~dp0\.."
dotnet run --project photo_background_replacer\PhotoBackgroundReplacer -- --input "%~1" --background "%~2"
