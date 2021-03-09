@echo off
setlocal enabledelayedexpansion  

for /r . %%a in (.vs) do (  
  if exist %%a (
  echo "delete" %%a
  rd /s /q "%%a" 
 )
)

for /r . %%a in (*.pdb) do (
  echo "delete" %%a
  del %%a
)
::for /r . %%a in (*.exe) do (
  ::echo "delete" %%a
::  del %%a
::)
pause
