@echo off

rem Recreates everything pause.bat removed, using the :latest images in ECR.

setlocal

set REGION=us-east-1

set DB_ID=research-agent-postgres

cd /d "%~dp0terraform"

echo Starting RDS...

aws rds start-db-instance --db-instance-identifier %DB_ID% --region %REGION% >nul 2>nul

aws rds wait db-instance-available --db-instance-identifier %DB_ID% --region %REGION%

if %errorlevel% neq 0 exit /b %errorlevel%

echo Recreating ALB, Redis, VPC endpoints, ECS services, auto-scaling and weekly red team (5-10 min)...

terraform apply -var="paused=false"

if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo Running. App URL:

terraform output -raw alb_dns

echo.
echo The ALB DNS name changes on every resume - update any bookmarks.

endlocal
