@echo off

setlocal

set REGION=us-east-1

for /f "delims=" %%i in ('aws sts get-caller-identity --query Account --output text') do set ACCOUNT=%%i

if "%ACCOUNT%"=="" (
    echo Could not read your AWS account id. Run "aws configure" first.
    exit /b 1
)

rem S3 bucket names are global - the account id keeps this one unique per AWS account
set BUCKET=research-agent-tfstate-%ACCOUNT%

set TABLE=research-agent-tf-locks

echo AWS account: %ACCOUNT%

echo Creating S3 bucket: %BUCKET% in region: %REGION%

rem Only skip creation when the bucket is really there; any other error (e.g. missing permissions) stops the script

aws s3api head-bucket --bucket %BUCKET% >nul 2>nul

if %errorlevel% equ 0 (
    echo Bucket already exists, continuing.
) else (
    aws s3api create-bucket --bucket %BUCKET% --region %REGION% >nul
    if errorlevel 1 goto :failed
    echo Bucket created.
)

echo Enabling versioning...

aws s3api put-bucket-versioning --bucket %BUCKET% --versioning-configuration Status=Enabled

if %errorlevel% neq 0 goto :failed

echo Blocking public access...

aws s3api put-public-access-block --bucket %BUCKET% --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

if %errorlevel% neq 0 goto :failed

echo Enabling server-side encryption...

aws s3api put-bucket-encryption --bucket %BUCKET% --server-side-encryption-configuration "{\"Rules\":[{\"ApplyServerSideEncryptionByDefault\":{\"SSEAlgorithm\":\"AES256\"}}]}"

if %errorlevel% neq 0 goto :failed

echo Creating DynamoDB table for state locking: %TABLE%

aws dynamodb describe-table --table-name %TABLE% --region %REGION% >nul 2>nul

if %errorlevel% equ 0 (
    echo DynamoDB table already exists, continuing.
) else (
    aws dynamodb create-table --table-name %TABLE% --attribute-definitions AttributeName=LockID,AttributeType=S --key-schema AttributeName=LockID,KeyType=HASH --billing-mode PAY_PER_REQUEST --region %REGION% >nul
    if errorlevel 1 goto :failed
    aws dynamodb wait table-exists --table-name %TABLE% --region %REGION%
    echo DynamoDB table created.
)

echo Writing terraform\backend.hcl...

> "%~dp0terraform\backend.hcl" echo bucket = "%BUCKET%"

>> "%~dp0terraform\backend.hcl" echo region = "%REGION%"

echo.
echo Bootstrap complete.
echo S3 bucket  : %BUCKET% (versioned, encrypted, private)
echo DynamoDB   : %TABLE% (state locking)
echo.
echo Next step: cd terraform then terraform init -reconfigure -backend-config=backend.hcl

endlocal

exit /b 0

:failed

echo.
echo Bootstrap FAILED - see the AWS error above. Nothing was written to terraform\backend.hcl.

endlocal

exit /b 1
