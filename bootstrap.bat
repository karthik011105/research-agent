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

aws s3api create-bucket --bucket %BUCKET% --region %REGION% 2>nul

if %errorlevel% equ 0 (
    echo Bucket created.
) else (
    echo Bucket already exists, continuing.
)

echo Enabling versioning...

aws s3api put-bucket-versioning --bucket %BUCKET% --versioning-configuration Status=Enabled

echo Blocking public access...

aws s3api put-public-access-block --bucket %BUCKET% --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

echo Enabling server-side encryption...

aws s3api put-bucket-encryption --bucket %BUCKET% --server-side-encryption-configuration "{\"Rules\":[{\"ApplyServerSideEncryptionByDefault\":{\"SSEAlgorithm\":\"AES256\"}}]}"

echo Creating DynamoDB table for state locking: %TABLE%

aws dynamodb create-table --table-name %TABLE% --attribute-definitions AttributeName=LockID,AttributeType=S --key-schema AttributeName=LockID,KeyType=HASH --billing-mode PAY_PER_REQUEST --region %REGION% 2>nul

if %errorlevel% equ 0 (
    echo DynamoDB table created.
) else (
    echo DynamoDB table already exists, continuing.
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
