#!/bin/bash
# Non-interactive pause (pause.bat asks for a typed "yes"): apply the pause plan only if it destroys nothing but the
# resources pause is meant to remove, then stop RDS. Data (RDS, ECR images, secret, guardrail, S3 state) is kept.
set -euo pipefail
REGION=us-east-1
DB_ID=research-agent-postgres
cd "$(dirname "$0")/../terraform"

terraform plan -no-color -input=false -var="paused=true" -out=pause.tfplan > /dev/null
terraform show -no-color pause.tfplan | grep -E "^Plan:"
# Anything holding data must never appear in a destroy line
forbidden=$(terraform show -no-color pause.tfplan | grep -E "(will be destroyed|must be replaced)" \
  | grep -E "aws_db_instance|aws_ecr_repository|aws_secretsmanager|aws_bedrock_guardrail|aws_s3|aws_dynamodb|aws_vpc\.|aws_subnet" || true)
if [ -n "$forbidden" ]; then
  echo "ABORT: the pause plan would destroy data or network resources:"; echo "$forbidden"; rm -f pause.tfplan; exit 1
fi
terraform apply -no-color -input=false pause.tfplan | grep -E "Apply complete|Error"
rm -f pause.tfplan
aws rds stop-db-instance --db-instance-identifier "$DB_ID" --region "$REGION" --query "DBInstance.DBInstanceStatus" --output text
echo "Paused. Resume with: bash scripts/resume_safe.sh"
