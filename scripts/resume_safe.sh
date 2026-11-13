#!/bin/bash
# Non-interactive resume (resume.bat asks for a typed "yes"): start RDS, then apply the resume plan only if it
# matches the expected shape - creates the paused resources, destroys nothing but replaced task definitions.
set -euo pipefail
REGION=us-east-1
DB_ID=research-agent-postgres
cd "$(dirname "$0")/../terraform"

status() { aws rds describe-db-instances --db-instance-identifier "$DB_ID" --region "$REGION" \
             --query "DBInstances[0].DBInstanceStatus" --output text; }

until [ "$(status)" = "stopped" ] || [ "$(status)" = "available" ]; do sleep 20; done
if [ "$(status)" = "stopped" ]; then
  echo "Starting RDS..."
  aws rds start-db-instance --db-instance-identifier "$DB_ID" --region "$REGION" >/dev/null
fi
aws rds wait db-instance-available --db-instance-identifier "$DB_ID" --region "$REGION"
echo "RDS available"

terraform plan -no-color -input=false -var="paused=false" -out=resume.tfplan > /dev/null
summary=$(terraform show -no-color resume.tfplan | grep -E "^Plan:")
echo "$summary"
# Expected: N to add, at most 1 change (EventBridge policy), at most 2 destroys (the replaced task definitions)
destroys=$(echo "$summary" | sed -E 's/.* ([0-9]+) to destroy.*/\1/')
unexpected=$(terraform show -no-color resume.tfplan | grep -E "will be destroyed" | grep -v "aws_ecs_task_definition" || true)
if [ "$destroys" -gt 2 ] || [ -n "$unexpected" ]; then
  echo "ABORT: unexpected destroys in the resume plan:"; echo "$unexpected"; rm -f resume.tfplan; exit 1
fi
terraform apply -no-color -input=false resume.tfplan | grep -E "Apply complete|Error"
rm -f resume.tfplan
echo "URL: http://$(terraform output -raw alb_dns)"
