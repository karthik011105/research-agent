#!/bin/bash
# Non-interactive resume (resume.bat asks for a typed "yes"): start RDS, then apply the resume plan only if it
# matches the expected shape - creates the paused resources, destroys nothing but replaced task definitions.
set -euo pipefail
REGION=us-east-1
DB_ID=research-agent-postgres
cd "$(dirname "$0")/../terraform"

# ECS won't start TensorZero if a Groq key it references is missing from the secret: check names first (values never printed)
missing=$(aws secretsmanager get-secret-value --secret-id research-agent/config --region "$REGION" \
  --query SecretString --output text | python -c "
import json, sys
s = json.load(sys.stdin)
need = ['GROQ_API_KEY', 'GROQ_API_KEY_2', 'GROQ_API_KEY_3']
print(' '.join(k for k in need if not str(s.get(k, '')).strip() or s.get(k) == 'REPLACE_ME'))")
if [ -n "$missing" ]; then
  echo "ABORT: set these in Secrets Manager -> research-agent/config before resuming: $missing"; exit 1
fi

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
