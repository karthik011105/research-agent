#!/bin/bash
# Recreates everything pause.sh removed, using the :latest images in ECR.
set -e

REGION=${AWS_REGION:-us-east-1}
DB_ID="research-agent-postgres"

cd "$(dirname "$0")/terraform"

echo "Starting RDS..."
aws rds start-db-instance --db-instance-identifier "$DB_ID" --region "$REGION" >/dev/null 2>&1 || true
aws rds wait db-instance-available --db-instance-identifier "$DB_ID" --region "$REGION"

echo "Recreating ALB, Redis, VPC endpoints, ECS services, auto-scaling and weekly red team (5-10 min)..."
terraform apply -var="paused=false"

echo ""
echo "Running. App URL: http://$(terraform output -raw alb_dns)"
echo "The ALB DNS name changes on every resume - update any bookmarks."
