#!/bin/bash
# Stops everything that bills by the hour. Data in RDS, images in ECR and secrets are kept.
# Bring it back with ./resume.sh
set -e

REGION=${AWS_REGION:-us-east-1}
DB_ID="research-agent-postgres"

cd "$(dirname "$0")/terraform"

echo "Making sure RDS is available (Terraform can't modify a stopped database)..."
aws rds start-db-instance --db-instance-identifier "$DB_ID" --region "$REGION" >/dev/null 2>&1 || true
aws rds wait db-instance-available --db-instance-identifier "$DB_ID" --region "$REGION"

echo "Removing ALB, Redis, VPC endpoints, ECS services, auto-scaling and weekly red team..."
terraform apply -var="paused=true"

echo "Stopping RDS..."
aws rds stop-db-instance --db-instance-identifier "$DB_ID" --region "$REGION" >/dev/null

echo ""
echo "Paused. Remaining cost is RDS storage, ECR images and Secrets Manager (a few cents a day)."
echo "Note: AWS restarts a stopped RDS instance automatically after 7 days - run ./pause.sh again or stop it in the console."
