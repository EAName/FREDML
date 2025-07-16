#!/bin/bash
# Grant E2E test permissions for FRED ML to IAM user 'edwin'
# Usage: bash scripts/aws_grant_e2e_policy.sh

set -e

POLICY_NAME="fredml-e2e-policy"
USER_NAME="edwin"
ACCOUNT_ID="785737749889"
BUCKET="fredmlv1"
POLICY_FILE="/tmp/${POLICY_NAME}.json"
POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${POLICY_NAME}"

cat > "$POLICY_FILE" <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "lambda:ListFunctions",
        "lambda:GetFunction",
        "lambda:InvokeFunction"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ssm:GetParameter"
      ],
      "Resource": "arn:aws:ssm:us-west-2:${ACCOUNT_ID}:parameter/fred-ml/api-key"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:ListBucket"
      ],
      "Resource": "arn:aws:s3:::${BUCKET}"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject"
      ],
      "Resource": "arn:aws:s3:::${BUCKET}/*"
    }
  ]
}
EOF

# Create the policy if it doesn't exist
if ! aws iam get-policy --policy-arn "$POLICY_ARN" > /dev/null 2>&1; then
  echo "Creating policy $POLICY_NAME..."
  aws iam create-policy --policy-name "$POLICY_NAME" --policy-document file://"$POLICY_FILE"
else
  echo "Policy $POLICY_NAME already exists."
fi

# Attach the policy to the user
aws iam attach-user-policy --user-name "$USER_NAME" --policy-arn "$POLICY_ARN"
echo "Policy $POLICY_NAME attached to user $USER_NAME." 