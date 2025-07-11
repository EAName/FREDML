#!/usr/bin/env python3
"""
AWS Deployment Script for FRED ML
Deploys Lambda function, S3 bucket, and EventBridge rule
"""

import boto3
import json
import os
import zipfile
import tempfile
import shutil
from pathlib import Path
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FredMLDeployer:
    def __init__(self, region='us-east-1'):
        """Initialize the deployer with AWS clients"""
        self.region = region
        self.cloudformation = boto3.client('cloudformation', region_name=region)
        self.s3 = boto3.client('s3', region_name=region)
        self.lambda_client = boto3.client('lambda', region_name=region)
        self.ssm = boto3.client('ssm', region_name=region)
        
    def create_lambda_package(self, source_dir: str, output_path: str):
        """Create Lambda deployment package"""
        logger.info("Creating Lambda deployment package...")
        
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add Python files
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, source_dir)
                        zipf.write(file_path, arcname)
            
            # Add requirements
            requirements_path = os.path.join(source_dir, 'requirements.txt')
            if os.path.exists(requirements_path):
                zipf.write(requirements_path, 'requirements.txt')
    
    def deploy_s3_bucket(self, stack_name: str, bucket_name: str):
        """Deploy S3 bucket using CloudFormation"""
        logger.info(f"Deploying S3 bucket: {bucket_name}")
        
        template_path = Path(__file__).parent.parent / 'infrastructure' / 's3' / 'bucket.yaml'
        
        with open(template_path, 'r') as f:
            template_body = f.read()
        
        try:
            response = self.cloudformation.create_stack(
                StackName=stack_name,
                TemplateBody=template_body,
                Parameters=[
                    {
                        'ParameterKey': 'BucketName',
                        'ParameterValue': bucket_name
                    }
                ],
                Capabilities=['CAPABILITY_NAMED_IAM']
            )
            
            logger.info(f"Stack creation initiated: {response['StackId']}")
            return response['StackId']
            
        except self.cloudformation.exceptions.AlreadyExistsException:
            logger.info(f"Stack {stack_name} already exists, updating...")
            
            response = self.cloudformation.update_stack(
                StackName=stack_name,
                TemplateBody=template_body,
                Parameters=[
                    {
                        'ParameterKey': 'BucketName',
                        'ParameterValue': bucket_name
                    }
                ],
                Capabilities=['CAPABILITY_NAMED_IAM']
            )
            
            logger.info(f"Stack update initiated: {response['StackId']}")
            return response['StackId']
    
    def deploy_lambda_function(self, function_name: str, s3_bucket: str, api_key: str):
        """Deploy Lambda function"""
        logger.info(f"Deploying Lambda function: {function_name}")
        
        # Create deployment package
        lambda_dir = Path(__file__).parent.parent / 'lambda'
        package_path = tempfile.mktemp(suffix='.zip')
        
        try:
            self.create_lambda_package(str(lambda_dir), package_path)
            
            # Update SSM parameter with API key
            try:
                self.ssm.put_parameter(
                    Name='/fred-ml/api-key',
                    Value=api_key,
                    Type='SecureString',
                    Overwrite=True
                )
                logger.info("Updated FRED API key in SSM")
            except Exception as e:
                logger.error(f"Failed to update API key: {e}")
            
            # Deploy function code
            with open(package_path, 'rb') as f:
                code = f.read()
            
            try:
                # Try to update existing function
                self.lambda_client.update_function_code(
                    FunctionName=function_name,
                    ZipFile=code
                )
                logger.info(f"Updated Lambda function: {function_name}")
                
            except self.lambda_client.exceptions.ResourceNotFoundException:
                # Create new function
                template_path = Path(__file__).parent.parent / 'infrastructure' / 'lambda' / 'function.yaml'
                
                with open(template_path, 'r') as f:
                    template_body = f.read()
                
                # Replace placeholder with actual code
                template_body = template_body.replace(
                    'import json\ndef lambda_handler(event, context):\n    return {\n        \'statusCode\': 200,\n        \'body\': json.dumps(\'Hello from Lambda!\')\n    }',
                    'import json\ndef lambda_handler(event, context):\n    return {\n        \'statusCode\': 200,\n        \'body\': json.dumps(\'FRED ML Lambda Function\')\n    }'
                )
                
                stack_name = f"{function_name}-stack"
                
                response = self.cloudformation.create_stack(
                    StackName=stack_name,
                    TemplateBody=template_body,
                    Parameters=[
                        {
                            'ParameterKey': 'FunctionName',
                            'ParameterValue': function_name
                        },
                        {
                            'ParameterKey': 'S3BucketName',
                            'ParameterValue': s3_bucket
                        }
                    ],
                    Capabilities=['CAPABILITY_NAMED_IAM']
                )
                
                logger.info(f"Lambda stack creation initiated: {response['StackId']}")
            
        finally:
            # Clean up
            if os.path.exists(package_path):
                os.remove(package_path)
    
    def deploy_eventbridge_rule(self, stack_name: str, lambda_function: str, s3_bucket: str):
        """Deploy EventBridge rule for quarterly scheduling"""
        logger.info(f"Deploying EventBridge rule: {stack_name}")
        
        template_path = Path(__file__).parent.parent / 'infrastructure' / 'eventbridge' / 'quarterly-rule.yaml'
        
        with open(template_path, 'r') as f:
            template_body = f.read()
        
        try:
            response = self.cloudformation.create_stack(
                StackName=stack_name,
                TemplateBody=template_body,
                Parameters=[
                    {
                        'ParameterKey': 'LambdaFunctionName',
                        'ParameterValue': lambda_function
                    },
                    {
                        'ParameterKey': 'S3BucketName',
                        'ParameterValue': s3_bucket
                    }
                ],
                Capabilities=['CAPABILITY_NAMED_IAM']
            )
            
            logger.info(f"EventBridge stack creation initiated: {response['StackId']}")
            return response['StackId']
            
        except self.cloudformation.exceptions.AlreadyExistsException:
            logger.info(f"Stack {stack_name} already exists, updating...")
            
            response = self.cloudformation.update_stack(
                StackName=stack_name,
                TemplateBody=template_body,
                Parameters=[
                    {
                        'ParameterKey': 'LambdaFunctionName',
                        'ParameterValue': lambda_function
                    },
                    {
                        'ParameterKey': 'S3BucketName',
                        'ParameterValue': s3_bucket
                    }
                ],
                Capabilities=['CAPABILITY_NAMED_IAM']
            )
            
            logger.info(f"EventBridge stack update initiated: {response['StackId']}")
            return response['StackId']
    
    def wait_for_stack_completion(self, stack_name: str):
        """Wait for CloudFormation stack to complete"""
        logger.info(f"Waiting for stack {stack_name} to complete...")
        
        waiter = self.cloudformation.get_waiter('stack_create_complete')
        try:
            waiter.wait(StackName=stack_name)
            logger.info(f"Stack {stack_name} completed successfully")
        except Exception as e:
            logger.error(f"Stack {stack_name} failed: {e}")
            raise
    
    def deploy_all(self, bucket_name: str, function_name: str, api_key: str):
        """Deploy all components"""
        logger.info("Starting FRED ML deployment...")
        
        try:
            # Deploy S3 bucket
            s3_stack_name = f"{bucket_name}-stack"
            self.deploy_s3_bucket(s3_stack_name, bucket_name)
            self.wait_for_stack_completion(s3_stack_name)
            
            # Deploy Lambda function
            self.deploy_lambda_function(function_name, bucket_name, api_key)
            
            # Deploy EventBridge rule
            eventbridge_stack_name = f"{function_name}-eventbridge-stack"
            self.deploy_eventbridge_rule(eventbridge_stack_name, function_name, bucket_name)
            self.wait_for_stack_completion(eventbridge_stack_name)
            
            logger.info("FRED ML deployment completed successfully!")
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Deploy FRED ML to AWS')
    parser.add_argument('--region', default='us-west-2', help='AWS region')
    parser.add_argument('--bucket', default='fredmlv1', help='S3 bucket name')
    parser.add_argument('--function', default='fred-ml-processor', help='Lambda function name')
    parser.add_argument('--api-key', required=True, help='FRED API key')
    
    args = parser.parse_args()
    
    deployer = FredMLDeployer(region=args.region)
    deployer.deploy_all(args.bucket, args.function, args.api_key)

if __name__ == "__main__":
    main() 