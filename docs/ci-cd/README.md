# CI/CD Pipeline Documentation

## Overview

The FRED ML project uses GitHub Actions for comprehensive CI/CD automation. The pipeline includes multiple workflows for different purposes:

- **Main CI/CD Pipeline** (`ci-cd.yml`): Full deployment pipeline for main branch
- **Pull Request Checks** (`pull-request.yml`): Quality checks for PRs and development
- **Scheduled Maintenance** (`scheduled.yml`): Automated maintenance tasks
- **Release Deployment** (`release.yml`): Versioned releases and production deployments

## Workflow Overview

### 🚀 Main CI/CD Pipeline (`ci-cd.yml`)

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` branch
- Daily scheduled runs at 2 AM UTC

**Jobs:**
1. **🧪 Test & Quality**: Linting, type checking, unit tests
2. **🔗 Integration Tests**: AWS integration testing
3. **🚀 End-to-End Tests**: Complete system testing
4. **🔒 Security Scan**: Security vulnerability scanning
5. **⚡ Deploy Lambda**: AWS Lambda function deployment
6. **🏗️ Deploy Infrastructure**: AWS infrastructure deployment
7. **🎨 Deploy Streamlit**: Streamlit Cloud deployment preparation
8. **📢 Notifications**: Deployment status notifications

### 🔍 Pull Request Checks (`pull-request.yml`)

**Triggers:**
- Pull requests to `main` or `develop` branches
- Push to `develop` branch

**Jobs:**
1. **🔍 Code Quality**: Formatting, linting, type checking
2. **🧪 Unit Tests**: Unit test execution with coverage
3. **🔒 Security Scan**: Security vulnerability scanning
4. **📦 Dependency Check**: Outdated dependencies and security
5. **📚 Documentation Check**: README and deployment docs validation
6. **🏗️ Build Test**: Lambda package and Streamlit app testing
7. **💬 Comment Results**: Automated PR comments with results

### ⏰ Scheduled Maintenance (`scheduled.yml`)

**Triggers:**
- Daily at 6 AM UTC: Health checks
- Weekly on Sundays at 8 AM UTC: Dependency updates
- Monthly on 1st at 10 AM UTC: Performance testing

**Jobs:**
1. **🏥 Daily Health Check**: AWS service status monitoring
2. **📦 Weekly Dependency Check**: Dependency updates and security
3. **⚡ Monthly Performance Test**: Performance benchmarking
4. **🧹 Cleanup Old Artifacts**: S3 cleanup and maintenance

### 🎯 Release Deployment (`release.yml`)

**Triggers:**
- GitHub releases (published)

**Jobs:**
1. **📦 Create Release Assets**: Lambda packages, docs, test results
2. **🚀 Deploy to Production**: Production deployment
3. **🧪 Production Tests**: Post-deployment testing
4. **📢 Notify Stakeholders**: Release notifications

## Required Secrets

Configure these secrets in your GitHub repository settings:

### AWS Credentials
```bash
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
```

### FRED API
```bash
FRED_API_KEY=your_fred_api_key
```

## Environment Variables

The workflows use these environment variables:

```yaml
AWS_REGION: us-west-2
S3_BUCKET: fredmlv1
LAMBDA_FUNCTION: fred-ml-processor
PYTHON_VERSION: '3.9'
```

## Workflow Features

### 🔄 Automated Testing
- **Unit Tests**: pytest with coverage reporting
- **Integration Tests**: AWS service integration
- **End-to-End Tests**: Complete system validation
- **Security Scans**: Bandit security scanning
- **Performance Tests**: Load and performance testing

### 🏗️ Infrastructure as Code
- **S3 Bucket**: Automated bucket creation and configuration
- **Lambda Function**: Automated deployment and configuration
- **EventBridge Rules**: Quarterly scheduling automation
- **SSM Parameters**: Secure parameter storage

### 📊 Monitoring & Reporting
- **Code Coverage**: Automated coverage reporting to Codecov
- **Test Results**: Detailed test result artifacts
- **Security Reports**: Vulnerability scanning reports
- **Performance Metrics**: Performance benchmarking

### 🔒 Security
- **Secret Management**: Secure handling of API keys
- **Vulnerability Scanning**: Automated security checks
- **Access Control**: Environment-based deployment controls
- **Audit Trail**: Complete deployment logging

## Deployment Process

### Development Workflow
1. Create feature branch from `develop`
2. Make changes and push to branch
3. Create pull request to `develop`
4. Automated checks run on PR
5. Merge to `develop` after approval
6. Automated testing on `develop` branch

### Production Deployment
1. Create pull request from `develop` to `main`
2. Automated checks and testing
3. Merge to `main` triggers production deployment
4. Lambda function updated
5. Infrastructure deployed
6. Production tests run
7. Notification sent

### Release Process
1. Create GitHub release with version tag
2. Automated release asset creation
3. Production deployment
4. Post-deployment testing
5. Stakeholder notification

## Monitoring & Alerts

### Health Checks
- Daily AWS service status monitoring
- Lambda function availability
- S3 bucket accessibility
- EventBridge rule status

### Performance Monitoring
- Monthly performance benchmarking
- Response time tracking
- Resource utilization monitoring
- Error rate tracking

### Security Monitoring
- Weekly dependency vulnerability scans
- Security best practice compliance
- Access control monitoring
- Audit log review

## Troubleshooting

### Common Issues

#### Lambda Deployment Failures
```bash
# Check Lambda function status
aws lambda get-function --function-name fred-ml-processor --region us-west-2

# Check CloudWatch logs
aws logs describe-log-groups --log-group-name-prefix /aws/lambda/fred-ml-processor
```

#### S3 Access Issues
```bash
# Check S3 bucket permissions
aws s3 ls s3://fredmlv1 --region us-west-2

# Test bucket access
aws s3 cp test.txt s3://fredmlv1/test.txt
```

#### EventBridge Rule Issues
```bash
# Check EventBridge rules
aws events list-rules --name-prefix "fred-ml" --region us-west-2

# Test rule execution
aws events test-event-pattern --event-pattern file://event-pattern.json
```

### Debug Workflows

#### Enable Debug Logging
Add to workflow:
```yaml
env:
  ACTIONS_STEP_DEBUG: true
  ACTIONS_RUNNER_DEBUG: true
```

#### Check Workflow Logs
1. Go to GitHub repository
2. Click "Actions" tab
3. Select workflow run
4. View detailed logs for each job

## Best Practices

### Code Quality
- Use pre-commit hooks for local checks
- Maintain high test coverage (>80%)
- Follow PEP 8 style guidelines
- Use type hints throughout codebase

### Security
- Never commit secrets to repository
- Use least privilege AWS IAM policies
- Regularly update dependencies
- Monitor security advisories

### Performance
- Optimize Lambda function cold starts
- Use S3 lifecycle policies for cleanup
- Monitor AWS service quotas
- Implement proper error handling

### Documentation
- Keep README updated
- Document deployment procedures
- Maintain architecture diagrams
- Update troubleshooting guides

## Advanced Configuration

### Custom Workflow Triggers
```yaml
on:
  push:
    branches: [ main, develop ]
    paths: [ 'lambda/**', 'frontend/**' ]
  pull_request:
    branches: [ main ]
    paths-ignore: [ 'docs/**' ]
```

### Environment-Specific Deployments
```yaml
jobs:
  deploy:
    environment:
      name: ${{ github.ref == 'refs/heads/main' && 'production' || 'staging' }}
      url: ${{ steps.deploy.outputs.url }}
```

### Conditional Job Execution
```yaml
jobs:
  deploy:
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    runs-on: ubuntu-latest
```

## Support

For issues with the CI/CD pipeline:

1. Check workflow logs in GitHub Actions
2. Review this documentation
3. Check AWS CloudWatch logs
4. Contact the development team

## Contributing

To contribute to the CI/CD pipeline:

1. Create feature branch
2. Make changes to workflow files
3. Test locally with `act` (GitHub Actions local runner)
4. Create pull request
5. Ensure all checks pass
6. Get approval from maintainers 