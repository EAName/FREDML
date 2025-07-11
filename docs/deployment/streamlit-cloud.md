# Streamlit Cloud Deployment Guide

This guide explains how to deploy the FRED ML frontend to Streamlit Cloud.

## Prerequisites

1. **GitHub Account**: Your code must be in a GitHub repository
2. **Streamlit Cloud Account**: Sign up at [streamlit.io/cloud](https://streamlit.io/cloud)
3. **AWS Credentials**: Configured for S3 and Lambda access

## Step 1: Prepare Your Repository

### Repository Structure

Ensure your repository has the following structure:

```
FRED_ML/
├── frontend/
│   ├── app.py
│   └── .streamlit/
│       └── config.toml
├── requirements.txt
└── README.md
```

### Update requirements.txt

Make sure your `requirements.txt` includes Streamlit dependencies:

```txt
streamlit==1.28.1
plotly==5.17.0
altair==5.1.2
boto3==1.34.0
pandas==2.1.4
numpy==1.24.3
```

## Step 2: Configure Streamlit App

### Main App File

Your `frontend/app.py` should be the main entry point. Streamlit Cloud will automatically detect and run this file.

### Streamlit Configuration

The `.streamlit/config.toml` file should be configured for production:

```toml
[global]
developmentMode = false

[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
```

## Step 3: Deploy to Streamlit Cloud

### 1. Connect Repository

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository
5. Set the main file path to `frontend/app.py`

### 2. Configure Environment Variables

In the Streamlit Cloud dashboard, add these environment variables:

```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=us-west-2

# Application Configuration
S3_BUCKET=fredmlv1
LAMBDA_FUNCTION=fred-ml-processor
```

### 3. Advanced Settings

- **Python version**: 3.9 or higher
- **Dependencies**: Use `requirements.txt` from root directory
- **Main file path**: `frontend/app.py`

## Step 4: Environment Variables Setup

### AWS Credentials

Create an IAM user with minimal permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::fredmlv1",
                "arn:aws:s3:::fredmlv1/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "lambda:InvokeFunction"
            ],
            "Resource": "arn:aws:lambda:us-east-1:*:function:fred-ml-processor"
        }
    ]
}
```

### Application Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `S3_BUCKET` | S3 bucket name | `fredmlv1` |
| `LAMBDA_FUNCTION` | Lambda function name | `fred-ml-processor` |
| `AWS_ACCESS_KEY_ID` | AWS access key | `AKIA...` |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | `...` |
| `AWS_DEFAULT_REGION` | AWS region | `us-east-1` |

## Step 5: Deploy and Test

### 1. Deploy

1. Click "Deploy" in Streamlit Cloud
2. Wait for the build to complete
3. Check the deployment logs for any errors

### 2. Test the Application

1. Open the provided Streamlit URL
2. Navigate to the "Analysis" page
3. Select indicators and run a test analysis
4. Check the "Reports" page for results

### 3. Monitor Logs

- Check Streamlit Cloud logs for frontend issues
- Monitor AWS CloudWatch logs for Lambda function issues
- Verify S3 bucket for generated reports

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: Module not found errors
**Solution**: Ensure all dependencies are in `requirements.txt`

#### 2. AWS Credentials

**Problem**: Access denied errors
**Solution**: Verify IAM permissions and credentials

#### 3. S3 Access

**Problem**: Cannot access S3 bucket
**Solution**: Check bucket name and IAM permissions

#### 4. Lambda Invocation

**Problem**: Lambda function not responding
**Solution**: Verify function name and permissions

### Debug Commands

```bash
# Test AWS credentials
aws sts get-caller-identity

# Test S3 access
aws s3 ls s3://fredmlv1/

# Test Lambda function
aws lambda invoke --function-name fred-ml-processor --payload '{}' response.json
```

## Production Considerations

### Security

1. **Use IAM Roles**: Instead of access keys when possible
2. **Rotate Credentials**: Regularly update AWS credentials
3. **Monitor Access**: Use CloudTrail to monitor API calls

### Performance

1. **Caching**: Use Streamlit caching for expensive operations
2. **Connection Pooling**: Reuse AWS connections
3. **Error Handling**: Implement proper error handling

### Monitoring

1. **Streamlit Cloud Metrics**: Monitor app performance
2. **AWS CloudWatch**: Monitor Lambda and S3 usage
3. **Custom Alerts**: Set up alerts for failures

## Custom Domain (Optional)

If you want to use a custom domain:

1. **Domain Setup**: Configure your domain in Streamlit Cloud
2. **SSL Certificate**: Streamlit Cloud handles SSL automatically
3. **DNS Configuration**: Update your DNS records

## Cost Optimization

### Streamlit Cloud

- **Free Tier**: 1 app, limited usage
- **Team Plan**: Multiple apps, more resources
- **Enterprise**: Custom pricing

### AWS Costs

- **Lambda**: Pay per invocation
- **S3**: Pay per storage and requests
- **EventBridge**: Minimal cost for scheduling

## Support

### Streamlit Cloud Support

- **Documentation**: [docs.streamlit.io](https://docs.streamlit.io)
- **Community**: [discuss.streamlit.io](https://discuss.streamlit.io)
- **GitHub**: [github.com/streamlit/streamlit](https://github.com/streamlit/streamlit)

### AWS Support

- **Documentation**: [docs.aws.amazon.com](https://docs.aws.amazon.com)
- **Support Center**: [aws.amazon.com/support](https://aws.amazon.com/support)

---

**Next Steps**: After deployment, test the complete workflow and monitor for any issues. 