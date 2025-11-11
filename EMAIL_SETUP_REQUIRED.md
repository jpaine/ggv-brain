# Email Setup Required

## Issue Found

The email functionality is **not working** because the domain `goldengate.vc` is not verified in Resend.

### Error Message
```
403: The goldengate.vc domain is not verified. Please, add and verify your domain on https://resend.com/domains
```

## Solutions

### Option 1: Verify Domain in Resend (Recommended)
1. Go to https://resend.com/domains
2. Add `goldengate.vc` as a domain
3. Add the required DNS records (SPF, DKIM, DMARC) to your domain's DNS settings
4. Wait for verification (usually takes a few minutes)

### Option 2: Use Resend's Test Domain (Quick Test)
For testing purposes, you can use Resend's test domain:
- Change `FROM_EMAIL` to use `onboarding@resend.dev` (Resend's test domain)
- This only works for sending to verified email addresses in your Resend account

### Option 3: Use a Different Email Service
- Switch to SendGrid, Mailgun, or AWS SES
- Update the `EmailAlertService` class in `workflow_1_crunchbase_daily_monitor.py`

## Current Configuration

```python
FROM_EMAIL = os.getenv('FROM_EMAIL', 'ggv-brain@goldengate.vc')
RECIPIENT_EMAIL = os.getenv('RECIPIENT_EMAIL', 'jeff@goldengate.vc')
RESEND_API_KEY = os.getenv('RESEND_API_KEY', 're_4D7gm8JB_4An2NRYKxWT1VuFXabqPnhD3')
```

## Test Script

Run `./test_email_curl.sh` to test email functionality after domain verification.

## Next Steps

1. **Immediate**: Verify `goldengate.vc` domain in Resend
2. **Alternative**: Use Resend's test domain for quick testing
3. **Long-term**: Consider using a verified domain for production

## Database Status

The workflow is running and scoring companies:
- 10 companies scored in the database
- All have `emailed = false` because emails are failing
- Workflow is configured to email for every score (not just high scores)

## Workflow Status

- ✅ Crunchbase scanning: Working (10 companies found)
- ✅ Feature extraction: Working
- ✅ Model scoring: Working
- ✅ Database saving: Working
- ❌ Email sending: **FAILING** (domain not verified)

