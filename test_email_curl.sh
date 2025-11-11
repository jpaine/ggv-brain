#!/bin/bash
# Test Email Functionality using curl
# This script tests the Resend API directly

RESEND_API_KEY="re_4D7gm8JB_4An2NRYKxWT1VuFXabqPnhD3"
FROM_EMAIL="ggv-brain@goldengate.vc"
TO_EMAIL="jeff@goldengate.vc"

echo "=========================================="
echo "Testing Email Functionality"
echo "=========================================="
echo "From: $FROM_EMAIL"
echo "To: $TO_EMAIL"
echo ""

# Test email payload
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
SUBJECT="🧪 GGV Brain Email Test - $TIMESTAMP"

HTML_CONTENT="<html><body style='font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;'><h2 style='color: #1a73e8;'>🧪 Email Functionality Test</h2><div style='background: #f5f5f5; padding: 20px; border-radius: 8px; margin: 20px 0;'><p>This is a test email from the GGV Brain workflow system.</p><p><strong>Status:</strong> Email API is working correctly ✅</p><p><strong>Timestamp:</strong> $TIMESTAMP</p></div><p style='color: #999; font-size: 12px;'>If you received this email, the Resend API integration is functioning properly.</p></body></html>"

# Create JSON payload
PAYLOAD=$(cat <<EOF
{
  "from": "$FROM_EMAIL",
  "to": ["$TO_EMAIL"],
  "subject": "$SUBJECT",
  "html": "$HTML_CONTENT"
}
EOF
)

echo "Sending test email..."
echo ""

# Send email via Resend API
RESPONSE=$(curl -s -w "\nHTTP_STATUS:%{http_code}" \
  -X POST "https://api.resend.com/emails" \
  -H "Authorization: Bearer $RESEND_API_KEY" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD")

# Extract HTTP status code
HTTP_STATUS=$(echo "$RESPONSE" | grep "HTTP_STATUS" | cut -d: -f2)
BODY=$(echo "$RESPONSE" | sed '/HTTP_STATUS/d')

echo "Response Status: $HTTP_STATUS"
echo "Response Body:"
echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
echo ""

if [ "$HTTP_STATUS" = "200" ]; then
    echo "✅ SUCCESS: Email sent successfully!"
    echo "Please check your inbox at: $TO_EMAIL"
    exit 0
else
    echo "❌ FAILED: Email send failed with status $HTTP_STATUS"
    exit 1
fi

