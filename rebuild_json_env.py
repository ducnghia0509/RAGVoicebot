import os
import json
import tempfile
import logging

logger = logging.getLogger(__name__)

def setup_gcp_credentials_from_env():
    """
    Setup Google Cloud credentials from environment variables or local file.
    
    Priority:
    1. If GOOGLE_APPLICATION_CREDENTIALS is already set, use it
    2. If client_secret.json exists locally, use it
    3. Otherwise, build from GCP_* environment variables
    """
    
    # Check if already set
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        logger.info(f"GOOGLE_APPLICATION_CREDENTIALS already set: {os.environ['GOOGLE_APPLICATION_CREDENTIALS']}")
        return
    
    # Check if local file exists
    local_file = "client_secret.json"
    if os.path.exists(local_file):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(local_file)
        logger.info(f"Using local credentials file: {local_file}")
        return
    
    # Build from environment variables with GCP_ prefix
    required_keys = [
        "GCP_TYPE",
        "GCP_PROJECT_ID",
        "GCP_PRIVATE_KEY_ID",
        "GCP_PRIVATE_KEY",
        "GCP_CLIENT_EMAIL",
        "GCP_CLIENT_ID",
        "GCP_AUTH_URI",
        "GCP_TOKEN_URI",
        "GCP_AUTH_PROVIDER_X509_CERT_URL",
        "GCP_CLIENT_X509_CERT_URL",
        "GCP_UNIVERSE_DOMAIN",
    ]

    missing = [k for k in required_keys if k not in os.environ]
    if missing:
        logger.warning(
            f"Missing GCP environment variables: {', '.join(missing)}. "
            f"Google Cloud services (ASR) will not be available. "
            f"Either provide a client_secret.json file or set these env vars."
        )
        # Set a flag to indicate GCP is not available
        os.environ["GCP_DISABLED"] = "true"
        return

    data = {
        "type": os.environ["GCP_TYPE"],
        "project_id": os.environ["GCP_PROJECT_ID"],
        "private_key_id": os.environ["GCP_PRIVATE_KEY_ID"],
        "private_key": os.environ["GCP_PRIVATE_KEY"].replace("\\n", "\n"),
        "client_email": os.environ["GCP_CLIENT_EMAIL"],
        "client_id": os.environ["GCP_CLIENT_ID"],
        "auth_uri": os.environ["GCP_AUTH_URI"],
        "token_uri": os.environ["GCP_TOKEN_URI"],
        "auth_provider_x509_cert_url": os.environ["GCP_AUTH_PROVIDER_X509_CERT_URL"],
        "client_x509_cert_url": os.environ["GCP_CLIENT_X509_CERT_URL"],
        "universe_domain": os.environ["GCP_UNIVERSE_DOMAIN"],
    }

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w")
    json.dump(data, tmp)
    tmp.close()

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name
    logger.info(f"Created temporary credentials file: {tmp.name}")