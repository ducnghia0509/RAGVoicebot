import os
import json
import tempfile

def setup_gcp_credentials_from_env():
    required_keys = [
        "type",
        "project_id",
        "private_key_id",
        "private_key",
        "client_email",
        "client_id",
        "auth_uri",
        "token_uri",
        "auth_provider_x509_cert_url",
        "client_x509_cert_url",
        "universe_domain",
    ]

    for k in required_keys:
        if k not in os.environ:
            raise RuntimeError(f"Missing env var: {k}")

    data = {
        "type": os.environ["type"],
        "project_id": os.environ["project_id"],
        "private_key_id": os.environ["private_key_id"],
        "private_key": os.environ["private_key"].replace("\\n", "\n"),
        "client_email": os.environ["client_email"],
        "client_id": os.environ["client_id"],
        "auth_uri": os.environ["auth_uri"],
        "token_uri": os.environ["token_uri"],
        "auth_provider_x509_cert_url": os.environ["auth_provider_x509_cert_url"],
        "client_x509_cert_url": os.environ["client_x509_cert_url"],
        "universe_domain": os.environ["universe_domain"],
    }

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w")
    json.dump(data, tmp)
    tmp.close()

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name
