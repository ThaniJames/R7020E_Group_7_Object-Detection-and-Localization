"""
Load environment variables from .env file
Can be used locally or in Google Colab
"""

import os
from pathlib import Path

def load_env_file(env_path='.env'):
    """
    Load environment variables from .env file

    Args:
        env_path: Path to .env file (default: '.env' in current directory)

    Returns:
        dict: Dictionary of environment variables
    """
    env_vars = {}

    # Check if .env file exists
    if not os.path.exists(env_path):
        print(f"⚠️  Warning: {env_path} not found")
        print("Using default values or environment variables")
        return env_vars

    # Read .env file
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            # Parse KEY=VALUE
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()

                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]

                # Set environment variable
                os.environ[key] = value
                env_vars[key] = value

    print(f"✓ Loaded {len(env_vars)} environment variables from {env_path}")
    return env_vars


def get_roboflow_config():
    """
    Get Roboflow configuration from environment variables

    Returns:
        dict: Roboflow configuration
    """
    config = {
        'api_key': os.getenv('ROBOFLOW_API_KEY', ''),
        'workspace': os.getenv('ROBOFLOW_WORKSPACE', 'kaycee-hn2a6'),
        'project': os.getenv('ROBOFLOW_PROJECT', 'fireextinguisher-z5atr-6ktlv'),
        'version': int(os.getenv('ROBOFLOW_VERSION', '1')),
    }

    if not config['api_key'] or config['api_key'] == 'your_api_key_here':
        print("⚠️  WARNING: ROBOFLOW_API_KEY not set or using placeholder!")
        print("Get your API key from: https://app.roboflow.com/settings/api")
        print("Then set it in .env file or as environment variable")

    return config


def get_training_config():
    """
    Get training configuration from environment variables

    Returns:
        dict: Training configuration
    """
    config = {
        'model_size': os.getenv('MODEL_SIZE', 'n'),
        'epochs': int(os.getenv('EPOCHS', '100')),
        'batch_size': int(os.getenv('BATCH_SIZE', '16')),
        'img_size': int(os.getenv('IMG_SIZE', '640')),
    }

    return config


if __name__ == '__main__':
    # Test loading
    print("=== Loading Environment Variables ===\n")

    env_vars = load_env_file()

    print("\n=== Roboflow Configuration ===")
    rf_config = get_roboflow_config()
    for key, value in rf_config.items():
        if key == 'api_key' and value:
            print(f"  {key}: {'*' * 20} (hidden)")
        else:
            print(f"  {key}: {value}")

    print("\n=== Training Configuration ===")
    train_config = get_training_config()
    for key, value in train_config.items():
        print(f"  {key}: {value}")
