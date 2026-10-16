"""
Export Roboflow annotations using SDK methods
Try to access raw project data without generating a version
"""

from roboflow import Roboflow
import os
from pathlib import Path

# Configuration
API_KEY = "ukXcCk2tVYVS5jYLHEqf"
WORKSPACE = "kaycee-hn2a6"
PROJECT = "fireextinguisher-z5atr-6ktlv"

def explore_roboflow_api():
    """Explore what's available in the Roboflow SDK"""
    print("=" * 60)
    print("Exploring Roboflow SDK Methods")
    print("=" * 60)

    # Initialize Roboflow
    print("\nInitializing Roboflow...")
    rf = Roboflow(api_key=API_KEY)
    print(f"✓ Roboflow initialized")

    # Get workspace
    print(f"\nAccessing workspace: {WORKSPACE}")
    workspace = rf.workspace(WORKSPACE)
    print(f"✓ Workspace: {workspace}")

    # Get project
    print(f"\nAccessing project: {PROJECT}")
    project = workspace.project(PROJECT)
    print(f"✓ Project: {project}")

    # Check project attributes
    print("\n--- Project Attributes ---")
    for attr in dir(project):
        if not attr.startswith('_'):
            print(f"  - {attr}")

    # Try to get project info
    print("\n--- Trying to Get Project Info ---")
    try:
        # Check if there's a method to list images
        if hasattr(project, 'list_images'):
            images = project.list_images()
            print(f"✓ Found {len(images)} images")
        else:
            print("⚠️  No list_images method")

        # Check for versions
        if hasattr(project, 'versions'):
            print(f"Project versions: {project.versions()}")

        # Check for raw data access
        if hasattr(project, 'get_annotations'):
            print("✓ Has get_annotations method")

    except Exception as e:
        print(f"Error exploring: {e}")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    explore_roboflow_api()
