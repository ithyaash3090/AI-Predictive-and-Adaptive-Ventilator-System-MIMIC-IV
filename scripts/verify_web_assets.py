"""
Web assets and FastAPI server entry point verification script for MIMIC-IV Ventilator System.
Verifies the existence of HTML templates, CSS/JS static assets, and FastAPI app initialization.
"""

import sys
from pathlib import Path

# Add project root to sys.path to enable direct importing of src modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REQUIRED_TEMPLATES = [
    Path("src/web/templates/dashboard.html"),
]

REQUIRED_STATIC_ASSETS = [
    Path("src/web/static/styles.css"),
    Path("src/web/static/dashboard.js"),
]


def format_file_size(size_bytes: int) -> str:
    """Formats file size into human-readable KB or MB."""
    size_kb = size_bytes / 1024
    if size_kb >= 1024:
        return f"{size_kb / 1024:.1f} MB ({size_kb:.1f} KB)"
    return f"{size_kb:.1f} KB"


def check_templates(root_dir: Path) -> bool:
    """Checks presence of required web template files."""
    print("[*] Checking HTML template files...")
    all_found = True
    for rel_path in REQUIRED_TEMPLATES:
        file_path = root_dir / rel_path
        if file_path.is_file():
            size_str = format_file_size(file_path.stat().st_size)
            print(f"  [PASS] {rel_path.as_posix()} found (size: {size_str})")
        else:
            print(f"  [FAIL] Missing required template: {rel_path.as_posix()}")
            all_found = False
    return all_found


def check_static_assets(root_dir: Path) -> bool:
    """Checks presence of required static CSS/JS files."""
    print("[*] Checking static frontend assets...")
    all_found = True
    for rel_path in REQUIRED_STATIC_ASSETS:
        file_path = root_dir / rel_path
        if file_path.is_file():
            size_str = format_file_size(file_path.stat().st_size)
            print(f"  [PASS] {rel_path.as_posix()} found (size: {size_str})")
        else:
            print(f"  [FAIL] Missing required static asset: {rel_path.as_posix()}")
            all_found = False
    return all_found


def check_fastapi_app() -> bool:
    """Tests importing the FastAPI server instance and verifying title."""
    print("[*] Checking FastAPI application initialization...")
    try:
        from src.web.app import app

        if hasattr(app, "title") and app.title:
            print(f"  [PASS] FastAPI app initialized successfully (title: '{app.title}')")
            return True
        else:
            print("  [FAIL] FastAPI app object loaded but title attribute is missing or empty")
            return False
    except Exception as e:
        print(f"  [FAIL] Failed to import or initialize FastAPI app: {e}")
        return False


def main():
    print("=" * 65)
    print("MIMIC-IV Ventilator System Web Assets & Server Verification")
    print("=" * 65)

    templates_ok = check_templates(PROJECT_ROOT)
    print("-" * 65)
    static_ok = check_static_assets(PROJECT_ROOT)
    print("-" * 65)
    app_ok = check_fastapi_app()

    print("=" * 65)
    if templates_ok and static_ok and app_ok:
        print("[SUCCESS] All web assets and FastAPI server initialized successfully!")
        sys.exit(0)
    else:
        print("[ERROR] Verification failed. Please check the missing files or errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
