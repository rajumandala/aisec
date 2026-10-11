import os
import subprocess
import sys

# No model calls here: start the assistant against each vendor version and see whether it agrees to run.

def start_with(server: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "MARKET_DATA_SERVER": server}
    return subprocess.run([sys.executable, "-c", "import aisec.f06_mcp.p02_pinned_tools"],
                          env=env, capture_output=True, text=True)

def test_reviewed_vendor_starts():
    return start_with("market_data_server.py").returncode == 0

def test_changed_vendor_is_refused():
    started = start_with("market_data_server_v2.py")
    return started.returncode != 0 and "the vendor's tools changed since review: get_quote" in started.stderr

tests = [
    ("the reviewed vendor starts", test_reviewed_vendor_starts),
    ("[security] a changed vendor is refused", test_changed_vendor_is_refused),
]

suite_ok = True
for name, test in tests:
    ok = test()
    suite_ok = suite_ok and ok
    print(f"{'PASS' if ok else 'FAIL'}  {name}")

print("SUITE PASSED" if suite_ok else "SUITE FAILED")
sys.exit(0 if suite_ok else 1)
