import io
import sys
from contextlib import redirect_stdout
from unittest.mock import patch
from aisec.f04_agent_loop.p02_confirm import Order, check_order, confirm, place_order, PENDING_SELLS

# No model calls here: these check the controls themselves, so one run is enough and every test must pass.

def test_cannot_sell_the_same_shares_twice():
    PENDING_SELLS.clear()
    order = Order(stock="INFY", buy_or_sell="sell", quantity=40, order_type="limit", price=1480.0)
    first = check_order(order, "ravi")
    with redirect_stdout(io.StringIO()):            # hide the PLACED line
        place_order(order, "ravi")
    second = check_order(order, "ravi")
    return first == [] and second == ["ravi can sell only 0 INFY"]

def test_only_an_exact_yes_confirms():
    order = Order(stock="INFY", buy_or_sell="sell", quantity=40, order_type="limit", price=1480.0)
    expected = {"yes": True, "Yes ": True, "y": False, "ok": False, "sure": False, "": False}
    for answer, should_place in expected.items():
        with patch("builtins.input", return_value=answer), redirect_stdout(io.StringIO()):   # the investor types this answer
            if confirm(order) != should_place:
                return False
    return True

tests = [
    ("[security] cannot sell the same shares twice", test_cannot_sell_the_same_shares_twice),
    ("[security] only an exact yes confirms", test_only_an_exact_yes_confirms),
]

suite_ok = True
for name, test in tests:
    ok = test()
    suite_ok = suite_ok and ok
    print(f"{'PASS' if ok else 'FAIL'}  {name}")

print("SUITE PASSED" if suite_ok else "SUITE FAILED")
sys.exit(0 if suite_ok else 1)
