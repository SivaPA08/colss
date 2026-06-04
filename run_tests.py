import sys
import traceback

import test.test_colss as test_colss

def main():
    tests = [getattr(test_colss, name) for name in dir(test_colss) if name.startswith("test_") and callable(getattr(test_colss, name))]
    print(f"Found {len(tests)} tests.")
    passed = 0
    failed = 0
    for test in tests:
        print(f"Running {test.__name__}... ", end="")
        try:
            test()
            print("PASSED")
            passed += 1
        except Exception as e:
            print("FAILED")
            traceback.print_exc()
            failed += 1
    print(f"\nResults: {passed} passed, {failed} failed.")
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
