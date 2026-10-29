"""Single-threaded worker for an exact trusted vertical mock fixture only."""
import json
import sys

from benchmarks.s5_vertical_validation import VerticalValidator


def main():
    request=json.load(sys.stdin)
    if not isinstance(request, dict) or set(request) != {'source'}:
        return 2
    result=VerticalValidator(max_attempts=1).validate(request['source'], arm='A')
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
