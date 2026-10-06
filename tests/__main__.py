from tests.smoke_tests import TESTS
for _t in TESTS: _t(); print(f'ok: {_t.__name__}')
print(f'{len(TESTS)}/{len(TESTS)} passed')
