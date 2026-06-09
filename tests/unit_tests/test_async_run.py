from opto import trace
from opto.trainer.utils import async_run
from opto.trace.nodes import USED_NODES
import asyncio
import gc
import warnings

from opto.trace.bundle import disable_external_dependencies_check


# Test passing args and kwargs
def test_passing_args_and_kwargs():

    def test(a, b):
        print(a, b)
        return a + b

    output = async_run([test]*2, kwargs_list=[{'a': 1, 'b': 2}, {'a': 3, 'b': 4}],)
    assert output[0] == 3
    assert output[1] == 7

    output = async_run([test]*2, args_list=[(1, 2), (3, 4)])
    assert output[0] == 3
    assert output[1] == 7

# Test switching between bundles
def test_switching_bundles():
    import time
    @trace.bundle()
    def fun_slow(x):
        print('Running slow function', USED_NODES.get())
        time.sleep(1)
        print('Finshing slow function', USED_NODES.get())
        return x + 1

    @trace.bundle()
    def fun_fast(x):
        print('Running fast function', USED_NODES.get())
        time.sleep(0.1)
        print('Finshing fast function', USED_NODES.get())
        return x + 1

    def run_slow():
        fun_slow(0)

    def run_fast():
        i = trace.node(0)
        i.data
        fun_fast(0)

    st = time.time()
    async_run([run_slow, run_fast])
    ed = time.time()
    print("Time taken: ", ed - st)


def test_async_run_inside_running_loop_does_not_leak_coroutine_warning():
    async def _run_inside_loop():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            output = async_run([lambda: 1], max_workers=2)
            gc.collect()
        return output, caught

    output, caught = asyncio.run(_run_inside_loop())

    assert output == [1]
    assert not any("was never awaited" in str(item.message) for item in caught)
