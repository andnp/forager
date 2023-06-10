window.BENCHMARK_DATA = {
  "lastUpdate": 1686429343216,
  "repoUrl": "https://github.com/andnp/forager",
  "entries": {
    "Python Benchmark with pytest-benchmark": [
      {
        "commit": {
          "author": {
            "email": "andnpatterson@gmail.com",
            "name": "andy",
            "username": "andnp"
          },
          "committer": {
            "email": "andnpatterson@gmail.com",
            "name": "andy",
            "username": "andnp"
          },
          "distinct": true,
          "id": "70027c871e50464f1b27fa1d16f3588b0cbfb8a6",
          "message": "ci: have pdm perform the benchmark",
          "timestamp": "2023-06-10T14:34:38-06:00",
          "tree_id": "da563ef8a9e44e6cc0ac24ba202f77cbe1df47d6",
          "url": "https://github.com/andnp/forager/commit/70027c871e50464f1b27fa1d16f3588b0cbfb8a6"
        },
        "date": 1686429341830,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_Env.py::test_benchmark_vision",
            "value": 98610.10197962994,
            "unit": "iter/sec",
            "range": "stddev: 5.604437789614004e-7",
            "extra": "mean: 10.140948847275016 usec\nrounds: 15268"
          },
          {
            "name": "tests/test_Env.py::test_benchmark_creation",
            "value": 28.954030606344322,
            "unit": "iter/sec",
            "range": "stddev: 0.00283975071668686",
            "extra": "mean: 34.53750580000019 msec\nrounds: 5"
          },
          {
            "name": "tests/test_Env.py::test_benchmark_small_env",
            "value": 514.2911216891703,
            "unit": "iter/sec",
            "range": "stddev: 0.000012989611228458542",
            "extra": "mean: 1.944424000001277 msec\nrounds: 5"
          },
          {
            "name": "tests/test_Env.py::test_benchmark_big_env",
            "value": 45.02386663495118,
            "unit": "iter/sec",
            "range": "stddev: 0.00003433334113757595",
            "extra": "mean: 22.210442477273126 msec\nrounds: 44"
          },
          {
            "name": "tests/test_Env.py::test_benchmark_small_env_color",
            "value": 505.9791044796607,
            "unit": "iter/sec",
            "range": "stddev: 0.000015548653637977756",
            "extra": "mean: 1.976366200000257 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "andnpatterson@gmail.com",
            "name": "andy",
            "username": "andnp"
          },
          "committer": {
            "email": "andnpatterson@gmail.com",
            "name": "andy",
            "username": "andnp"
          },
          "distinct": true,
          "id": "70027c871e50464f1b27fa1d16f3588b0cbfb8a6",
          "message": "ci: have pdm perform the benchmark",
          "timestamp": "2023-06-10T14:34:38-06:00",
          "tree_id": "da563ef8a9e44e6cc0ac24ba202f77cbe1df47d6",
          "url": "https://github.com/andnp/forager/commit/70027c871e50464f1b27fa1d16f3588b0cbfb8a6"
        },
        "date": 1686429341830,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_Env.py::test_benchmark_vision",
            "value": 98610.10197962994,
            "unit": "iter/sec",
            "range": "stddev: 5.604437789614004e-7",
            "extra": "mean: 10.140948847275016 usec\nrounds: 15268"
          },
          {
            "name": "tests/test_Env.py::test_benchmark_creation",
            "value": 28.954030606344322,
            "unit": "iter/sec",
            "range": "stddev: 0.00283975071668686",
            "extra": "mean: 34.53750580000019 msec\nrounds: 5"
          },
          {
            "name": "tests/test_Env.py::test_benchmark_small_env",
            "value": 514.2911216891703,
            "unit": "iter/sec",
            "range": "stddev: 0.000012989611228458542",
            "extra": "mean: 1.944424000001277 msec\nrounds: 5"
          },
          {
            "name": "tests/test_Env.py::test_benchmark_big_env",
            "value": 45.02386663495118,
            "unit": "iter/sec",
            "range": "stddev: 0.00003433334113757595",
            "extra": "mean: 22.210442477273126 msec\nrounds: 44"
          },
          {
            "name": "tests/test_Env.py::test_benchmark_small_env_color",
            "value": 505.9791044796607,
            "unit": "iter/sec",
            "range": "stddev: 0.000015548653637977756",
            "extra": "mean: 1.976366200000257 msec\nrounds: 5"
          }
        ]
      }
    ]
  }
}