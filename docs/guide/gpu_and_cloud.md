# GPU and Qoro cloud

## GPU

`device="GPU"` runs on Maestro's GPU backend (statevector, MPS, tensor network
and Pauli propagation). To run it on your own machine you need GPU access: see
[Get GPU access](https://qoroquantum.github.io/maestro/getting-started/gpu-access.html).
Maestro reads the licence key from `MAESTRO_LICENSE_KEY` itself. If no GPU is
available or the licence isn't accepted, qiskit-maestro raises
`MaestroGpuError`; it never falls back to the CPU silently.

No GPU of your own? `device="GPU"` together with `service=` runs on the Qoro
GPU cluster instead: see [Qoro cloud](#qoro-cloud).

## Qoro cloud

Pass a `QoroService` from [Divi](https://github.com/QoroQuantum/divi) as
`service=` and the backend, sampler or estimator runs on Maestro in the Qoro
cloud instead of on your machine. Sign up at
[dash.qoroquantum.net](https://dash.qoroquantum.net) for an API key, then:

```bash
pip install "qiskit-maestro[cloud]"
export QORO_API_KEY=...   # or put it in a .env file
```

The `cloud` extra installs Divi 0.16 or later, which needs Python 3.11 or later
and Qiskit 2.6 or later. [The Qoro cloud example](../examples/qoro_cloud.md)
runs the backend, the sampler and the estimator on the cloud.

### Cloud jobs

Each `run()` sends its circuits as one cloud job and returns at once.
`job_id()` is the cloud's job ID, `status()` asks the cloud, `result()` waits
(the backend's job takes a `timeout`, after which the job keeps running), and
`cancel()` cancels it. The simulator options travel with the job, and results
come back in the same form and bit order as local runs. The estimator sends
expectation-value jobs, so its values are exact, as they are locally.
`device="GPU"` runs the job on the Qoro GPU cluster; otherwise it runs on the
service's cluster. A parameter sweep goes as one circuit template with a table
of parameter values, which the cloud binds, when every circuit in the job takes
the same shots (or, for the estimator, the same observables); otherwise each
parameter set goes as its own circuit. A job that fails, times out on the cloud
or runs out of credits raises `QoroCloudError`, with its `job_id`.

### Not on the cloud yet

Not available on the cloud yet, and raising a clear error instead: noise
models, classical control (`if_test`, `switch`), Aer's save instructions,
resets in estimator circuits, and `run_evolution`.
