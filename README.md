# Clock Probe

Clock Probe calibrates cross-node clocks for offline PyTorch/Kineto Trace
alignment. It never steps `CLOCK_REALTIME`.

The package has three public capabilities:

- one hardware-first Ray calibration entry point with software fallback;
- low-level PHC/PTP model construction for external orchestration;
- one offline Trace processing pipeline.

## Guarantees

- Hardware sessions fail closed above 2 μs total uncertainty.
- Candidate models are selected on tuning time and checked on unseen time.
- Trace events outside model coverage are rejected; no extrapolation.
- Raw, aligned, and optional CLC files are always distinct.
- NCCL/RCCL matching failure makes `primary_timeline=none`.
- CLC is forbidden after an NCCL FAIL.
- `phc2sys` is never run and the host wall clock is never stepped.

## Install

```bash
python -m pip install .
```

Ray orchestration is optional:

```bash
python -m pip install '.[ray]'
```

The core package has no third-party runtime dependency.

## Python API

### Hardware-first cluster calibration

```python
from clock_probe import ProbeConfig, probe

run = probe.start(
    ProbeConfig(
        ray_address="auto",
        mode="auto",
        hardware_interface="enp196s0f1np1",
        hardware_phc_device="/dev/ptp3",
        hardware_ptp_logs={
            "cse-ai-6": "/tmp/ptp4l-gm.log",
            "cse-ai-9": "/tmp/ptp4l-slave.log",
        },
        reference_host="10.67.93.244",
        working_dir=None,  # package is already installed on every node
    )
)

# Run the workload/profile while sampling continues.
print(run.status())
session = run.stop("clock-session.json")
```

Every Ray node must pass PHC access, NIC mapping, configured ptp4l log, and
current lock checks before sampling starts. If preflight fails in `auto` mode,
the run records the reasons and starts software timestamp calibration instead.
Once hardware sampling starts, quality failure is fatal and never triggers a
silent fallback. Use `mode="hardware"` or `mode="software"` to require one path.

Use `probe.run(config)` for a fixed-duration synchronous calibration.

### Hardware calibration

```python
from clock_probe import HardwareModelConfig, hardware

samples = hardware.sample(
    interface="enp196s0f1np1",
    phc_device="/dev/ptp3",
    duration_seconds=180,
    interval_ms=50,
    output="node.phc.jsonl",
)

model = hardware.fit(
    samples,
    role="slave",
    ptp_log="ptp4l-slave.log",
    source={"hostname": "cse-ai-9"},
    config=HardwareModelConfig(),
)

session = hardware.session([gm_model, model])
```

The low-level `hardware` API remains available when another application owns
cross-host orchestration. Containers need `/dev/ptpN` plus the required
network/time capabilities. Do not run `phc2sys`.

### Trace processing

```python
from pathlib import Path
from clock_probe import TraceInput, process_traces

manifest = process_traces(
    {
        0: TraceInput(Path("rank0.json"), "cse-ai-9", boot_id="..."),
        8: TraceInput(Path("rank8.json"), "cse-ai-6", boot_id="..."),
    },
    "clock-session.json",
    "clock-output",
    apply_clc_on_warning=True,
)

assert manifest.status == "PASS"
print(manifest.primary_timeline)
```

The pipeline performs batch alignment, NCCL held-out validation, optional
restricted CLC, and writes `manifest.json` plus the sidecar reports.

## Automatic model selection

Software candidates vary low-RTT anchor windows, sample counts, RTT slack, and
interpolation/affine models. Hardware candidates vary interpolation stride and
piecewise-affine duration.

The first time partition ranks candidates. The unseen partition validates the
winner. The selected parameters are then rebuilt on all samples. The objective
is worst total uncertainty, not training residual. Candidate summaries are
stored in `model_selection`; NCCL is never used for tuning.

For hardware, PTP uncertainty is deducted from the fixed 2 μs limit before PHC
bridge candidates are evaluated.

## Session compatibility

```python
from clock_probe import load_session, session_uncertainty_us

session = load_session("clock-session.json")
uncertainty_us = session_uncertainty_us(session)
```

Current hardware/software sessions and historical sessions with a missing
`clock_source` are accepted. Missing `clock_source` means `udp_software`.
Legacy worker sessions without a REALTIME bridge can be loaded for inspection,
but cannot safely align Kineto traces and are rejected at processing time.

## CLI

There is one thin command:

```text
clock-probe start --config probe.json
clock-probe status
clock-probe stop --output clock-session.json
clock-probe run --config probe.json
clock-probe process --spec process.json
clock-probe inspect ...
```

Configuration-heavy commands accept JSON files that map directly to the Python
configuration dataclasses. The Python API is the stable integration surface.

PTP examples are installed under `share/clock-probe/ptp/`.

## Internal layout

```text
sampling/      Linux timestamp and PHC acquisition
calibration/   fitting, auto-selection, bridge, PTP policy
execution/     Ray cluster execution and network topology
postprocess/   Trace alignment, NCCL, CLC, report pipeline
```
