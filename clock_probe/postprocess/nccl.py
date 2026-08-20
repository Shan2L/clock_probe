"""Held-out NCCL happened-before checks on already-aligned Traces."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .chrome import (
    is_nccl_kernel,
    iter_parsed_events,
    microseconds_to_ns,
)

COLLECTIVE_ALIASES = {
    "allreduce": "allreduce",
    "all_reduce": "allreduce",
    "allreduce_coalesced": "allreduce",
    "allgather": "allgather",
    "all_gather": "allgather",
    "_allgather_base": "allgather",
    "all_gather_into_tensor_coalesced": "allgather",
    "allgather_into_tensor_coalesced": "allgather",
    "reducescatter": "reducescatter",
    "reduce_scatter": "reducescatter",
    "_reduce_scatter_base": "reducescatter",
    "reduce_scatter_tensor_coalesced": "reducescatter",
    "alltoall": "alltoall",
    "all_to_all": "alltoall",
    "alltoallv": "alltoall",
    "all_to_allv": "alltoall",
    "barrier": "barrier",
}
IMPLICIT_SYNC = {
    "allreduce",
    "allgather",
    "reducescatter",
    "alltoall",
    "barrier",
}
MAX_RECORDED_INVERSIONS = 100
SEQ_NUM_KEYS = ("Seq", "seq_num", "Sequence number")
EXTERNAL_ID_KEYS = ("External id", "External ID", "external_id")


@dataclass(frozen=True)
class NCCLKernel:  # pylint: disable=too-many-instance-attributes
    """One GPU NCCL/RCCL kernel on one rank."""

    rank: int
    ts_ns: int
    dur_ns: int
    name: str
    collective: str
    process_group: str
    group_size: int | None
    dtype: str | None
    in_nelems: int | None
    out_nelems: int | None
    stream: int | None
    correlation: int | None
    seq_num: int | None
    external_id: int | None
    fingerprint: str
    pid: int | None
    tid: int | None

    @property
    def end_ns(self) -> int:
        """Return the kernel end in nanoseconds."""
        return self.ts_ns + self.dur_ns


@dataclass(frozen=True)
class CommsMetadata:
    """CPU-side record_param_comms metadata linked to one GPU kernel."""

    seq_num: int | None
    collective: str | None
    process_group: str | None
    group_size: int | None
    in_nelems: int | None
    out_nelems: int | None
    dtype: str | None


@dataclass
class MatchCollectivesResult:
    """Cross-rank NCCL kernel pairing outcome."""

    matched: dict[str, list[NCCLKernel]]
    unmatched: int
    matching_mode: str | None
    ambiguous_groups: int
    matching_failed: bool
    failure_reason: str | None = None


@dataclass
class Inversion:  # pylint: disable=too-many-instance-attributes
    """One impossible end-before-start pair inside a matched collective."""

    collective_id: str
    collective: str
    process_group: str
    early_rank: int
    late_rank: int
    early_end_ns: int
    late_start_ns: int
    gap_us: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable inversion."""
        return asdict(self)


@dataclass
class NCCLCheckReport:  # pylint: disable=too-many-instance-attributes
    """Session-level NCCL validation result. Does not rewrite Traces."""

    status: str
    uncertainty_us: float
    ranks: list[int]
    kernel_count: int
    matched_collectives: int
    unmatched_events: int
    inversion_count: int
    max_gap_us: float
    matching_mode: str | None = None
    ambiguous_groups: int = 0
    inversions: list[Inversion] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable check report."""
        payload = asdict(self)
        payload["inversions"] = [
            inversion.to_dict() for inversion in self.inversions
        ]
        return payload


def normalize_collective(name: str | None) -> str:
    """Map Profiler collective names onto a small implicit-sync set."""
    if not name:
        return "unknown"
    key = str(name).strip().lower()
    return COLLECTIVE_ALIASES.get(key, key.replace(" ", "_"))


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _first_optional_int(args: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        if key in args and args[key] is not None:
            return int(args[key])
    return None


def build_payload_fingerprint(
    *,
    process_group: str,
    collective: str,
    group_size: int | None,
    in_nelems: int | None,
    out_nelems: int | None,
    dtype: str | None,
) -> str:
    """Stable payload key for cross-rank fingerprint matching."""
    return "|".join(
        (
            process_group,
            collective,
            "" if group_size is None else str(group_size),
            "" if in_nelems is None else str(in_nelems),
            "" if out_nelems is None else str(out_nelems),
            dtype or "",
        )
    )


def _is_record_param_comms(event: dict[str, Any]) -> bool:
    if event.get("name") == "record_param_comms":
        return True
    args = event.get("args") or {}
    return event.get("cat") == "cpu_op" and "Collective name" in args


def _comms_metadata_from_args(args: dict[str, Any]) -> CommsMetadata:
    return CommsMetadata(
        seq_num=_first_optional_int(args, SEQ_NUM_KEYS),
        collective=normalize_collective(args.get("Collective name")),
        process_group=str(args.get("Process Group Name") or "unknown"),
        group_size=_optional_int(args.get("Group size")),
        in_nelems=_optional_int(args.get("In msg nelems")),
        out_nelems=_optional_int(args.get("Out msg nelems")),
        dtype=None if args.get("dtype") is None else str(args["dtype"]),
    )


def _build_comms_indexes(
    path: Path,
) -> tuple[dict[int, CommsMetadata], dict[int, CommsMetadata]]:
    """Index CPU-side comms metadata by External id and correlation."""
    by_external: dict[int, CommsMetadata] = {}
    by_correlation: dict[int, CommsMetadata] = {}
    for event in iter_parsed_events(path):
        if not _is_record_param_comms(event):
            continue
        args = event.get("args") or {}
        metadata = _comms_metadata_from_args(args)
        external_id = _first_optional_int(args, EXTERNAL_ID_KEYS)
        correlation = _optional_int(args.get("correlation"))
        if external_id is not None:
            by_external[external_id] = metadata
        if correlation is not None:
            by_correlation[correlation] = metadata
    return by_external, by_correlation


def _lookup_comms_metadata(
    args: dict[str, Any],
    *,
    by_external: dict[int, CommsMetadata],
    by_correlation: dict[int, CommsMetadata],
) -> CommsMetadata | None:
    external_id = _first_optional_int(args, EXTERNAL_ID_KEYS)
    correlation = _optional_int(args.get("correlation"))
    if external_id is not None and external_id in by_external:
        return by_external[external_id]
    if correlation is not None and correlation in by_correlation:
        return by_correlation[correlation]
    return None


def _build_nccl_kernel(
    event: dict[str, Any],
    *,
    rank: int,
    comms: CommsMetadata | None,
) -> NCCLKernel:
    args = event.get("args") or {}
    collective = normalize_collective(args.get("Collective name"))
    if collective == "unknown":
        collective = normalize_collective(str(event.get("name", "")))
    if comms is not None and collective == "unknown" and comms.collective:
        collective = comms.collective

    process_group = str(args.get("Process Group Name") or "unknown")
    if process_group == "unknown" and comms is not None and comms.process_group:
        process_group = comms.process_group

    group_size = _optional_int(args.get("Group size"))
    if group_size is None and comms is not None:
        group_size = comms.group_size

    in_nelems = _optional_int(args.get("In msg nelems"))
    if in_nelems is None and comms is not None:
        in_nelems = comms.in_nelems

    out_nelems = _optional_int(args.get("Out msg nelems"))
    if out_nelems is None and comms is not None:
        out_nelems = comms.out_nelems

    dtype = None if args.get("dtype") is None else str(args["dtype"])
    if dtype is None and comms is not None:
        dtype = comms.dtype

    seq_num = _first_optional_int(args, SEQ_NUM_KEYS)
    if seq_num is None and comms is not None:
        seq_num = comms.seq_num

    external_id = _first_optional_int(args, EXTERNAL_ID_KEYS)

    return NCCLKernel(
        rank=rank,
        ts_ns=microseconds_to_ns(event["ts"]),
        dur_ns=microseconds_to_ns(event.get("dur", 0)),
        name=str(event.get("name", "")),
        collective=collective,
        process_group=process_group,
        group_size=group_size,
        dtype=dtype,
        in_nelems=in_nelems,
        out_nelems=out_nelems,
        stream=_optional_int(args.get("stream")),
        correlation=_optional_int(args.get("correlation")),
        seq_num=seq_num,
        external_id=external_id,
        fingerprint=build_payload_fingerprint(
            process_group=process_group,
            collective=collective,
            group_size=group_size,
            in_nelems=in_nelems,
            out_nelems=out_nelems,
            dtype=dtype,
        ),
        pid=_optional_int(event.get("pid")),
        tid=_optional_int(event.get("tid")),
    )


def extract_nccl_kernels(path: Path, rank: int) -> list[NCCLKernel]:
    """Load NCCL/RCCL GPU kernels from one pretty-printed Trace."""
    by_external, by_correlation = _build_comms_indexes(path)
    kernels: list[NCCLKernel] = []
    for event in iter_parsed_events(path):
        if not is_nccl_kernel(event):
            continue
        args = event.get("args") or {}
        comms = _lookup_comms_metadata(
            args,
            by_external=by_external,
            by_correlation=by_correlation,
        )
        kernels.append(_build_nccl_kernel(event, rank=rank, comms=comms))
    kernels.sort(key=lambda kernel: (kernel.process_group, kernel.ts_ns, kernel.rank))
    return kernels


def _collect_ranks(kernels: list[NCCLKernel]) -> list[int]:
    return sorted({kernel.rank for kernel in kernels})


def _validate_member_group(
    collective_id: str,
    members: list[NCCLKernel],
    ranks: list[int],
) -> str | None:
    """Return a failure reason when one matched group is invalid."""
    if not members:
        return f"{collective_id} has no members"
    ranks_present = {member.rank for member in members}
    if len(members) != len(ranks_present):
        return f"{collective_id} contains duplicate ranks"
    expected_size = next(
        (member.group_size for member in members if member.group_size is not None),
        None,
    )
    if expected_size is not None and len(ranks_present) != expected_size:
        return (
            f"{collective_id} has {len(ranks_present)} ranks but "
            f"group_size={expected_size}"
        )
    trace_ranks = set(ranks)
    if not ranks_present.issubset(trace_ranks):
        return f"{collective_id} references ranks outside the Trace set"
    return None


def _match_by_seq_num(kernels: list[NCCLKernel]) -> MatchCollectivesResult:
    ranks = _collect_ranks(kernels)
    if any(kernel.seq_num is None for kernel in kernels):
        return MatchCollectivesResult(
            matched={},
            unmatched=len(kernels),
            matching_mode=None,
            ambiguous_groups=0,
            matching_failed=True,
            failure_reason="Not every NCCL kernel has seq_num for seq_num matching",
        )

    per_rank_key: dict[tuple[str, int, int], list[NCCLKernel]] = defaultdict(list)
    for kernel in kernels:
        per_rank_key[(kernel.process_group, kernel.seq_num, kernel.rank)].append(
            kernel
        )
    ambiguous_groups = sum(1 for bucket in per_rank_key.values() if len(bucket) > 1)
    if ambiguous_groups:
        return MatchCollectivesResult(
            matched={},
            unmatched=len(kernels),
            matching_mode="seq_num",
            ambiguous_groups=ambiguous_groups,
            matching_failed=True,
            failure_reason=(
                "Duplicate seq_num on one rank within the same process group"
            ),
        )

    grouped: dict[tuple[str, int], list[NCCLKernel]] = defaultdict(list)
    for kernel in kernels:
        grouped[(kernel.process_group, kernel.seq_num)].append(kernel)

    matched: dict[str, list[NCCLKernel]] = {}
    for (process_group, seq_num), members in sorted(grouped.items()):
        collective_id = f"pg={process_group}:seq={seq_num}"
        failure = _validate_member_group(collective_id, members, ranks)
        if failure is not None:
            return MatchCollectivesResult(
                matched={},
                unmatched=len(kernels),
                matching_mode="seq_num",
                ambiguous_groups=1,
                matching_failed=True,
                failure_reason=failure,
            )
        matched[collective_id] = sorted(members, key=lambda item: item.rank)

    return MatchCollectivesResult(
        matched=matched,
        unmatched=0,
        matching_mode="seq_num",
        ambiguous_groups=0,
        matching_failed=False,
    )


def _match_by_fingerprint(kernels: list[NCCLKernel]) -> MatchCollectivesResult:
    ranks = _collect_ranks(kernels)
    buckets: dict[tuple[str, str, int], list[NCCLKernel]] = defaultdict(list)
    for kernel in kernels:
        buckets[(kernel.process_group, kernel.fingerprint, kernel.rank)].append(
            kernel
        )
    for bucket in buckets.values():
        bucket.sort(key=lambda item: (item.ts_ns, item.correlation or 0))

    pg_fingerprints = sorted(
        {(process_group, fingerprint) for process_group, fingerprint, _rank in buckets}
    )
    matched: dict[str, list[NCCLKernel]] = {}
    ambiguous_groups = 0
    unmatched = 0

    for process_group, fingerprint in pg_fingerprints:
        counts = {
            rank: len(buckets.get((process_group, fingerprint, rank), []))
            for rank in ranks
        }
        unique_counts = set(counts.values())
        if len(unique_counts) != 1:
            ambiguous_groups += 1
            unmatched += sum(counts.values())
            continue

        shared = next(iter(unique_counts))
        for index in range(shared):
            members = [
                buckets[(process_group, fingerprint, rank)][index]
                for rank in ranks
            ]
            collective_id = f"pg={process_group}:fp={fingerprint}:idx={index}"
            failure = _validate_member_group(collective_id, members, ranks)
            if failure is not None:
                return MatchCollectivesResult(
                    matched={},
                    unmatched=len(kernels),
                    matching_mode="fingerprint",
                    ambiguous_groups=ambiguous_groups + 1,
                    matching_failed=True,
                    failure_reason=failure,
                )
            matched[collective_id] = members

    if ambiguous_groups:
        return MatchCollectivesResult(
            matched=matched,
            unmatched=unmatched,
            matching_mode="fingerprint",
            ambiguous_groups=ambiguous_groups,
            matching_failed=True,
            failure_reason=(
                "Fingerprint buckets disagree on per-rank kernel counts "
                "within one process group"
            ),
        )

    return MatchCollectivesResult(
        matched=matched,
        unmatched=unmatched,
        matching_mode="fingerprint",
        ambiguous_groups=0,
        matching_failed=False,
    )


def match_collectives(kernels: list[NCCLKernel]) -> MatchCollectivesResult:
    """Pair kernels across ranks using seq_num or payload fingerprint."""
    if not kernels:
        return MatchCollectivesResult(
            matched={},
            unmatched=0,
            matching_mode=None,
            ambiguous_groups=0,
            matching_failed=False,
        )

    if all(kernel.seq_num is not None for kernel in kernels):
        return _match_by_seq_num(kernels)

    if any(kernel.seq_num is not None for kernel in kernels):
        return MatchCollectivesResult(
            matched={},
            unmatched=len(kernels),
            matching_mode=None,
            ambiguous_groups=0,
            matching_failed=True,
            failure_reason=(
                "Mixed seq_num coverage across ranks; refusing order-based fallback"
            ),
        )

    return _match_by_fingerprint(kernels)


def _inversions_for_group(
    collective_id: str,
    members: list[NCCLKernel],
) -> list[Inversion]:
    if not any(
        member.collective in IMPLICIT_SYNC or member.collective == "unknown"
        for member in members
    ):
        return []
    inversions: list[Inversion] = []
    for early in members:
        if early.collective not in IMPLICIT_SYNC and early.collective != "unknown":
            continue
        for late in members:
            if early.rank == late.rank:
                continue
            if early.end_ns >= late.ts_ns:
                continue
            gap_us = (late.ts_ns - early.end_ns) / 1_000.0
            inversions.append(
                Inversion(
                    collective_id=collective_id,
                    collective=early.collective,
                    process_group=early.process_group,
                    early_rank=early.rank,
                    late_rank=late.rank,
                    early_end_ns=early.end_ns,
                    late_start_ns=late.ts_ns,
                    gap_us=gap_us,
                )
            )
    return inversions


def check_nccl_traces(
    traces: dict[int, Path],
    *,
    uncertainty_us: float,
) -> NCCLCheckReport:
    """Validate aligned Traces. This check is not used to fit clock models."""
    if len(traces) < 2:
        raise ValueError("NCCL checks need at least two rank Traces")
    if uncertainty_us < 0:
        raise ValueError("uncertainty_us must be non-negative")

    kernels: list[NCCLKernel] = []
    notes: list[str] = []
    for rank, path in sorted(traces.items()):
        if not path.is_file():
            raise FileNotFoundError(path)
        kernels.extend(extract_nccl_kernels(path, rank))

    match_result = match_collectives(kernels)
    if match_result.matching_mode is not None:
        notes.append(f"Matched collectives using {match_result.matching_mode}")
    if match_result.matching_failed:
        notes.append(match_result.failure_reason or "Collective matching failed")
    if match_result.ambiguous_groups:
        notes.append(
            f"{match_result.ambiguous_groups} collective groups were ambiguous"
        )
    if match_result.unmatched:
        notes.append(
            f"{match_result.unmatched} NCCL kernels were not matched across ranks"
        )

    inversions: list[Inversion] = []
    if not match_result.matching_failed:
        for collective_id, members in match_result.matched.items():
            inversions.extend(_inversions_for_group(collective_id, members))
    inversions.sort(key=lambda item: item.gap_us, reverse=True)
    max_gap_us = inversions[0].gap_us if inversions else 0.0

    if not kernels:
        notes.append("No NCCL/RCCL GPU kernels were found")
        status = "FAIL"
    elif match_result.matching_failed:
        status = "FAIL"
        notes.append(
            "Collective matching failed; do not run CLC or use aligned Traces "
            "as the primary timeline"
        )
    elif inversions and max_gap_us > uncertainty_us:
        status = "FAIL"
        notes.append(
            "Inversions exceed uncertainty_us; do not run CLC or use "
            "aligned Traces as the primary timeline"
        )
    elif inversions:
        status = "WARNING"
        notes.append(
            "Inversions are within uncertainty_us; CLC may repair them"
        )
    else:
        status = "PASS"
        notes.append("No implicit-sync happened-before inversions")

    return NCCLCheckReport(
        status=status,
        uncertainty_us=uncertainty_us,
        ranks=sorted(traces),
        kernel_count=len(kernels),
        matched_collectives=len(match_result.matched),
        unmatched_events=match_result.unmatched,
        inversion_count=len(inversions),
        max_gap_us=max_gap_us,
        matching_mode=match_result.matching_mode,
        ambiguous_groups=match_result.ambiguous_groups,
        inversions=inversions[:MAX_RECORDED_INVERSIONS],
        notes=notes,
    )
