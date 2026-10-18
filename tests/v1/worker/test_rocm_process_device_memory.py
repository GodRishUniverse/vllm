# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parsing of DRM fdinfo and KFD topology behind
RocmPlatform.get_process_device_memory_bytes(). The inputs
are text in the format the amdgpu driver prints, but importing
vllm.platforms.rocm needs a ROCm GPU."""

import pytest

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    # vllm.platforms.rocm queries the GPU at import time.
    pytest.skip("requires ROCm", allow_module_level=True)

from vllm.platforms import rocm  # noqa: E402

# Shortened /proc/self/fdinfo/<fd> of an amdgpu render node (Linux 7.0).
# Real output; pasid and the GTT/resident keys are not used by the parser.
FDINFO_NEW_KERNEL = """\
pos:\t0
flags:\t02100002
drm-driver:\tamdgpu
drm-client-id:\t122
drm-pdev:\t0000:e3:00.0
pasid:\t117
drm-total-gtt:\t2128 KiB
drm-total-vram:\t1052672 KiB
drm-resident-vram:\t1052672 KiB
drm-memory-vram:\t1052672 KiB
amd-requested-vram:\t1052672 KiB
"""
# Kernels before drm-total-*: only the amdgpu-specific key.
FDINFO_OLD_KERNEL = (
    "drm-driver:\tamdgpu\ndrm-client-id:\t7\namd-requested-vram:\t4096\n"
)
# MI300 partitions 1..N-1 print no memory keys at all.
FDINFO_NO_MEMORY = "drm-driver:\tamdgpu\ndrm-client-id:\t9\n"


def test_sum_fdinfo_vram_reads_the_vram_key():
    assert rocm._sum_fdinfo_vram([FDINFO_NEW_KERNEL]) == 1052672 << 10
    assert rocm._sum_fdinfo_vram([FDINFO_OLD_KERNEL]) == 4096


def test_sum_fdinfo_vram_without_memory_keys_is_none():
    assert rocm._sum_fdinfo_vram([FDINFO_NO_MEMORY]) is None
    assert rocm._sum_fdinfo_vram([]) is None


def test_sum_fdinfo_vram_counts_each_client_once():
    # A duplicated fd (same drm-client-id) is the same open file: count once.
    # A second open of the render node is a separate client: add it.
    texts = [FDINFO_NEW_KERNEL, FDINFO_NEW_KERNEL, FDINFO_OLD_KERNEL]
    assert rocm._sum_fdinfo_vram(texts) == (1052672 << 10) + 4096


def _write_node(root, name, **props):
    node = root / name
    node.mkdir()
    node.joinpath("properties").write_text(
        "".join(f"{k} {v}\n" for k, v in props.items())
    )


def test_drm_render_minor_matches_uuid_then_pci(tmp_path, monkeypatch):
    monkeypatch.setattr(rocm, "_KFD_TOPOLOGY_NODES", str(tmp_path))
    rocm._drm_render_minor.cache_clear()
    _write_node(tmp_path, "0", simd_count=0)  # CPU node, skipped
    # A GPU node. unique_id and location_id are independent properties;
    # location_id 0xE300 is PCI bus 0xE3, device 0, function 0.
    _write_node(
        tmp_path,
        "6",
        simd_count=64,
        drm_render_minor=132,
        location_id=0xE300,
        domain=0,
        unique_id=0x13A691B49A73A35C,
    )
    # torch writes the unique_id as ASCII hex characters in the UUID bytes.
    uuid = b"13a691b49a73a35c".hex()
    assert rocm._drm_render_minor(uuid, (0, 0x99, 0)) == 132
    assert rocm._drm_render_minor("not-hex", (0, 0xE3, 0)) == 132
    # Same bus and device, function 1. The lookup ignores the function (torch
    # does not report it), so two nodes now match (0, 0xE3, 0); with no UUID
    # match the GPU is ambiguous.
    _write_node(tmp_path, "7", simd_count=64, drm_render_minor=133, location_id=0xE301)
    rocm._drm_render_minor.cache_clear()
    assert rocm._drm_render_minor("not-hex", (0, 0xE3, 0)) is None
    rocm._drm_render_minor.cache_clear()
