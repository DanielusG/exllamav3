"""
Page-cache hygiene of the persistent disk tier (EXL3_DISK_CACHE_UNCACHE, DiskPageCache._uncache_file).

No GPU or model is required. Upstream's DiskPageCache runs, writer thread, reader pool and index included,
over a stand-in for the CPU page tier whose slots are host buffers of random bytes, as in
tests/test_tail_checkpoint.py (whose loader this file uses: without the compiled extension the sources under
test are loaded on their own, with the rest of the package stubbed).

What is checked: off by default, and then not one call more than upstream; with the switch on, a page file
and a checkpoint file are synced and dropped once complete and before the rename publishes them, a page
image and a checkpoint read back are dropped after the read, without a sync, and still verified; errors of
the sync and of the advice are swallowed; a platform without posix_fadvise (Windows) or without fdatasync;
tensor-parallel transfers left alone; and, on a file system where the kernel drops pages on advice (not
tmpfs: set EXL3_TEST_DISK_DIR to an existing directory on a real disk), that nothing of the files is left in
the page cache.
"""

import ctypes
import ctypes.util
import logging
import mmap
import os
import shutil
import sys
import tempfile
import types

import pytest
import torch

import test_tail_checkpoint as TC

pytestmark = pytest.mark.nogpu

DISK = TC.DISK
REC = TC.REC
PAGE = TC.PAGE
SLAB = 16 * 4096
ENV = "EXL3_DISK_CACHE_UNCACHE"


# --------------------------------------------------------------------------------------------
#   The tier
# --------------------------------------------------------------------------------------------

class Tier:
    """The CPU page tier as DiskPageCache sees it: one local segment of `slab_size` bytes, no
    tensor-parallel group (unless a test gives one), nothing in system RAM; each slot a host buffer
    of random bytes of its own, so a round trip and the checksum mean something"""

    def __init__(self, slab_size = SLAB, slots = 4, tp_groups = ()):
        g = torch.Generator().manual_seed(slab_size + slots)
        self.slab_size = slab_size
        self.segments = [(torch.zeros((1, slab_size), dtype = torch.uint8), 0, (slab_size,), torch.uint8)]
        self.tp_groups = list(tp_groups)
        self.max_slots = 64
        self.entries = {}
        self.slabs = [torch.randint(0, 256, (slab_size,), dtype = torch.uint8, generator = g) for _ in range(slots)]

    def __contains__(self, h):
        return False

    def __len__(self):
        return 0

    def slab_view(self, slot):
        return self.slabs[slot]


def open_disk(root, monkeypatch, on: bool, tier = None):
    if on:
        monkeypatch.setenv(ENV, "1")
    else:
        monkeypatch.delenv(ENV, raising = False)
    # Not `tier or Tier()`: the stand-in is empty, so it is falsy
    tier = tier if tier is not None else Tier()
    return DISK.DiskPageCache(str(root), [], tier, 1 << 30, fingerprint = "uncache-test",
                              min_state_distance = 0, min_free = 0)


def key(i: int) -> bytes:
    return f"uncache:{i}".encode()


def write_page(disk, i: int, slot: int | None = None):
    tokens = torch.full((PAGE,), i, dtype = torch.long)
    assert disk.write_page(key(i), key(i - 1) if i else None, tokens, i if slot is None else slot, [])


def make_stash(position: int, n: int = 3000):
    """A checkpoint stash in 1.6.2's paired shape (the target side, two layers), from the stash pool"""
    a = REC.host_pool.take((n,), torch.float32)
    b = REC.host_pool.take((2, n), torch.float16)
    a.copy_(torch.arange(n, dtype = torch.float32) + position)
    b.copy_(torch.arange(2 * n, dtype = torch.float16).view(2, n))
    size = a.numel() * a.element_size() + b.numel() * b.element_size()
    return {"position": position, "checkpoint_size": size, "target": {"layer.0": a, "layer.1": {"conv": b}}}


def same_stash(x, y) -> bool:
    if isinstance(x, torch.Tensor):
        return isinstance(y, torch.Tensor) and x.dtype == y.dtype and torch.equal(x, y)
    if isinstance(x, dict):
        return isinstance(y, dict) and x.keys() == y.keys() and all(same_stash(x[k], y[k]) for k in x)
    return x == y


# --------------------------------------------------------------------------------------------
#   The calls
# --------------------------------------------------------------------------------------------

class Calls:
    """Records posix_fadvise, fdatasync, fsync and replace (the real ones still run), each with the
    inode it was made on. At a sync it also records what the file holds then, through the same
    descriptor, and whether the published path exists yet. `fail` maps a name to an errno to raise
    instead of calling the real function."""

    def __init__(self, monkeypatch, fail = None):
        self.log = []
        self.fail = fail or {}
        for name in ("posix_fadvise", "fdatasync", "fsync"):
            real = getattr(os, name, None)
            if real is not None:
                monkeypatch.setattr(os, name, self._wrap(name, real))
        real_replace = os.replace

        def replace(src, dst, **kw):
            self.log.append(("replace", os.stat(src).st_ino, str(dst)))
            return real_replace(src, dst, **kw)
        monkeypatch.setattr(os, "replace", replace)

    def _wrap(self, name, real):
        def fn(fd, *args):
            st = os.fstat(fd)
            entry = {"name": name, "ino": st.st_ino, "args": args}
            if name in ("fdatasync", "fsync"):
                entry["size"] = st.st_size
                entry["bytes"] = os.pread(fd, st.st_size, 0)
            self.log.append((name, st.st_ino, entry))
            if name in self.fail:
                raise OSError(self.fail[name], os.strerror(self.fail[name]))
            return real(fd, *args)
        return fn

    def names(self, ino = None):
        return [e[0] for e in self.log if ino is None or e[1] == ino]

    def entries(self, name):
        return [e[2] for e in self.log if e[0] == name]


def hygiene_calls(calls):
    return [n for n in calls.names() if n != "replace"]


# --------------------------------------------------------------------------------------------
#   Tests
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("value, on", [(None, False), ("0", False), ("", False), ("1", True), ("yes", True)])
def test_the_switch_is_off_by_default_and_read_when_the_tier_is_made(tmp_path, monkeypatch, value, on):
    if value is None:
        monkeypatch.delenv(ENV, raising = False)
    else:
        monkeypatch.setenv(ENV, value)
    disk = DISK.DiskPageCache(str(tmp_path), [], Tier(), 1 << 30, fingerprint = "uncache-test",
                              min_state_distance = 0, min_free = 0)
    try:
        assert disk.uncache is on
        # Read once: changing the environment afterwards does not change the tier
        monkeypatch.setenv(ENV, "0" if on else "1")
        assert disk.uncache is on
    finally:
        disk.close()


def test_with_the_switch_off_no_call_is_made(tmp_path, monkeypatch):
    calls = Calls(monkeypatch)
    disk = open_disk(tmp_path, monkeypatch, on = False)
    try:
        write_page(disk, 0)
        stash = make_stash(PAGE)
        assert disk.write_state(key(0), stash)
        disk.flush()
        assert disk.has_page(key(0)) and key(0) in disk.states
        slab = torch.zeros(SLAB, dtype = torch.uint8)
        assert disk.read_page_into(key(0), slab) is not None
        assert torch.equal(slab, disk.cpu_tier.slabs[0])
        assert same_stash(disk.read_state(key(0)), stash)
        # Upstream's calls only: the two renames that publish the files
        assert calls.names() == ["replace", "replace"]
        assert disk.uncache_stats == {"writes": 0, "reads": 0, "sync_s": 0.0, "errors": 0}
    finally:
        disk.close()


def _published_after_sync_and_drop(calls, path):
    """The calls on the file now at `path`: a sync of the complete file, the advice on the whole of it, then
    the rename that published it, in that order and nothing else"""
    ino = os.stat(path).st_ino
    assert calls.names(ino) == ["fdatasync", "posix_fadvise", "replace"]
    sync = next(e for e in calls.entries("fdatasync") if e["ino"] == ino)
    with open(path, "rb") as f:
        final = f.read()
    assert sync["size"] == len(final) and sync["bytes"] == final, "synced before the file was complete"
    advice = next(e for e in calls.entries("posix_fadvise") if e["ino"] == ino)
    assert advice["args"] == (0, 0, os.POSIX_FADV_DONTNEED)
    rename = next(e for e in calls.log if e[0] == "replace" and e[1] == ino)
    assert rename[2] == path


def test_a_page_is_synced_and_dropped_once_complete_and_before_it_is_published(tmp_path, monkeypatch):
    calls = Calls(monkeypatch)
    disk = open_disk(tmp_path, monkeypatch, on = True)
    try:
        for i in range(3):
            write_page(disk, i)
        disk.flush()
        for i in range(3):
            path = disk._page_path(key(i))
            assert disk.has_page(key(i))
            _published_after_sync_and_drop(calls, path)
        assert hygiene_calls(calls) == ["fdatasync", "posix_fadvise"] * 3
        assert disk.uncache_stats["writes"] == 3 and disk.uncache_stats["errors"] == 0
        assert disk.metrics["page_writes"] == 3 and disk.metrics["write_errors"] == 0
    finally:
        disk.close()


def test_a_checkpoint_is_synced_and_dropped_once_complete_and_before_it_is_published(tmp_path, monkeypatch):
    calls = Calls(monkeypatch)
    disk = open_disk(tmp_path, monkeypatch, on = True)
    try:
        write_page(disk, 0)
        disk.flush()
        stash = make_stash(PAGE)
        assert disk.write_state(key(0), stash)
        disk.flush()
        assert key(0) in disk.states and not disk.pending_states
        _published_after_sync_and_drop(calls, disk._state_path(key(0)))
        assert hygiene_calls(calls) == ["fdatasync", "posix_fadvise"] * 2
        assert disk.uncache_stats["writes"] == 2
    finally:
        disk.close()


def test_a_page_and_a_checkpoint_read_back_are_dropped_without_a_sync(tmp_path, monkeypatch):
    disk = open_disk(tmp_path, monkeypatch, on = True)
    stash = make_stash(PAGE)
    try:
        write_page(disk, 0)
        assert disk.write_state(key(0), stash)
        disk.flush()
    finally:
        disk.close()

    # A new process: everything on disk
    calls = Calls(monkeypatch)
    disk = open_disk(tmp_path, monkeypatch, on = True)
    try:
        slab = torch.zeros(SLAB, dtype = torch.uint8)
        tokens = disk.read_page_into(key(0), slab)
        assert tokens is not None and torch.equal(tokens, torch.full((PAGE,), 0, dtype = torch.long))
        assert torch.equal(slab, disk.cpu_tier.slabs[0]), "page did not come back bit-exact"
        page_ino = os.stat(disk._page_path(key(0))).st_ino
        assert calls.names(page_ino) == ["posix_fadvise"]

        back = disk.read_state(key(0))
        assert same_stash(back, stash), "checkpoint did not come back bit-exact"
        state_ino = os.stat(disk._state_path(key(0))).st_ino
        assert calls.names(state_ino) == ["posix_fadvise"]
        assert all(e["args"] == (0, 0, os.POSIX_FADV_DONTNEED) for e in calls.entries("posix_fadvise"))
        assert "fdatasync" not in calls.names() and "fsync" not in calls.names()
        assert disk.uncache_stats["reads"] == 2 and disk.metrics["page_reads"] == 1 and disk.metrics["state_reads"] == 1
    finally:
        disk.close()


def test_a_damaged_page_still_fails_its_checksum(tmp_path, monkeypatch):
    disk = open_disk(tmp_path, monkeypatch, on = True)
    try:
        write_page(disk, 0)
        disk.flush()
        path = disk._page_path(key(0))
        with open(path, "r+b") as f:
            f.seek(os.path.getsize(path) - 100)
            b = f.read(1)
            f.seek(-1, 1)
            f.write(bytes([b[0] ^ 0xFF]))
        slab = torch.zeros(SLAB, dtype = torch.uint8)
        assert disk.read_page_into(key(0), slab) is None
        assert not disk.has_page(key(0)) and disk.metrics["page_read_failures"] == 1
    finally:
        disk.close()


@pytest.mark.parametrize("fail", [{"fdatasync": 5}, {"posix_fadvise": 22}, {"fdatasync": 28, "posix_fadvise": 22}])
def test_errors_of_the_sync_and_of_the_advice_are_swallowed(tmp_path, monkeypatch, caplog, fail):
    calls = Calls(monkeypatch, fail = fail)
    disk = open_disk(tmp_path, monkeypatch, on = True)
    stash = make_stash(PAGE)
    try:
        with caplog.at_level(logging.WARNING):
            write_page(disk, 0)
            write_page(disk, 1)
            assert disk.write_state(key(1), stash)
            disk.flush()
            # Published as without the switch, no write error, nothing disabled
            assert disk.has_page(key(0)) and disk.has_page(key(1)) and key(1) in disk.states
            assert disk.metrics["write_errors"] == 0 and not disk.write_disabled
            slab = torch.zeros(SLAB, dtype = torch.uint8)
            assert disk.read_page_into(key(1), slab) is not None and torch.equal(slab, disk.cpu_tier.slabs[1])
            assert same_stash(disk.read_state(key(1)), stash)
        failed_reads = 2 if "posix_fadvise" in fail else 0
        assert disk.uncache_stats["errors"] == 3 + failed_reads
        assert disk.uncache_stats["writes"] == 0 and disk.uncache_stats["reads"] == 2 - failed_reads
        warnings = [r for r in caplog.records if "page cache not dropped" in r.getMessage()]
        assert len(warnings) == 1, "a failure is logged once, then counted"
        # A failed sync is not followed by the advice
        if "fdatasync" in fail:
            assert "posix_fadvise" not in [n for n in calls.names() if n != "replace"][:3]
    finally:
        disk.close()


def test_without_posix_fadvise_nothing_is_done(tmp_path, monkeypatch, caplog):
    """Windows (and macOS) have neither posix_fadvise nor fdatasync"""
    monkeypatch.delattr(os, "posix_fadvise", raising = False)
    monkeypatch.delattr(os, "fdatasync", raising = False)
    calls = Calls(monkeypatch)
    with caplog.at_level(logging.WARNING):
        disk = open_disk(tmp_path, monkeypatch, on = True)
    try:
        assert any("no effect on this platform" in r.getMessage() for r in caplog.records)
        write_page(disk, 0)
        stash = make_stash(PAGE)
        assert disk.write_state(key(0), stash)
        disk.flush()
        slab = torch.zeros(SLAB, dtype = torch.uint8)
        assert disk.read_page_into(key(0), slab) is not None and torch.equal(slab, disk.cpu_tier.slabs[0])
        assert same_stash(disk.read_state(key(0)), stash)
        assert calls.names() == ["replace", "replace"]
        assert disk.uncache_stats == {"writes": 0, "reads": 0, "sync_s": 0.0, "errors": 0}
    finally:
        disk.close()


def test_without_fdatasync_fsync_takes_its_place(tmp_path, monkeypatch):
    if not hasattr(os, "posix_fadvise"):
        pytest.skip("no posix_fadvise on this platform")
    monkeypatch.delattr(os, "fdatasync", raising = False)
    calls = Calls(monkeypatch)
    disk = open_disk(tmp_path, monkeypatch, on = True)
    try:
        write_page(disk, 0)
        disk.flush()
        ino = os.stat(disk._page_path(key(0))).st_ino
        assert calls.names(ino) == ["fsync", "posix_fadvise", "replace"]
    finally:
        disk.close()


def test_tensor_parallel_transfers_are_left_alone(tmp_path, monkeypatch):
    """With a tensor-parallel group the ranks write and read their parts of the same file (np.memmap): the
    main process's part is written and read as upstream does. A group with no ranks runs the TP paths
    in this process (the markers are all there at once)"""
    model = types.SimpleNamespace(active_devices = [], loaded_tp = True,
                                  tp_worker_dispatch_wait_multi = lambda devices, fn, args, per = None: [],
                                  tp_dispatch_all = lambda fn, args: None)
    calls = Calls(monkeypatch)
    disk = open_disk(tmp_path, monkeypatch, on = True, tier = Tier(tp_groups = [(model, set())]))
    try:
        write_page(disk, 0)
        disk.flush()
        assert disk.has_page(key(0))
        slab = torch.zeros(SLAB, dtype = torch.uint8)
        assert disk.read_page_into(key(0), slab) is not None and torch.equal(slab, disk.cpu_tier.slabs[0])
        # A checkpoint whose side is tensor-parallel when it is read (written here as a local side)
        stash = make_stash(PAGE)
        disk.tp_groups = []
        assert disk.write_state(key(0), stash)
        disk.flush()
        n_written = len(hygiene_calls(calls))
        disk.recurrent_sides["target"] = (model, types.SimpleNamespace())
        back = disk.read_state(key(0))
        assert back is not None and "tp_handle" in back["target"]
        assert n_written == 2, "the local checkpoint write is synced and dropped"
        assert hygiene_calls(calls) == ["fdatasync", "posix_fadvise"], "a TP page or a TP side's read was touched"
    finally:
        disk.recurrent_sides.clear()
        disk.close()


# --------------------------------------------------------------------------------------------
#   Residency
# --------------------------------------------------------------------------------------------

def _libc():
    if not sys.platform.startswith("linux"):
        return None
    name = ctypes.util.find_library("c")
    if name is None:
        return None
    libc = ctypes.CDLL(name, use_errno = True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]
    return libc


def resident_pages(libc, path: str) -> int:
    """Pages of the file in the page cache (mincore over a shared mapping, which faults nothing in)"""
    size = os.path.getsize(path)
    if size == 0:
        return 0
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        if addr is None or addr == ctypes.c_void_p(-1).value:
            raise OSError(ctypes.get_errno(), "mmap")
        try:
            n = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE
            vec = (ctypes.c_ubyte * n)()
            if libc.mincore(ctypes.c_void_p(addr), size, vec) != 0:
                raise OSError(ctypes.get_errno(), "mincore")
            return sum(b & 1 for b in vec)
        finally:
            libc.munmap(ctypes.c_void_p(addr), size)
    finally:
        os.close(fd)


@pytest.fixture
def real_disk_dir(tmp_path):
    """A directory on a file system where the kernel drops a file's pages on advice: EXL3_TEST_DISK_DIR (an
    existing directory; a fresh subdirectory is made there and removed afterwards) or the test's tmp_path.
    Skipped where advice drops nothing (tmpfs: the page cache is the file's only copy)"""
    libc = _libc()
    if libc is None or not hasattr(os, "posix_fadvise"):
        pytest.skip("needs Linux, mincore and posix_fadvise")
    parent = os.environ.get("EXL3_TEST_DISK_DIR")
    d = tempfile.mkdtemp(prefix = "exl3-uncache-test-", dir = parent) if parent else str(tmp_path)
    try:
        probe = os.path.join(d, "probe")
        with open(probe, "wb") as f:
            f.write(os.urandom(16 * 4096))
            f.flush()
            os.fsync(f.fileno())
            before = resident_pages(libc, probe)
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        after = resident_pages(libc, probe)
        os.remove(probe)
        if before == 0 or after != 0:
            pytest.skip(f"the page cache of {d} cannot be measured or dropped here (tmpfs?); "
                        f"set EXL3_TEST_DISK_DIR to a directory on a real disk")
        yield libc, d
    finally:
        if parent:
            shutil.rmtree(d, ignore_errors = True)


def _tier_files(disk):
    out = []
    for base in (disk.pages_dir, disk.states_dir):
        for dirpath, _, names in os.walk(base):
            out += [os.path.join(dirpath, n) for n in names]
    return sorted(out)


def _drop_all(paths):
    for p in paths:
        fd = os.open(p, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


@pytest.mark.parametrize("on", [False, True])
def test_nothing_of_the_files_is_left_in_the_page_cache(real_disk_dir, monkeypatch, on):
    libc, d = real_disk_dir
    disk = open_disk(os.path.join(d, "on" if on else "off"), monkeypatch, on = on, tier = Tier(slots = 4))
    stash = make_stash(PAGE, n = 40000)
    try:
        for i in range(4):
            write_page(disk, i)
        disk.flush()
        assert disk.write_state(key(3), stash)
        disk.flush()
        files = _tier_files(disk)
        assert len(files) == 5
        written = {p: resident_pages(libc, p) for p in files}
        if on:
            assert all(n == 0 for n in written.values()), f"left in the page cache after the write: {written}"
        else:
            # The control: upstream's write leaves the files in the page cache, header page included
            assert all(n > 0 for n in written.values()), written

        # Read back from the drive (dirty pages are written back by now in the control)
        os.sync()
        _drop_all(files)
        assert all(resident_pages(libc, p) == 0 for p in files)
        for i in range(4):
            slab = torch.zeros(SLAB, dtype = torch.uint8)
            assert disk.read_page_into(key(i), slab) is not None and torch.equal(slab, disk.cpu_tier.slabs[i])
        assert same_stash(disk.read_state(key(3)), stash)
        read = {p: resident_pages(libc, p) for p in files}
        if on:
            assert all(n == 0 for n in read.values()), f"left in the page cache after the read: {read}"
        else:
            assert all(n > 0 for n in read.values()), read
    finally:
        disk.close()
