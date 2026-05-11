import threading
import time

from search.rwlock import RWLock


def test_concurrent_reads_do_not_block():
    lock = RWLock()
    barrier = threading.Barrier(3)
    durations: list[float] = []

    def reader():
        with lock.read():
            barrier.wait()
            time.sleep(0.05)

    threads = [threading.Thread(target=reader) for _ in range(3)]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - start
    durations.append(elapsed)

    assert elapsed < 0.15, f"3 concurrent reads of 50ms each took {elapsed:.3f}s; should be ~0.05s"


def test_writer_excludes_reader():
    lock = RWLock()
    seq: list[str] = []
    started = threading.Event()

    def writer():
        with lock.write():
            started.set()
            time.sleep(0.05)
            seq.append("w")

    def reader():
        started.wait()
        with lock.read():
            seq.append("r")

    w = threading.Thread(target=writer)
    r = threading.Thread(target=reader)
    w.start()
    r.start()
    w.join()
    r.join()

    assert seq == ["w", "r"]


def test_writer_excludes_writer():
    lock = RWLock()
    seq: list[int] = []

    def writer(tag: int, delay: float):
        with lock.write():
            seq.append(tag)
            time.sleep(delay)
            seq.append(tag)

    a = threading.Thread(target=writer, args=(1, 0.05))
    b = threading.Thread(target=writer, args=(2, 0.0))
    a.start()
    time.sleep(0.01)
    b.start()
    a.join()
    b.join()

    assert seq == [1, 1, 2, 2]


def test_writer_preference_blocks_new_readers():
    lock = RWLock()
    seq: list[str] = []
    reader_holds = threading.Event()
    writer_waiting = threading.Event()

    def initial_reader():
        with lock.read():
            reader_holds.set()
            writer_waiting.wait(timeout=1.0)
            time.sleep(0.05)
            seq.append("r1")

    def writer():
        reader_holds.wait()
        writer_waiting.set()
        with lock.write():
            seq.append("w")

    def latecomer_reader():
        reader_holds.wait()
        writer_waiting.wait()
        time.sleep(0.01)
        with lock.read():
            seq.append("r2")

    r1 = threading.Thread(target=initial_reader)
    w = threading.Thread(target=writer)
    r2 = threading.Thread(target=latecomer_reader)
    r1.start()
    w.start()
    r2.start()
    r1.join()
    w.join()
    r2.join()

    assert seq == ["r1", "w", "r2"]
