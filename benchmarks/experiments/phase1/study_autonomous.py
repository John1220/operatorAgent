"""Production autonomous supervisor/worker with an explicit study sampling scope."""
import sys
from operator_agent.app import autonomous
from .sampling import bounded


def main(argv=None):
    original=autonomous._start_worker
    def worker(run,deadline,started,invocation):
        from operator_agent.execute.exec_guard import start_task_worker
        return start_task_worker([sys.executable,'-m',__spec__.name,'--worker',str(run),
            str(deadline),str(started),invocation],cwd=str(autonomous.Path(autonomous.__file__).resolve().parents[2]))
    autonomous._start_worker=worker
    try:
        with bounded():return autonomous.main(argv)
    finally:autonomous._start_worker=original


if __name__=='__main__':raise SystemExit(main())
