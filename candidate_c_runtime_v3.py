"""Launch the actual interpreter with explicitly pinned venv packages, without site hooks."""
import json,sys,sysconfig,subprocess,time
from pathlib import Path
def command(script,args):
    paths=list(dict.fromkeys([str(Path(script).resolve().parent),sysconfig.get_path('purelib'),sysconfig.get_path('platlib')]))
    bootstrap='import sys,runpy;sys.path[:0]='+repr(paths)+';sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name="__main__")'
    return [getattr(sys,'_base_executable',None) or sys.executable,'-I','-S','-c',bootstrap,str(Path(script).resolve()),*map(str,args)]

def supervise(script,args,rss_fn,seconds=120,memory_bytes=512*1024**2):
    start=time.monotonic();peak=0;proc=subprocess.Popen(command(script,args));reason=None
    try:
        while proc.poll() is None:
            peak=max(peak,rss_fn(proc.pid));elapsed=time.monotonic()-start
            if peak>memory_bytes:reason='memory_limit';break
            if elapsed>seconds:reason='wall_time_limit';break
            time.sleep(.1)
        if reason:
            proc.kill();proc.wait()
        else:proc.wait()
        info=dict(pid=proc.pid,returncode=proc.returncode,seconds=time.monotonic()-start,peak_rss_bytes=peak,
            stop_reason=reason or ('child_exit' if proc.returncode else 'completed'),
            executable=getattr(sys,'_base_executable',None) or sys.executable,
            wall_limit_seconds=seconds,memory_limit_bytes=memory_bytes)
        return info
    finally:
        if proc.poll() is None:proc.kill();proc.wait()
