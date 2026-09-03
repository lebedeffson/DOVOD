from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'v24_runtime'))
from playable_runtime import PlayableRuntimeV24

def main():
    rt=PlayableRuntimeV24();print('DOVOD procedural runtime prototype. Commands: state, actions, try <id>, do <id>, reset, quit')
    while True:
        try:s=input('dovod> ').strip()
        except EOFError:break
        if not s:continue
        if s in {'quit','exit'}:break
        if s=='reset':rt.reset();print(rt.snapshot());continue
        if s=='state':print(rt.snapshot());continue
        if s=='actions':
            for a in rt.proc.actions:print(('*' if rt.feasible(a.action_id) else ' '),a.action_id,'-',a.label)
            continue
        cmd,*rest=s.split(maxsplit=1);pa=rest[0] if rest else ''
        if pa not in rt.by_proc:print('unknown action');continue
        x=rt.by_proc[pa];d=rt.decide(int(x['verb']),int(x['noun']),int(x['riskActionLabel']),pa);print('decision',d)
        if cmd=='do':print('commit',rt.commit(pa)[0],rt.snapshot())
        elif cmd!='try':print('use try/do')
if __name__=='__main__':main()
