import asyncio
import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Set, Dict, Any

@dataclass
class Task:
    id: str
    desc: str
    deps: Set[str] = field(default_factory=set)
    is_dynamic: bool = False  # For completeness, though not used here
    status: str = "pending"
    result: Any = None

    async def run(self, dep_results: Dict[str, Any]) -> Any:
        """Concatenate all dependencies' results (sorted order), sha256 them, return hex digest"""
        # For roots (no deps): hash its own ID (or desc) for determinism
        if not self.deps:
            data = self.id.encode()
        else:
            concat = b''.join(bytes.fromhex(dep_results[d]) for d in sorted(self.deps))
            data = concat
        digest = hashlib.sha256(data).hexdigest()
        self.result = digest
        return digest

class TaskRegistry:
    def __init__(self):
        self.tasks: Dict[str, Task] = {}

    def add_task(self, task: Task):
        if task.id in self.tasks:
            raise ValueError(f"Task {task.id} already exists")
        self.tasks[task.id] = task

    def add_dependency(self, task_id: str, dep_id: str):
        if dep_id not in self.tasks:
            raise ValueError(f"Dependency {dep_id} does not exist")
        self.tasks[task_id].deps.add(dep_id)

    def all_tasks(self):
        return list(self.tasks.values())

async def run_dag_async(registry: TaskRegistry):
    finished = {}
    pending = set(registry.tasks.keys())
    while pending:
        # Eligible tasks: all deps finished and not yet run
        ready = [
            tid for tid in pending
            if all(d in finished for d in registry.tasks[tid].deps)
        ]
        if not ready:
            raise Exception("DAG is not runnable (cycle?)")
        tasks = []
        for tid in ready:
            t = registry.tasks[tid]
            dep_res = {d: finished[d] for d in t.deps}
            tasks.append(asyncio.create_task(t.run(dep_res)))
        for tid, atask in zip(ready, tasks):
            h = await atask
            registry.tasks[tid].status = "complete"
            registry.tasks[tid].result = h
            finished[tid] = h
            print(f"{tid}: {registry.tasks[tid].desc} => {h}")
        for tid in ready:
            pending.remove(tid)
    return finished

def build_example_registry():
    reg = TaskRegistry()
    # Build a diamond-shaped DAG:
    #        root
    #       /    \
    #    midA   midB
    #       \    /
    #        leaf
    root = Task(id="root", desc="Root task (no deps)")
    midA = Task(id="midA", desc="A depends on root", deps={"root"})
    midB = Task(id="midB", desc="B depends on root", deps={"root"})
    leaf = Task(id="leaf", desc="Leaf depends on A and B", deps={"midA", "midB"})
    for t in [root, midA, midB, leaf]:
        reg.add_task(t)
    return reg

if __name__ == '__main__':
    reg = build_example_registry()
    asyncio.run(run_dag_async(reg))

    # Now, you can verify any output:
    print("\nVerifying all outputs...")
    for t in reg.all_tasks():
        # Locally recompute what the output should have been:
        if not t.deps:
            data = t.id.encode()
        else:
            concat = b''.join(bytes.fromhex(reg.tasks[d].result) for d in sorted(t.deps))
            data = concat
        digest = hashlib.sha256(data).hexdigest()
        assert digest == t.result, f"Verification failed for task {t.id}"
    print("All tasks verified!")