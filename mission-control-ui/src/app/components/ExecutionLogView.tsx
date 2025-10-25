"use client";

import { Task } from "@/lib/types";
import { ExecutionLog } from "./ExecutionLog";

export const ExecutionLogView = ({ task }: { task: Task | null }) => {
  return (
    <div className="bg-zinc-800/50 p-4 rounded-lg border border-zinc-700 h-full flex flex-col">
      <div className="flex-shrink-0 mb-4">
        <h2 className="text-xl font-bold text-zinc-100">Execution Log</h2>
        {task ? (
          <p className="font-mono text-xs text-zinc-400">
            Displaying internal steps for Task:{" "}
            <span className="text-zinc-200">{task.id}</span>
          </p>
        ) : (
          <p className="text-zinc-500">No task selected.</p>
        )}
      </div>
      <div className="flex-grow overflow-y-auto min-h-0 pr-2">
        {task ? (
          <ExecutionLog log={task.execution_log} />
        ) : (
          <div className="text-center text-zinc-500 py-8">
             No task selected.
          </div>
        )}
      </div>
    </div>
  );
};