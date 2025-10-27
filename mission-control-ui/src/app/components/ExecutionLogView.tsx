"use client";

import { useState } from "react";
import { Task, ExecutionLogEntry } from "@/lib/types";
import { ExecutionLog } from "./ExecutionLog";

// Define a new type for the global log entries which include the taskId
type GlobalLogEntry = ExecutionLogEntry & { taskId: string };

interface ExecutionLogViewProps {
  selectedTask: Task | null;
  globalLog: GlobalLogEntry[];
}

export const ExecutionLogView = ({ selectedTask, globalLog }: ExecutionLogViewProps) => {
  const [viewScope, setViewScope] = useState<"local" | "global">("local");

  const localLog = selectedTask?.execution_log ?? [];

  return (
    <div className="bg-zinc-800/50 p-4 rounded-lg border border-zinc-700 h-full flex flex-col">
      <div className="flex-shrink-0 mb-4 flex justify-between items-center">
        <div>
          <h2 className="text-xl font-bold text-zinc-100">Execution Log</h2>
          {viewScope === 'local' ? (
            <p className="font-mono text-xs text-zinc-400">
              Displaying steps for Task:{" "}
              <span className="text-zinc-200">{selectedTask?.id ?? "None"}</span>
            </p>
          ) : (
            <p className="font-mono text-xs text-zinc-400">
              Displaying global, chronological log for the entire swarm.
            </p>
          )}
        </div>

        {/* The Toggle Buttons from your Mockup */}
        <div className="flex items-center p-1 bg-zinc-900 rounded-lg border border-zinc-700">
          <button
            onClick={() => setViewScope("global")}
            className={`px-3 py-1 text-sm rounded-md ${viewScope === "global" ? "bg-blue-600 text-white" : "text-zinc-400 hover:bg-zinc-700"}`}
          >
            Global
          </button>
          <button
            onClick={() => setViewScope("local")}
            className={`px-3 py-1 text-sm rounded-md ${viewScope === "local" ? "bg-blue-600 text-white" : "text-zinc-400 hover:bg-zinc-700"}`}
          >
            Local
          </button>
        </div>
      </div>

      <div className="flex-grow overflow-y-auto min-h-0 pr-2">
        {viewScope === 'local' && (
          <ExecutionLog log={localLog} />
        )}
        {viewScope === 'global' && (
           // Passing the new taskId prop
          <ExecutionLog log={globalLog} showTaskId={true} />
        )}
      </div>
    </div>
  );
};