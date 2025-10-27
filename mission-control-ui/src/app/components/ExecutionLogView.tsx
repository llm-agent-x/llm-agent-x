// mission-control-ui/src/app/components/ExecutionLogView.tsx
"use client";

import { useState, useMemo } from "react";
import { Task, ExecutionLogEntry } from "@/lib/types";
import { ExecutionLog } from "./ExecutionLog";
import { LayoutList, LayoutGrid } from "lucide-react";

type GlobalLogEntry = ExecutionLogEntry & { taskId: string };

interface ExecutionLogViewProps {
  selectedTask: Task | null;
  globalLog: GlobalLogEntry[];
}

export const ExecutionLogView = ({ selectedTask, globalLog }: ExecutionLogViewProps) => {
  const [viewMode, setViewMode] = useState<'list' | 'grid'>('list');

  // This hook now calculates the data needed for our CSS Grid layout.
  const { taskIdsForGrid, taskIdToColumn } = useMemo(() => {
    if (viewMode !== 'grid') {
      return { taskIdsForGrid: [], taskIdToColumn: new Map() };
    }
    // Get unique task IDs that have logs. Using a Set then converting to an array
    // preserves the order in which tasks first appeared in the log.
    const uniqueTaskIds = Array.from(new Set(globalLog.map(entry => entry.taskId)));

    // Create a map for quick lookup: { taskId -> columnIndex }
    const map = new Map<string, number>();
    uniqueTaskIds.forEach((id, index) => {
      map.set(id, index + 1); // CSS Grid columns are 1-indexed
    });

    return { taskIdsForGrid: uniqueTaskIds, taskIdToColumn: map };
  }, [globalLog, viewMode]);

  return (
    <div className="bg-zinc-800/50 p-4 rounded-lg border border-zinc-700 h-full flex flex-col">
      <div className="flex-shrink-0 mb-4 flex justify-between items-center">
        <h2 className="text-xl font-bold text-zinc-100">Execution Log</h2>
        <div className="flex items-center p-1 bg-zinc-900 rounded-lg border border-zinc-700">
          <button
            onClick={() => setViewMode("list")}
            className={`px-3 py-1 text-sm rounded-md flex items-center gap-2 ${viewMode === "list" ? "bg-blue-600 text-white" : "text-zinc-400 hover:bg-zinc-700"}`}
          >
            <LayoutList size={16} /> List
          </button>
          <button
            onClick={() => setViewMode("grid")}
            className={`px-3 py-1 text-sm rounded-md flex items-center gap-2 ${viewMode === "grid" ? "bg-blue-600 text-white" : "text-zinc-400 hover:bg-zinc-700"}`}
          >
            <LayoutGrid size={16} /> Grid
          </button>
        </div>
      </div>

      <div className="flex-grow overflow-auto min-h-0">
        {viewMode === 'list' && (
          // The linear view remains unchanged
          <ExecutionLog log={globalLog} showTaskId={true} />
        )}

        {viewMode === 'grid' && (
          taskIdsForGrid.length > 0 ? (
            <div
              className="grid gap-x-2"
              style={{
                // Define the grid columns dynamically. Each column is at least 300px wide.
                gridTemplateColumns: `repeat(${taskIdsForGrid.length}, minmax(300px, 1fr))`,
              }}
            >
              {/* --- 1. RENDER STICKY HEADERS --- */}
              {taskIdsForGrid.map((taskId, index) => (
                <div
                  key={`header-${taskId}`}
                  className="sticky top-0 bg-zinc-800 z-10 py-1"
                  style={{ gridColumn: index + 1, gridRow: 1 }}
                >
                  <h3 className="font-mono text-sm text-zinc-300 truncate font-semibold bg-zinc-900/50 border border-zinc-700 p-2 rounded-md">
                    {taskId}
                  </h3>
                </div>
              ))}

              {/* --- 2. RENDER LOG ENTRIES IN THEIR CORRECT GRID CELL --- */}
              {globalLog.map((entry, logIndex) => {
                const columnIndex = taskIdToColumn.get(entry.taskId);
                if (!columnIndex) return null;

                return (
                  <div
                    key={logIndex}
                    className="py-1 pr-2 border-t border-zinc-800/50"
                    style={{
                      gridRow: logIndex + 2, // Headers are in row 1, so logs start at row 2
                      gridColumn: columnIndex,
                    }}
                  >
                    {/*
                      Here, we reuse the ExecutionLog component to render a *single* log entry.
                      This is slightly inefficient but great for code reuse. We pass an array with one item.
                    */}
                    <ExecutionLog log={[entry]} showTaskId={false} />
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="text-center text-zinc-500 py-8 w-full">
              No execution steps logged.
            </div>
          )
        )}
      </div>
    </div>
  );
};