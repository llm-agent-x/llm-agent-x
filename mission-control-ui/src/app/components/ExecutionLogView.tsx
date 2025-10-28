// mission-control-ui/src/app/components/ExecutionLogView.tsx
"use client";

import { useState, useMemo } from "react";
import { Task, ExecutionLogEntry } from "@/lib/types";
import { ExecutionLog } from "./ExecutionLog";
import { LogFilterBar } from "./LogFilterBar";
import { ColumnManager } from "./ColumnManager";
import { LayoutGrid, LayoutList, XCircle } from "lucide-react";

type GlobalLogEntry = ExecutionLogEntry & { taskId: string };

interface ExecutionLogViewProps {
  tasks: Task[];
  globalLog: GlobalLogEntry[];
  hoveredTaskId: string | null;
  onHoverTask: (id: string | null) => void;
  focusedTaskId: string | null;
  onFocusTask: (id: string | null) => void;
}

export const ExecutionLogView = ({
  tasks,
  globalLog,
  hoveredTaskId,
  onHoverTask,
  focusedTaskId,
  onFocusTask,
}: ExecutionLogViewProps) => {
  const [viewMode, setViewMode] = useState<"list" | "grid">("list");
  const [filterText, setFilterText] = useState("");
  const [visibleTaskIds, setVisibleTaskIds] = useState<Set<string> | null>(
    null,
  );
  const [pinnedTaskIds, setPinnedTaskIds] = useState<Set<string>>(new Set());
  const [collapsedGroups, setCollapsedGroups] = useState<Set<string>>(
    new Set(),
  );

  const {
    filteredLog,
    allTaskIdsInLog,
    taskIdsForGrid,
    taskIdToColumn,
    groups,
  } = useMemo(() => {
    // --- 1. Apply Focus Filter ---
    let logToProcess = globalLog;
    if (focusedTaskId) {
      const descendantIds = new Set([focusedTaskId]);
      const queue = [focusedTaskId];
      while (queue.length > 0) {
        const currentId = queue.shift()!;
        const children = tasks.filter((t) => t.parent === currentId);
        children.forEach((child) => {
          if (!descendantIds.has(child.id)) {
            descendantIds.add(child.id);
            queue.push(child.id);
          }
        });
      }
      logToProcess = globalLog.filter((entry) =>
        descendantIds.has(entry.taskId),
      );
    }

    // --- 2. Apply Text/Tag Filter ---
    const lowerCaseFilter = filterText.toLowerCase();
    const finalFilteredLog = lowerCaseFilter
      ? logToProcess.filter((entry) => {
          const task = tasks.find((t) => t.id === entry.taskId);
          const tagsMatch =
            task?.tags?.some((tag) => `#${tag}`.includes(lowerCaseFilter)) ??
            false;
          return entry.taskId.toLowerCase().includes(lowerCaseFilter) || tagsMatch;
        })
      : logToProcess;

    const allIds = Array.from(new Set(finalFilteredLog.map((e) => e.taskId)));

    // --- 3. Determine Visible Columns ---
    let columnsToShow = visibleTaskIds
      ? allIds.filter((id) => visibleTaskIds.has(id))
      : allIds;

    // --- 4. Grouping Logic ---
    const taskMap = new Map(tasks.map((t) => [t.id, t]));
    const parentToChildren = new Map<string, string[]>();
    columnsToShow.forEach((id) => {
      const task = taskMap.get(id);
      if (task?.parent && columnsToShow.includes(task.parent)) {
        if (!parentToChildren.has(task.parent)) {
          parentToChildren.set(task.parent, []);
        }
        parentToChildren.get(task.parent)!.push(id);
      }
    });

    const finalGroups = new Map<string, string[]>();
    const standaloneIds = new Set(columnsToShow);
    parentToChildren.forEach((children, parentId) => {
      finalGroups.set(parentId, children);
      standaloneIds.delete(parentId);
      children.forEach((childId) => standaloneIds.delete(childId));
    });

    // --- 5. Apply Collapsing ---
    if (collapsedGroups.size > 0) {
      const idsToHide = new Set<string>();
      collapsedGroups.forEach((groupId) => {
        parentToChildren.get(groupId)?.forEach((childId) => idsToHide.add(childId));
      });
      columnsToShow = columnsToShow.filter((id) => !idsToHide.has(id));
    }

    // --- 6. Apply Pinning (Sort Order) ---
    columnsToShow.sort((a, b) => {
      const aIsPinned = pinnedTaskIds.has(a);
      const bIsPinned = pinnedTaskIds.has(b);
      if (aIsPinned && !bIsPinned) return -1;
      if (!aIsPinned && bIsPinned) return 1;
      return a.localeCompare(b);
    });

    // --- 7. Create Grid Mapping ---
    const map = new Map<string, number>();
    columnsToShow.forEach((id, index) => map.set(id, index + 1));

    return {
      filteredLog: finalFilteredLog,
      allTaskIdsInLog: allIds,
      taskIdsForGrid: columnsToShow,
      taskIdToColumn: map,
      groups: finalGroups,
    };
  }, [
    globalLog,
    tasks,
    focusedTaskId,
    filterText,
    visibleTaskIds,
    pinnedTaskIds,
    collapsedGroups,
  ]);

  return (
    <div className="bg-zinc-800/50 p-4 rounded-lg border border-zinc-700 h-full flex flex-col">
      <div className="flex-shrink-0 mb-4 flex justify-between items-center gap-4">
        <h2 className="text-xl font-bold text-zinc-100">Execution Log</h2>

        {focusedTaskId && (
          <div className="flex items-center gap-2 bg-blue-900/50 text-blue-300 border border-blue-700/50 rounded-full px-3 py-1 text-sm">
            <span>Focusing on subtree: {focusedTaskId}</span>
            <button
              onClick={() => onFocusTask(null)}
              className="p-0.5 rounded-full hover:bg-blue-700"
              title="Clear focus"
            >
              <XCircle size={16} />
            </button>
          </div>
        )}

        <div className="flex-grow">
          <LogFilterBar filterText={filterText} onFilterChange={setFilterText} />
        </div>

        <div className="flex items-center gap-2">
          <ColumnManager
            allTaskIds={allTaskIdsInLog}
            visibleTaskIds={visibleTaskIds}
            onToggleVisibility={(id) => {
              setVisibleTaskIds((prev) => {
                const next = new Set(prev ?? allTaskIdsInLog);
                next.has(id) ? next.delete(id) : next.add(id);
                return next;
              });
            }}
            pinnedTaskIds={pinnedTaskIds}
            onTogglePin={(id) => {
              setPinnedTaskIds((prev) => {
                const next = new Set(prev);
                next.has(id) ? next.delete(id) : next.add(id);
                return next;
              });
            }}
          />
          <div className="flex items-center p-1 bg-zinc-900 rounded-lg border border-zinc-700">
            <button
              onClick={() => setViewMode("list")}
              className={`px-3 py-1 text-sm rounded-md flex items-center gap-2 ${
                viewMode === "list"
                  ? "bg-blue-600 text-white"
                  : "text-zinc-400 hover:bg-zinc-700"
              }`}
            >
              <LayoutList size={16} /> List
            </button>
            <button
              onClick={() => setViewMode("grid")}
              className={`px-3 py-1 text-sm rounded-md flex items-center gap-2 ${
                viewMode === "grid"
                  ? "bg-blue-600 text-white"
                  : "text-zinc-400 hover:bg-zinc-700"
              }`}
            >
              <LayoutGrid size={16} /> Grid
            </button>
          </div>
        </div>
      </div>

      <div className="flex-grow overflow-auto min-h-0">
        {viewMode === "list" && (
          <ExecutionLog
            log={filteredLog}
            showTaskId={true}
            hoveredTaskId={hoveredTaskId}
            onHoverTask={onHoverTask}
          />
        )}

        {viewMode === "grid" &&
          (taskIdsForGrid.length > 0 ? (
            <div
              className="grid gap-x-2 relative"
              style={{
                gridTemplateColumns: `repeat(${taskIdsForGrid.length}, minmax(300px, 1fr))`,
              }}
            >
              {taskIdsForGrid.map((taskId, index) => (
                <div
                  key={`header-${taskId}`}
                  className="sticky top-0 bg-zinc-800 z-10 py-1"
                  style={{ gridColumn: index + 1, gridRow: 1 }}
                >
                  <h3 className="font-mono text-sm text-zinc-300 truncate font-semibold bg-zinc-900/50 border border-zinc-700 p-2 rounded-md hover:bg-zinc-900 cursor-pointer"
                      onMouseEnter={() => onHoverTask(taskId)}
                      onMouseLeave={() => onHoverTask(null)}
                  >
                    {taskId}
                  </h3>
                </div>
              ))}

              {filteredLog.map((entry, logIndex) => {
                const columnIndex = taskIdToColumn.get(entry.taskId);
                if (!columnIndex) return null;

                return (
                  <div
                    key={logIndex}
                    className="py-1 pr-2 border-t border-zinc-800/50"
                    style={{
                      gridRow: logIndex + 2,
                      gridColumn: columnIndex,
                    }}
                  >
                    <ExecutionLog
                      log={[entry]}
                      showTaskId={false}
                      hoveredTaskId={hoveredTaskId}
                      onHoverTask={onHoverTask}
                    />
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="text-center text-zinc-500 py-8 w-full">
              No execution steps match your filters.
            </div>
          ))}
      </div>
    </div>
  );
};