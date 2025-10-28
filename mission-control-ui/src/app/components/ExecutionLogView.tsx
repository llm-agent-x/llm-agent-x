// mission-control-ui/src/app/components/ExecutionLogView.tsx
"use client";

import { useState, useMemo } from "react";
import { Task, ExecutionLogEntry } from "@/lib/types";
import { ExecutionLog } from "./ExecutionLog";
import { LogFilterBar } from "./LogFilterBar";
import { ColumnManager } from "./ColumnManager";
import { LogGroupHeader } from "./LogGroupHeader";
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

  const toggleGroupCollapse = (groupId: string) => {
    setCollapsedGroups((prev) => {
      const next = new Set(prev);
      next.has(groupId) ? next.delete(groupId) : next.add(groupId);
      return next;
    });
  };

  const {
    filteredLog,
    allTaskIdsInLog,
    displayedColumns,
    taskIdToColumn,
    groups,
    standaloneColumns,
    timestampToRowMap, // <-- The key to the new logic
  } = useMemo(() => {
    // Step 1: Apply focus and text filters (correct)
    let logToProcess = globalLog;
    if (focusedTaskId) {
      const descendantIds = new Set([focusedTaskId]);
      const queue = [focusedTaskId];
      while (queue.length > 0) {
        const currentId = queue.shift()!;
        tasks.filter((t) => t.parent === currentId).forEach((child) => {
          if (!descendantIds.has(child.id)) {
            descendantIds.add(child.id);
            queue.push(child.id);
          }
        });
      }
      logToProcess = globalLog.filter((entry) => descendantIds.has(entry.taskId));
    }
    const lowerCaseFilter = filterText.toLowerCase();
    const finalFilteredLog = lowerCaseFilter
      ? logToProcess.filter((entry) => {
        const task = tasks.find((t) => t.id === entry.taskId);
        const tagsMatch = task?.tags?.some((tag) => `#${tag}`.includes(lowerCaseFilter)) ?? false;
        return entry.taskId.toLowerCase().includes(lowerCaseFilter) || tagsMatch;
      })
      : logToProcess;

    // Step 2: Determine columns to show based on visibility filters (correct)
    const allIds = Array.from(new Set(finalFilteredLog.map((e) => e.taskId)));
    const columnsToShow = visibleTaskIds ? allIds.filter((id) => visibleTaskIds.has(id)) : allIds;

    // Step 3: Grouping logic (correct)
    const taskMap = new Map(tasks.map((t) => [t.id, t]));
    const parentToChildren = new Map<string, string[]>();
    columnsToShow.forEach((id) => {
      const task = taskMap.get(id);
      if (task?.parent && columnsToShow.includes(task.parent)) {
        if (!parentToChildren.has(task.parent)) parentToChildren.set(task.parent, []);
        parentToChildren.get(task.parent)!.push(id);
      }
    });

    // Step 4: Correctly Apply Sorting and Collapsing
    const pinned = columnsToShow.filter((id) => pinnedTaskIds.has(id)).sort();
    const unpinned = columnsToShow.filter((id) => !pinnedTaskIds.has(id)).sort();
    const sortedColumns = [...pinned, ...unpinned];

    const finalDisplayedColumns = sortedColumns.filter((id) => {
      const task = taskMap.get(id);
      return !(task?.parent && collapsedGroups.has(task.parent));
    });

    const map = new Map<string, number>();
    finalDisplayedColumns.forEach((id, index) => map.set(id, index + 1));

    const childIdsInGroups = new Set([...parentToChildren.values()].flat());
    const finalStandaloneColumns = finalDisplayedColumns.filter((id) => !childIdsInGroups.has(id));

    // --- FIX: Create the timestamp-to-row mapping ---
    const uniqueTimestamps = Array.from(new Set(finalFilteredLog.map(e => e.timestamp))).sort();
    const tsMap = new Map<string, number>();
    // Headers will occupy rows 1 and 2, so logs start on row 3
    uniqueTimestamps.forEach((ts, index) => ts && tsMap.set(ts, index + 3));

    return {
      filteredLog: finalFilteredLog,
      allTaskIdsInLog: allIds,
      displayedColumns: finalDisplayedColumns,
      taskIdToColumn: map,
      groups: parentToChildren,
      standaloneColumns: finalStandaloneColumns,
      timestampToRowMap: tsMap,
    };
  }, [ globalLog, tasks, focusedTaskId, filterText, visibleTaskIds, pinnedTaskIds, collapsedGroups ]);

  return (
    <div className="bg-zinc-800/50 p-4 rounded-lg border border-zinc-700 h-full flex flex-col">
      {/* Header Bar is correct */}
      <div className="flex-shrink-0 mb-4 flex justify-between items-center gap-4">
        <h2 className="text-xl font-bold text-zinc-100">Execution Log</h2>
        {focusedTaskId && ( <div /* ... */ ></div> )}
        <div className="flex-grow">
          <LogFilterBar filterText={filterText} onFilterChange={setFilterText} />
        </div>
        <div className="flex items-center gap-2">
            <ColumnManager allTaskIds={allTaskIdsInLog} visibleTaskIds={visibleTaskIds} onToggleVisibility={id => setVisibleTaskIds(prev => {
                const next = new Set(prev ?? allTaskIdsInLog);
                next.has(id) ? next.delete(id) : next.add(id);
                return next;
            })} pinnedTaskIds={pinnedTaskIds} onTogglePin={id => setPinnedTaskIds(prev => {
                const next = new Set(prev);
                next.has(id) ? next.delete(id) : next.add(id);
                return next;
            })} />
            <div className="flex items-center p-1 bg-zinc-900 rounded-lg border border-zinc-700">
                <button onClick={() => setViewMode("list")} className={`px-3 py-1 text-sm rounded-md flex items-center gap-2 ${viewMode === "list" ? "bg-blue-600 text-white" : "text-zinc-400 hover:bg-zinc-700"}`}>
                    <LayoutList size={16} /> List
                </button>
                <button onClick={() => setViewMode("grid")} className={`px-3 py-1 text-sm rounded-md flex items-center gap-2 ${viewMode === "grid" ? "bg-blue-600 text-white" : "text-zinc-400 hover:bg-zinc-700"}`}>
                    <LayoutGrid size={16} /> Grid
                </button>
            </div>
        </div>
      </div>

      <div className="flex-grow overflow-auto min-h-0">
        {viewMode === "list" && (
            <ExecutionLog log={filteredLog} showTaskId={true} hoveredTaskId={hoveredTaskId} onHoverTask={onHoverTask} />
        )}

        {viewMode === "grid" &&
          (displayedColumns.length > 0 ? (
            <div
              className="grid gap-x-2 relative"
              style={{
                gridTemplateColumns: `repeat(${displayedColumns.length}, minmax(300px, 1fr))`,
              }}
            >
              {/* --- Layer 1: Group Headers --- */}
              {Array.from(groups.entries()).map(([parentId, children]) => {
                const startCol = taskIdToColumn.get(parentId);
                if (!startCol) return null;
                const visibleChildren = children.filter(childId => taskIdToColumn.has(childId));
                const spanCount = 1 + visibleChildren.length;

                return (
                  <div key={`group-header-${parentId}`} className="sticky top-0 bg-zinc-800 z-20 pt-1" style={{ gridColumn: `${startCol} / span ${spanCount}`, gridRow: 1 }}>
                    <LogGroupHeader parentId={parentId} isCollapsed={collapsedGroups.has(parentId)} onToggle={() => toggleGroupCollapse(parentId)} onHover={onHoverTask} />
                  </div>
                );
              })}

              {/* --- Layer 2: Individual Task Headers --- */}
              {standaloneColumns.map((taskId) => {
                const col = taskIdToColumn.get(taskId);
                if (!col) return null;
                const isGroupParent = groups.has(taskId);

                return (
                  <div key={`header-${taskId}`} className="sticky top-0 bg-zinc-800 z-10 pt-1" style={{ gridColumn: col, gridRow: isGroupParent ? 2 : 1 }}>
                    <h3
                      className="font-mono text-sm text-zinc-300 truncate font-semibold bg-zinc-900/50 border border-zinc-700 p-2 rounded-md hover:bg-zinc-900 cursor-pointer"
                      onMouseEnter={() => onHoverTask(taskId)}
                      onMouseLeave={() => onHoverTask(null)}
                    >
                      {taskId}
                    </h3>
                  </div>
                );
              })}

              {/* --- Layer 3: Log Entries --- */}
              {filteredLog.map((entry, logIndex) => {
                const columnIndex = taskIdToColumn.get(entry.taskId);
                const rowIndex = entry.timestamp ? timestampToRowMap.get(entry.timestamp) : undefined;
                if (!columnIndex || !rowIndex) return null;

                return (
                  <div
                    key={logIndex}
                    className="pr-2"
                    style={{ gridRow: rowIndex, gridColumn: columnIndex }}
                  >
                    <ExecutionLog log={[entry]} showTaskId={false} hoveredTaskId={hoveredTaskId} onHoverTask={onHoverTask} />
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