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

// --- NEW HELPER FUNCTION ---
const findTopMostVisibleAncestor = (
  taskId: string,
  taskMap: Map<string, Task>,
  visibleIds: Set<string>
): string | null => {
  const task = taskMap.get(taskId);
  if (!task?.parent || !visibleIds.has(task.parent)) {
    return null; // No visible parent, so it's a root of a group or standalone
  }

  let current = task;
  let topMostParent = null;

  while (current.parent && visibleIds.has(current.parent)) {
    topMostParent = current.parent;
    current = taskMap.get(current.parent)!;
  }

  return topMostParent;
};


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
    timestampToRowMap,
  } = useMemo(() => {
    // Step 1: Apply focus and text filters (correct)
    let logToProcess = globalLog;
    if (focusedTaskId) {
      const descendantIds = new Set([focusedTaskId]);
      const queue = [focusedTaskId];
      while (queue.length > 0) {
        const currentId = queue.shift()!;
        tasks
          .filter((t) => t.parent === currentId)
          .forEach((child) => {
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
    const columnsToShow = visibleTaskIds
      ? allIds.filter((id) => visibleTaskIds.has(id))
      : allIds;

    const columnsToShowSet = new Set(columnsToShow);

    // --- FIX: REVISED GROUPING LOGIC ---
    const taskMap = new Map(tasks.map((t) => [t.id, t]));
    const parentToChildren = new Map<string, string[]>();
    columnsToShow.forEach((id) => {
      const topMostParent = findTopMostVisibleAncestor(id, taskMap, columnsToShowSet);
      if (topMostParent) {
        if (!parentToChildren.has(topMostParent)) parentToChildren.set(topMostParent, []);
        parentToChildren.get(topMostParent)!.push(id);
      }
    });

    // Step 4: Correctly Apply Sorting and Collapsing
    const pinned = columnsToShow.filter((id) => pinnedTaskIds.has(id)).sort();
    const unpinned = columnsToShow.filter((id) => !pinnedTaskIds.has(id)).sort();
    const sortedColumns = [...pinned, ...unpinned];

    // Children within groups also need to be sorted to maintain order
    parentToChildren.forEach((children) => children.sort());

    const finalDisplayedColumns = sortedColumns.filter((id) => {
        const topMostParent = findTopMostVisibleAncestor(id, taskMap, columnsToShowSet);
        return !(topMostParent && collapsedGroups.has(topMostParent));
    });

    const map = new Map<string, number>();
    finalDisplayedColumns.forEach((id, index) => map.set(id, index + 1));

    const childIdsInGroups = new Set([...parentToChildren.values()].flat());
    const finalStandaloneColumns = finalDisplayedColumns.filter(
      (id) => !childIdsInGroups.has(id),
    );

    const uniqueTimestamps = Array.from(new Set(finalFilteredLog.map(e => e.timestamp))).sort();
    const tsMap = new Map<string, number>();
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
        {focusedTaskId && ( <div className="flex items-center gap-2 bg-blue-900/50 text-blue-300 border border-blue-700/50 rounded-full px-3 py-1 text-sm">
            <span>Focusing on subtree: {focusedTaskId}</span>
            <button onClick={() => onFocusTask(null)} className="p-0.5 rounded-full hover:bg-blue-700" title="Clear focus">
                <XCircle size={16} />
            </button>
        </div> )}
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
              {Array.from(groups.entries()).map(([parentId, children]) => {
                const parentColIdx = taskIdToColumn.get(parentId);
                if (!parentColIdx) return null;

                // Find the columns of the visible children to calculate the span
                const childCols = children
                    .map(childId => taskIdToColumn.get(childId))
                    .filter((c): c is number => c !== undefined)
                    .sort((a,b) => a - b);

                if (childCols.length === 0 && !collapsedGroups.has(parentId)) return null;

                const startCol = parentColIdx;
                // Span should be the distance from the parent to the last child in the group
                const endCol = childCols.length > 0 ? Math.max(...childCols) : startCol;
                const spanCount = endCol - startCol + 1;

                return (
                  <div key={`group-header-${parentId}`} className="sticky top-0 bg-zinc-800 z-20 pt-1" style={{ gridColumn: `${startCol} / span ${spanCount}`, gridRow: 1 }}>
                    <LogGroupHeader parentId={parentId} isCollapsed={collapsedGroups.has(parentId)} onToggle={() => toggleGroupCollapse(parentId)} onHover={onHoverTask} />
                  </div>
                );
              })}

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