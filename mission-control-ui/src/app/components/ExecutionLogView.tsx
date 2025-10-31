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

const findTopMostVisibleAncestor = (
  taskId: string,
  taskMap: Map<string, Task>,
  visibleIds: Set<string>,
): string | null => {
  const task = taskMap.get(taskId);
  if (!task?.parent || !visibleIds.has(task.parent)) {
    return null;
  }
  let current = task;
  let topMostParent = null;
  while (current.parent && visibleIds.has(current.parent)) {
    topMostParent = current.parent;
    current = taskMap.get(current.parent)!;
  }
  return topMostParent;
};

// --- NEW COMPONENT to render the content of a collapsed group ---
const CollapsedGroupContent = ({ memberIds, log, hoveredTaskId, onHoverTask }: { memberIds: string[], log: GlobalLogEntry[], hoveredTaskId: string | null, onHoverTask: (id: string | null) => void }) => {
  const memberIdSet = new Set(memberIds);
  const relevantLog = log.filter(entry => memberIdSet.has(entry.taskId));
  return (
    <div className="bg-zinc-900/50 border border-zinc-700 p-2 rounded-md h-full overflow-y-auto">
      <ExecutionLog log={relevantLog} showTaskId={true} hoveredTaskId={hoveredTaskId} onHoverTask={onHoverTask} />
    </div>
  );
};

// --- Main Component ---
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
  const [visibleTaskIds, setVisibleTaskIds] = useState<Set<string> | null>(null);
  const [pinnedTaskIds, setPinnedTaskIds] = useState<Set<string>>(new Set());
  const [collapsedGroups, setCollapsedGroups] = useState<Set<string>>(new Set());

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
    displayedLayout, // <-- The new layout plan
    taskIdToColumn,
    timestampToRowMap,
  } = useMemo(() => {
    // Step 1: Filtering (no changes)
    let logToProcess = globalLog;
    if (focusedTaskId) {
      const descendantIds = new Set([focusedTaskId]);
      const queue = [focusedTaskId];
      while (queue.length > 0) {
        const currentId = queue.shift()!;
        tasks.filter((t) => t.parent === currentId).forEach((child) => {
          if (!descendantIds.has(child.id)) { descendantIds.add(child.id); queue.push(child.id); }
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

    const allIds = Array.from(new Set(finalFilteredLog.map((e) => e.taskId)));
    const columnsToShow = visibleTaskIds ? allIds.filter((id) => visibleTaskIds.has(id)) : allIds;
    const columnsToShowSet = new Set(columnsToShow);
    const taskMap = new Map(tasks.map((t) => [t.id, t]));

    // Step 2: Semantic Grouping via Tags + Hierarchy (no changes)
    const topLevelParents = new Set<string>();
    const childToTopParent = new Map<string, string>();
    columnsToShow.forEach((id) => {
      const topMostParent = findTopMostVisibleAncestor(id, taskMap, columnsToShowSet) || id;
      topLevelParents.add(topMostParent);
      childToTopParent.set(id, topMostParent);
    });

    const tagToGroups = new Map<string, string[]>();
    topLevelParents.forEach((parentId) => {
      const groupMemberIds = columnsToShow.filter((id) => childToTopParent.get(id) === parentId);
      const groupTags = new Set<string>();
      groupMemberIds.forEach((memberId) => { taskMap.get(memberId)?.tags.forEach((tag) => groupTags.add(tag)); });
      groupTags.forEach((tag) => {
        if (!tagToGroups.has(tag)) tagToGroups.set(tag, []);
        tagToGroups.get(tag)!.push(parentId);
      });
    });

    const groupGraph = new Map<string, Set<string>>();
    tagToGroups.forEach((groupsWithTag) => {
      for (let i = 0; i < groupsWithTag.length; i++) {
        for (let j = i + 1; j < groupsWithTag.length; j++) {
          const group1 = groupsWithTag[i]; const group2 = groupsWithTag[j];
          if (!groupGraph.has(group1)) groupGraph.set(group1, new Set());
          if (!groupGraph.has(group2)) groupGraph.set(group2, new Set());
          groupGraph.get(group1)!.add(group2); groupGraph.get(group2)!.add(group1);
        }
      }
    });

    const visited = new Set<string>();
    const superGroups = new Map<string, string[]>();
    const taskToSuperGroup = new Map<string, string>();
    const dfs = (node: string, component: string[]) => {
      visited.add(node); component.push(node);
      groupGraph.get(node)?.forEach((neighbor) => { if (!visited.has(neighbor)) { dfs(neighbor, component); } });
    };
    topLevelParents.forEach((parentId) => {
      if (!visited.has(parentId)) {
        const component: string[] = []; dfs(parentId, component); component.sort();
        const representativeId = component[0];
        superGroups.set(representativeId, component);
        component.forEach(memberId => taskToSuperGroup.set(memberId, representativeId));
      }
    });

    // --- START: NEW LAYOUT GENERATION LOGIC ---

    // Step 3: Build the layout plan based on collapsed state
    const pinned = columnsToShow.filter((id) => pinnedTaskIds.has(id)).sort();
    const unpinned = columnsToShow.filter((id) => !pinnedTaskIds.has(id)).sort();
    const sortedColumns = [...pinned, ...unpinned];

    type LayoutItem = { type: 'task'; id: string } | { type: 'group'; representativeId: string; memberIds: string[] };
    const displayedLayout: LayoutItem[] = [];
    const processedTasks = new Set<string>();

    for (const taskId of sortedColumns) {
      if (processedTasks.has(taskId)) continue;

      const topParent = childToTopParent.get(taskId);
      const superGroupRep = topParent ? taskToSuperGroup.get(topParent) : taskId;

      if (superGroupRep && superGroups.has(superGroupRep) && superGroups.get(superGroupRep)!.length > 1) {
        // This task belongs to a multi-item super-group
        if (collapsedGroups.has(superGroupRep)) {
          const memberIds = superGroups.get(superGroupRep)!
            .flatMap(pId => columnsToShow.filter(cId => childToTopParent.get(cId) === pId));
          displayedLayout.push({ type: 'group', representativeId: superGroupRep, memberIds });
          memberIds.forEach(id => processedTasks.add(id));
        } else {
          displayedLayout.push({ type: 'task', id: taskId });
          processedTasks.add(taskId);
        }
      } else {
        // This is a standalone task
        displayedLayout.push({ type: 'task', id: taskId });
        processedTasks.add(taskId);
      }
    }

    // Step 4: Rebuild taskIdToColumn map from the new layout
    const map = new Map<string, number>();
    displayedLayout.forEach((item, index) => {
      const colIndex = index + 1;
      if (item.type === 'task') {
        map.set(item.id, colIndex);
      } else if (item.type === 'group') {
        item.memberIds.forEach(memberId => map.set(memberId, colIndex));
      }
    });

    // --- END: NEW LAYOUT GENERATION LOGIC ---

    const uniqueTimestamps = Array.from(new Set(finalFilteredLog.map(e => e.timestamp))).sort();
    const tsMap = new Map<string, number>();
    uniqueTimestamps.forEach((ts, index) => ts && tsMap.set(ts, index + 3));

    return {
      filteredLog: finalFilteredLog,
      allTaskIdsInLog: allIds,
      displayedLayout: displayedLayout, // <-- Use this for rendering
      taskIdToColumn: map,
      timestampToRowMap: tsMap,
    };
  }, [ globalLog, tasks, focusedTaskId, filterText, visibleTaskIds, pinnedTaskIds, collapsedGroups ]);

  return (
    <div className="bg-zinc-800/50 p-4 rounded-lg border border-zinc-700 h-full flex flex-col">
      {/* Header Bar is correct (no changes) */}
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
          (displayedLayout.length > 0 ? (
            <div
              className="grid gap-x-2 relative"
              style={{
                gridTemplateColumns: `repeat(${displayedLayout.length}, minmax(300px, 1fr))`,
              }}
            >
              {/* --- NEW RENDER LOOP FOR HEADERS AND CONTENT --- */}
              {displayedLayout.map((item, index) => {
                const col = index + 1;
                if (item.type === 'group') {
                  return (
                    <div key={`layout-item-${item.representativeId}`} style={{ gridColumn: col, gridRow: '1 / -1' }} className="flex flex-col">
                       <div className="sticky top-0 bg-zinc-800 z-20 pt-1">
                          <LogGroupHeader parentId={item.representativeId} isCollapsed={true} onToggle={() => toggleGroupCollapse(item.representativeId)} onHover={onHoverTask} />
                       </div>
                       <div className="mt-2 flex-grow min-h-0">
                          <CollapsedGroupContent memberIds={item.memberIds} log={filteredLog} hoveredTaskId={hoveredTaskId} onHoverTask={onHoverTask} />
                       </div>
                    </div>
                  )
                }

                // Item is a task
                const taskId = item.id;
                const taskEntries = filteredLog.filter(entry => entry.taskId === taskId);

                return (
                  <div key={`layout-item-${taskId}`} style={{ gridColumn: col, gridRow: '1 / -1' }} className="flex flex-col">
                    <div className="sticky top-0 bg-zinc-800 z-10 pt-1">
                      <h3
                        className="font-mono text-sm text-zinc-300 truncate font-semibold bg-zinc-900/50 border border-zinc-700 p-2 rounded-md hover:bg-zinc-900 cursor-pointer"
                        onMouseEnter={() => onHoverTask(taskId)}
                        onMouseLeave={() => onHoverTask(null)}
                      >
                        {taskId}
                      </h3>
                    </div>
                    <div className="mt-2">
                       <ExecutionLog log={taskEntries} showTaskId={false} hoveredTaskId={hoveredTaskId} onHoverTask={onHoverTask} />
                    </div>
                  </div>
                )
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