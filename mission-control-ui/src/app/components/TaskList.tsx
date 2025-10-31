// app/components/TaskList.tsx

import { StatusBadge } from "./StatusBadge";
import { Task } from "@/lib/types";

interface TaskListProps {
  tasks: Task[];
  selectedTaskId: string | null;
  onSelectTask: (id: string) => void;
  hoveredTaskId: string | null;
  onHoverTask: (id: string | null) => void;
}

export const TaskList = ({
  tasks,
  selectedTaskId,
  onSelectTask,
  hoveredTaskId,
  onHoverTask,
}: TaskListProps) => {
  if (tasks.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-zinc-500 text-center p-4 max-w-[20rem]">
        <p className="text-lg">Swarm is Idle</p>
        <p className="text-sm">
          No tasks in the registry. Use the form below to launch a new task.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2 h-full w-full">
      <h2 className="text-lg font-bold text-zinc-300 px-2 mb-2">Task Swarm</h2>
      {tasks.map((task) => {
        const isSelected = selectedTaskId === task.id;
        const isHovered = hoveredTaskId === task.id;

        let backgroundClass = "hover:bg-zinc-700/70";
        if (isSelected) {
            backgroundClass = "bg-blue-600/30";
        } else if (isHovered) {
            backgroundClass = "bg-yellow-900/30";
        }

        return (
          <button
            key={task.id}
            onClick={() => onSelectTask(task.id)}
            onMouseEnter={() => onHoverTask(task.id)}
            onMouseLeave={() => onHoverTask(null)}
            className={`w-full text-left p-2.5 rounded-lg transition-colors border ${isSelected ? "border-blue-500" : isHovered ? "border-yellow-700/50" : "border-transparent"} ${backgroundClass}`}
            style={{
              overflowWrap: "break-word",
              wordWrap: "break-word",
            }}
          >
            <div className="flex justify-between items-center mb-1">
              <p className="font-mono text-sm text-zinc-400 truncate">
                {task.id}
              </p>
              <StatusBadge status={task.status} />
            </div>
            <p className="text-zinc-200 text-sm whitespace-nowrap overflow-hidden text-ellipsis">
              {task.desc}
            </p>
          </button>
        );
      })}
    </div>
  );
};