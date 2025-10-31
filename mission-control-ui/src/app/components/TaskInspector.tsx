// app/components/TaskInspector.tsx

import { format, parseISO } from "date-fns";
import { StatusBadge } from "./StatusBadge";
import { CommandPalette } from "./CommandPalette";
import {Task} from "@/lib/types";

// The DetailRow component is already well-typed, no changes needed
const DetailRow = ({
  label,
  value,
}: {
  label: string;
  value: React.ReactNode;
}) => (
  <div className="py-3 border-b border-zinc-700/50">
    <dt className="text-sm font-medium text-zinc-400">{label}</dt>
    <dd className="mt-1 text-sm text-zinc-200 break-words">{value}</dd>
  </div>
);

// --- CHANGE 2: Apply the strict `Task` type to the `task` prop in DocumentDetails ---
const DocumentDetails = ({ task }: { task: Task }) => {
  if (!task.document_state) {
    return (
      <DetailRow
        label="Document Content"
        value={
          <pre className="whitespace-pre-wrap font-sans text-red-400">
            Error: Document state is missing.
          </pre>
        }
      />
    );
  }

  const { content, version, updated_at } = task.document_state;
  const formattedDate = updated_at
    ? format(parseISO(updated_at), "yyyy-MM-dd HH:mm:ss 'UTC'")
    : "Unknown date";

  return (
    <>
      <DetailRow
        label="Version History"
        value={
          <div className="flex items-center gap-2 text-xs">
            <span className="bg-zinc-700 px-2 py-0.5 rounded-md font-mono">
              Version {version}
            </span>
            <span className="text-zinc-400">Last updated: {formattedDate}</span>
          </div>
        }
      />
      <DetailRow
        label="Document Content"
        value={
          <pre className="whitespace-pre-wrap font-sans bg-zinc-900/80 p-3 rounded-md max-h-96 overflow-y-auto">
            {content || "Document is empty."}
          </pre>
        }
      />
    </>
  );
};

// --- CHANGE 3: Apply the strict `Task` type to the `task` prop in TaskInspector ---
export const TaskInspector = ({ task, completedTasks }: { task: Task | null, completedTasks: Task[] }) => {
  if (!task) {
    return (
      <div className="flex items-center justify-center h-full bg-zinc-800/50 p-4 rounded-lg border border-zinc-700 text-zinc-400">
        Select a task to inspect its details and issue directives.
      </div>
    );
  }

  // This type guard is still useful and correct
  const depsArray = Array.isArray(task.deps) ? task.deps : [];

  return (
    <div className="bg-zinc-800/50 p-4 rounded-lg border border-zinc-700 h-full flex flex-col gap-4">
      {/* Header Section */}
      <div>
        <div className="flex justify-between items-start mb-1">
          <h2 className="text-xl font-bold text-zinc-100 pr-4">{task.desc}</h2>
          <StatusBadge status={task.status} />
        </div>
        <p className="font-mono text-xs text-zinc-500">{task.id}</p>
      </div>

      {task.tags && task.tags.length > 0 && (
        <DetailRow
          label="Tags"
          value={
            <div className="flex flex-wrap gap-2">
              {task.tags.map((tag) => (
                <span
                  key={tag}
                  className="px-2 py-0.5 text-xs font-mono rounded-full bg-zinc-700 text-zinc-300"
                >
                  #{tag}
                </span>
              ))}
            </div>
          }
        />
      )}
      <div className="flex-grow overflow-y-auto min-h-0 pr-2 -mr-2">
        <dl>
          <DetailRow
            label="Dependencies"
            value={
              depsArray.length > 0 ? (
                <pre className="font-mono text-xs">{depsArray.join("\n")}</pre>
              ) : (
                "None"
              )
            }
          />

          {task.task_type === "document" ? (
            <DocumentDetails task={task} />
          ) : (
            <DetailRow
              label="Result"
              value={
                <pre className="whitespace-pre-wrap font-sans bg-zinc-900/80 p-2 rounded-md max-h-48 overflow-y-auto">
                  {task.result || "Not available"}
                </pre>
              }
            />
          )}

          {task.human_directive && (
            <DetailRow
              label="Active Human Directive"
              value={
                <span className="bg-blue-900/50 text-blue-300 p-2 rounded-md block animate-pulse">
                  {task.human_directive.instruction}
                </span>
              }
            />
          )}
          {task.current_question && (
            <DetailRow
              label="Agent's Question"
              value={
                <span className="bg-orange-900/50 text-orange-300 p-2 rounded-md block animate-pulse">
                  Priority {task.current_question.priority}/10:{" "}
                  {task.current_question.question}
                </span>
              }
            />
          )}
          {task.user_response && (
            <DetailRow
              label="Your Last Response"
              value={
                <span className="bg-green-900/50 text-green-300 p-2 rounded-md block">
                  {task.user_response}
                </span>
              }
            />
          )}
        </dl>
      </div>

      {/* Footer Command Palette Section */}
      <div className="flex-shrink-0">
        <CommandPalette
          taskId={task.id}
          taskStatus={task.status}
          currentQuestion={task.current_question}
          completedTasks={completedTasks}
        />
      </div>
    </div>
  );
};