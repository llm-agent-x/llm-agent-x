// app/components/TaskInspector.tsx

import { StatusBadge } from "./StatusBadge";
import { CommandPalette } from "./CommandPalette";

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

export const TaskInspector = ({ task }: { task: any | null }) => {
  if (!task) {
    return (
      <div className="flex items-center justify-center h-full bg-zinc-800/50 p-4 rounded-lg border border-zinc-700 text-zinc-400">
        Select a task to inspect its details and issue directives.
      </div>
    );
  }

  const depsArray = Array.isArray(task.deps) ? task.deps : [];

  return (
    <div className="bg-zinc-800/50 p-4 rounded-lg border border-zinc-700 h-full overflow-y-auto">
      <div className="flex justify-between items-start mb-1">
        <h2 className="text-xl font-bold text-zinc-100 pr-4">{task.desc}</h2>
        <StatusBadge status={task.status} />
      </div>
      <p className="font-mono text-xs text-zinc-500 mb-4">{task.id}</p>

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
        <DetailRow
          label="Result"
          value={
            <pre className="whitespace-pre-wrap font-sans bg-zinc-900/80 p-2 rounded-md max-h-48 overflow-y-auto">
              {task.result || "Not available"}
            </pre>
          }
        />
        {task.human_directive && (
          <DetailRow
            label="Active Human Directive"
            value={
              <span className="bg-blue-900/50 text-blue-300 p-2 rounded-md block animate-pulse">
                {task.human_directive}
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

      <CommandPalette
        taskId={task.id}
        taskStatus={task.status}
        currentQuestion={task.current_question}
      />
    </div>
  );
};
