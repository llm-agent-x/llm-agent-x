// app/components/CommandPalette.tsx

import { useState, ChangeEvent } from "react";
import { sendDirective, injectDependencyDirective } from "@/lib/api";
import {
  Play,
  Pause,
  X,
  Send,
  MessageSquareQuoteIcon,
  GitMerge, // <-- ADDED ICON
} from "lucide-react";
import { StatusBadge } from "./StatusBadge";
import { CurrentQuestion, Task } from "@/lib/types"; // <-- CORRECTED IMPORTS

// --- DetailRow component ---
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

// --- TaskInspector component ---
export const TaskInspector = ({
  task,
  completedTasks, // <-- ADDED PROP
}: {
  task: Task | null;
  completedTasks: Task[]; // <-- ADDED PROP
}) => {
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

      <CommandPalette
        taskId={task.id}
        taskStatus={task.status}
        currentQuestion={task.current_question}
        completedTasks={completedTasks} // <-- PASS PROP
      />
    </div>
  );
};

// --- CommandPalette Props Interface ---
interface CommandPaletteProps {
  taskId: string;
  taskStatus: string;
  currentQuestion: CurrentQuestion | null;
  completedTasks: Task[];
}

// --- CommandPalette component ---
export const CommandPalette = ({
  taskId,
  taskStatus,
  currentQuestion,
  completedTasks, // <-- DESTRUCTURE PROP
}: CommandPaletteProps) => {
  const [redirectInput, setRedirectInput] = useState("");
  const [answerInput, setAnswerInput] = useState("");
  const [depToInject, setDepToInject] = useState<string>("");

  const handleCommand = async (command: string, payload?: string) => {
    await sendDirective(taskId, command, payload);
    setRedirectInput("");
    setAnswerInput("");
  };

  const handleInject = async (depth: "shallow" | "deep") => {
    if (!depToInject) return;
    await injectDependencyDirective(taskId, depToInject, depth);
    setDepToInject(""); // Reset dropdown
  };

  return (
    <div className="bg-zinc-900/50 border border-zinc-700 p-4 rounded-lg mt-4">
      <h3 className="text-md font-semibold text-zinc-300 mb-4">
        Operator Directives
      </h3>

      {taskStatus === "waiting_for_user_response" && currentQuestion ? (
        // --- START: UI FOR QUESTION STATE ---
        <>
          {/* Section 1: Answer the question directly */}
          <div className="bg-orange-900/30 border border-orange-700 p-3 rounded-md mb-4">
            <div className="flex items-center gap-2 mb-2 text-orange-200">
              <MessageSquareQuoteIcon size={18} />
              <span className="font-semibold">
                Agent Question (Priority: {currentQuestion.priority}/10)
              </span>
            </div>
            <p className="text-sm text-orange-100 mb-3">
              {currentQuestion.question}
            </p>
            <div className="flex gap-2">
              <input
                type="text"
                value={answerInput}
                onChange={(e: ChangeEvent<HTMLInputElement>) =>
                  setAnswerInput(e.target.value)
                }
                placeholder="Type your answer here..."
                className="w-full bg-zinc-800 border border-orange-600 rounded-md p-2 text-sm focus:ring-2 focus:ring-orange-500 focus:border-orange-500 outline-none"
                required
              />
              <button
                onClick={() => handleCommand("ANSWER_QUESTION", answerInput)}
                disabled={!answerInput.trim()}
                className="flex items-center gap-2 p-2 bg-orange-600 hover:bg-orange-700 rounded-md disabled:bg-zinc-600 shrink-0"
              >
                <Send size={16} />
                Submit Answer
              </button>
            </div>
          </div>

          {/* Section 2: Inject a dependency as an alternative */}
          <div className="bg-zinc-800/50 border border-zinc-700 p-3 rounded-md">
            <div className="flex items-center gap-2 mb-2 text-zinc-300">
              <GitMerge size={18} />
              <span className="font-semibold">
                Inject Context from Completed Task
              </span>
            </div>
            <p className="text-xs text-zinc-400 mb-3">
              Provide the result of another task as additional context to help
              the agent.
            </p>
            <div className="flex gap-2">
              <select
                value={depToInject}
                onChange={(e) => setDepToInject(e.target.value)}
                className="w-full bg-zinc-800 border border-zinc-600 rounded-md p-2 text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
              >
                <option value="">-- Select a completed task --</option>
                {completedTasks.map((task) => (
                  <option key={task.id} value={task.id}>
                    {task.id} - {task.desc.slice(0, 50)}...
                  </option>
                ))}
              </select>
            </div>
            {depToInject && (
              <div className="flex gap-2 mt-2">
                <button
                  onClick={() => handleInject("shallow")}
                  className="flex-1 p-2 bg-blue-600 hover:bg-blue-700 rounded-md text-sm"
                >
                  Inject Shallow (Result Only)
                </button>
                <button
                  onClick={() => handleInject("deep")}
                  className="flex-1 p-2 bg-purple-600 hover:bg-purple-700 rounded-md text-sm"
                >
                  Inject Deep (Result + History)
                </button>
              </div>
            )}
          </div>
        </>
      ) : (
        // --- END: UI FOR QUESTION STATE ---

        // --- START: UI FOR OTHER STATES ---
        <div className="grid grid-cols-2 md:grid-cols-3 gap-2 mb-4">
          {taskStatus === "paused_by_human" ? (
            <button
              onClick={() => handleCommand("RESUME")}
              className="flex items-center justify-center gap-2 p-2 bg-green-600 hover:bg-green-700 rounded-md text-sm transition-colors"
            >
              <Play size={16} /> Resume
            </button>
          ) : (
            <button
              onClick={() => handleCommand("PAUSE")}
              className="flex items-center justify-center gap-2 p-2 bg-yellow-600 hover:bg-yellow-700 rounded-md text-sm transition-colors disabled:bg-zinc-600"
              disabled={["complete", "failed"].includes(taskStatus)}
            >
              <Pause size={16} /> Pause
            </button>
          )}
          <button
            onClick={() => handleCommand("TERMINATE", "Operator intervention")}
            className="flex items-center justify-center gap-2 p-2 bg-red-600 hover:bg-red-700 rounded-md text-sm transition-colors disabled:bg-zinc-600"
            disabled={["complete", "failed"].includes(taskStatus)}
          >
            <X size={16} /> Terminate
          </button>
          <button
            onClick={() => handleCommand("CANCEL", "Operator intervention")}
            className="flex items-center justify-center gap-2 p-2 bg-zinc-400 hover:bg-zinc-500 rounded-md text-sm transition-colors disabled:bg-zinc-600"
            disabled={["complete", "failed", "cancelled"].includes(taskStatus)}
          >
            <X size={16} /> Cancel
          </button>
        </div>
      )}
      {/* --- END: UI FOR OTHER STATES --- */}

      <div className="space-y-2 mt-4">
        <label className="text-sm font-medium text-zinc-400">
          Redirect (Corrective Guidance)
        </label>
        <div className="flex gap-2">
          <input
            type="text"
            value={redirectInput}
            onChange={(e: ChangeEvent<HTMLInputElement>) =>
              setRedirectInput(e.target.value)
            }
            placeholder="Describe the correction..."
            className="w-full bg-zinc-800 border border-zinc-700 rounded-md p-2 text-sm focus:ring-2 focus:ring-orange-500 focus:border-orange-500 outline-none"
          />
          <button
            onClick={() => handleCommand("REDIRECT", redirectInput)}
            disabled={!redirectInput.trim()}
            className="p-2 bg-orange-600 hover:bg-orange-700 rounded-md disabled:bg-zinc-600 shrink-0"
          >
            <Send size={16} /> Redirect
          </button>
        </div>
      </div>
    </div>
  );
};