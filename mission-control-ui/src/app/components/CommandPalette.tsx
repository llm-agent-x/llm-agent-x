// app/components/CommandPalette.tsx

// --- CHANGE 1: Import ChangeEvent for typing input events ---
import { useState, ChangeEvent } from "react";
import { sendDirective } from "@/lib/api";
import {
  Play,
  Pause,
  X,
  Send,
  MessageSquareQuoteIcon,
} from "lucide-react";
import { StatusBadge } from "./StatusBadge";
import {CurrentQuestion} from "@/lib/types";

// --- CHANGE 2: Define strict types for your data structures ---
interface UserQuestion {
  priority: number;
  question: string;
  details: Record<string, unknown>; // Use `unknown` for better type safety than `any`
}

interface Task {
  id: string;
  desc: string;
  status: string;
  deps: string[];
  result: string | null;
  human_directive: string | null;
  current_question: CurrentQuestion | null;
  user_response: string | null;
}

// --- DetailRow component remains the same, its types are already good ---
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

// --- CHANGE 3: Apply the strict `Task` type to the `task` prop ---
export const TaskInspector = ({ task }: { task: Task | null }) => {
  if (!task) {
    return (
      <div className="flex items-center justify-center h-full bg-zinc-800/50 p-4 rounded-lg border border-zinc-700 text-zinc-400">
        Select a task to inspect its details and issue directives.
      </div>
    );
  }

  // No change needed here, this is a safe type guard
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

// --- CHANGE 4: Define a specific props interface for CommandPalette ---
interface CommandPaletteProps {
  taskId: string;
  taskStatus: string;
  currentQuestion: CurrentQuestion | null;
}

export const CommandPalette = ({
  taskId,
  taskStatus,
  currentQuestion,
}: CommandPaletteProps) => {
  const [redirectInput, setRedirectInput] = useState("");
  const [overrideInput, setOverrideInput] = useState("");
  const [answerInput, setAnswerInput] = useState("");

  const handleCommand = async (command: string, payload?: string) => {
    await sendDirective(taskId, command, payload);
    setRedirectInput("");
    setOverrideInput("");
    setAnswerInput("");
  };

  return (
    <div className="bg-zinc-900/50 border border-zinc-700 p-4 rounded-lg mt-4">
      <h3 className="text-md font-semibold text-zinc-300 mb-4">
        Operator Directives
      </h3>

      {taskStatus === "waiting_for_user_response" && currentQuestion ? (
        <div className="bg-orange-900/30 border border-orange-700 p-3 rounded-md mb-4 animate-pulse">
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
              // --- CHANGE 5: Type the `onChange` event handler ---
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
              className="p-2 bg-orange-600 hover:bg-orange-700 rounded-md disabled:bg-zinc-600 shrink-0"
            >
              <Send size={16} /> Submit Answer
            </button>
          </div>
        </div>
      ) : (
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

      <div className="space-y-2 mb-4">
        <label className="text-sm font-medium text-zinc-400">
          Redirect (Corrective Guidance)
        </label>
        <div className="flex gap-2">
          <input
            type="text"
            value={redirectInput}
            // --- CHANGE 6: Type the `onChange` event handler ---
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