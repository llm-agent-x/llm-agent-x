"use client";

import { useEffect, useRef } from "react";
import { ExecutionLogEntry } from "@/lib/types";
import {
  Cog,
  Wrench,
  ClipboardCheck,
  Sparkles,
  AlertTriangle,
} from "lucide-react";

const iconMap = {
  thought: <Cog size={16} className="text-zinc-400" />,
  tool_call: <Wrench size={16} className="text-blue-400" />,
  tool_result: <ClipboardCheck size={16} className="text-green-400" />,
  final_answer: <Sparkles size={16} className="text-yellow-400" />,
  error: <AlertTriangle size={16} className="text-red-400" />,
};

const LogEntry = ({ entry }: { entry: ExecutionLogEntry }) => {
  const icon = iconMap[entry.type] || <Cog size={16} />;

  return (
    <div className="flex items-start gap-3 py-2 px-1 border-b border-zinc-800/50">
      <div className="flex-shrink-0 pt-1">{icon}</div>
      <div className="flex-grow min-w-0">
        <p className="text-xs font-semibold capitalize text-zinc-300">
          {entry.type.replace("_", " ")}
        </p>
        {entry.type === "thought" && (
          <p className="text-sm text-zinc-400">{entry.content}</p>
        )}
        {entry.type === "tool_call" && (
          <div className="text-sm">
            <p className="font-semibold text-blue-300">{entry.tool_name}</p>
            <pre className="mt-1 text-xs bg-zinc-900 p-2 rounded-md overflow-x-auto text-zinc-400">
              {JSON.stringify(entry.args, null, 2)}
            </pre>
          </div>
        )}
        {entry.type === "tool_result" && (
          <div className="text-sm">
            <p className="font-semibold text-green-300">{entry.tool_name}</p>
            <pre className="mt-1 text-xs bg-zinc-900 p-2 rounded-md overflow-x-auto text-zinc-400">
              {JSON.stringify(entry.result, null, 2)}
            </pre>
          </div>
        )}
        {entry.type === "final_answer" && (
          <p className="text-sm font-semibold text-yellow-300">
            {entry.content}
          </p>
        )}
        {entry.type === "error" && (
          <p className="text-sm font-mono bg-red-900/50 text-red-300 p-2 rounded-md">
            {entry.content}
          </p>
        )}
      </div>
    </div>
  );
};

export const ExecutionLog = ({ log }: { log: ExecutionLogEntry[] }) => {
  const logEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [log]);

  if (!log || log.length === 0) {
    return (
      <div className="text-center text-zinc-500 py-8">
        No execution steps logged for this task yet.
      </div>
    );
  }

  return (
    <div className="font-mono text-sm">
      {log.map((entry, index) => (
        <LogEntry key={index} entry={entry} />
      ))}
      <div ref={logEndRef} />
    </div>
  );
};