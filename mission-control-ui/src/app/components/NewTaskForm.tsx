"use client";

import { useState, useEffect } from "react";
import { Send, Bot } from "lucide-react"; // Import Bot icon
import { addTask, fetchAvailableAgents } from "@/lib/api"; // Import fetch
import { McpServer } from "./McpServerManager";
import { McpServerSelector } from "./McpServerSelector";

interface NewTaskFormProps {
  mcpServers: McpServer[];
}

export const NewTaskForm = ({ mcpServers }: NewTaskFormProps) => {
  const [description, setDescription] = useState("");
  const [selectedServerIds, setSelectedServerIds] = useState<string[]>([]);

  // --- NEW STATE ---
  const [availableAgents, setAvailableAgents] = useState<string[]>([]);
  const [selectedAgent, setSelectedAgent] = useState<string>("");

  useEffect(() => {
    setSelectedServerIds(mcpServers.map((s) => s.id));
  }, [mcpServers]);

  // --- FETCH AGENTS ON MOUNT ---
  useEffect(() => {
    const loadAgents = async () => {
      try {
        const data = await fetchAvailableAgents();
        setAvailableAgents(data.agents);
        setSelectedAgent(data.default);
      } catch (e) {
        console.error("Failed to load agents", e);
        // Fallback
        setAvailableAgents(["interactive_dag"]);
        setSelectedAgent("interactive_dag");
      }
    };
    loadAgents();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (description.trim()) {
      const selectedServers = mcpServers.filter((server) =>
          selectedServerIds.includes(server.id)
        );
      // --- PASS SELECTED AGENT ---
      await addTask(description, selectedServers, selectedAgent);
      setDescription("");
    }
  };

  const isSubmitDisabled = !description.trim();

  return (
    <div className="mt-4 p-3 bg-zinc-800/70 rounded-lg border border-zinc-700">
      <h3 className="text-md font-semibold text-zinc-300 mb-2">
        Launch New Task
      </h3>
      <form onSubmit={handleSubmit} className="flex flex-col gap-3">

        {/* --- AGENT TYPE SELECTOR --- */}
        {availableAgents.length > 1 && (
          <div className="flex items-center gap-2 bg-zinc-900 border border-zinc-700 rounded-md px-2">
            <Bot size={16} className="text-zinc-400" />
            <select
              value={selectedAgent}
              onChange={(e) => setSelectedAgent(e.target.value)}
              className="w-full bg-transparent text-sm text-zinc-200 py-2 outline-none cursor-pointer"
            >
              {availableAgents.map((agent) => (
                <option key={agent} value={agent} className="bg-zinc-800">
                  {agent.replace("_", " ").toUpperCase()}
                </option>
              ))}
            </select>
          </div>
        )}

        <McpServerSelector
          allServers={mcpServers}
          selectedServerIds={selectedServerIds}
          onSelectionChange={setSelectedServerIds}
        />

        <div className="flex gap-2">
          <input
            type="text"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder={`Enter objective for ${selectedAgent.replace("_", " ")}...`}
            className="w-full bg-zinc-900 border border-zinc-700 rounded-md p-2 text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
            required
          />
          <button
            type="submit"
            disabled={isSubmitDisabled}
            className="p-2 bg-blue-600 hover:bg-blue-700 rounded-md text-white disabled:bg-zinc-600 disabled:cursor-not-allowed transition-colors shrink-0"
            aria-label="Submit new task"
          >
            <Send size={18} />
          </button>
        </div>
      </form>
    </div>
  );
};