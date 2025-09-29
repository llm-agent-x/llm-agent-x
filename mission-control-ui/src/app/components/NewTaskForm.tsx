// app/components/NewTaskForm.tsx
"use client";

import { useState, useEffect } from 'react';
import { Send } from 'lucide-react';
import { addTask } from '@/lib/api';
import { McpServer } from './McpServerManager';
import { McpServerSelector } from './McpServerSelector';

interface NewTaskFormProps {
  mcpServers: McpServer[];
}

export const NewTaskForm = ({ mcpServers }: NewTaskFormProps) => {
  const [description, setDescription] = useState('');
  const [selectedServerIds, setSelectedServerIds] = useState<string[]>([]);

  useEffect(() => {
    setSelectedServerIds(mcpServers.map(s => s.id));
  }, [mcpServers]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (description.trim()) {
      const selectedServers = selectedServerIds.length > 0
        ? mcpServers.filter(server => selectedServerIds.includes(server.id))
        : mcpServers;
      await addTask(description, selectedServers);
      setDescription('');
    }
  };

  const isSubmitDisabled = !description.trim();

  return (
    <div className="mt-4 p-3 bg-zinc-800/70 rounded-lg border border-zinc-700">
      <h3 className="text-md font-semibold text-zinc-300 mb-2">Launch New Task</h3>
      <form onSubmit={handleSubmit}>

        {/* --- MOVED THE SELECTOR TO BE ABOVE THE INPUT --- */}
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
              placeholder="Enter a new objective for the swarm..."
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