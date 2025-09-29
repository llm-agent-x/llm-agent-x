// app/components/NewTaskForm.tsx
"use client";

import { useState, useEffect } from 'react';
import { Send } from 'lucide-react';
import { addTask } from '@/lib/api';
import { McpServer } from './McpServerManager'; // Import the type
import { McpServerSelector } from './McpServerSelector'; // Import the new component

// Define props for the component
interface NewTaskFormProps {
  mcpServers: McpServer[];
}

export const NewTaskForm = ({ mcpServers }: NewTaskFormProps) => {
  const [description, setDescription] = useState('');
  // State to hold the IDs of the selected servers for this task
  const [selectedServerIds, setSelectedServerIds] = useState<string[]>([]);

  // Effect to pre-select all available servers by default
  useEffect(() => {
    setSelectedServerIds(mcpServers.map(s => s.id));
  }, [mcpServers]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (description.trim() && selectedServerIds.length > 0) {
      // Pass the selected server IDs to the API call
      await addTask(description, selectedServerIds);
      setDescription('');
      // We don't reset the selection, as the user might want to launch similar tasks
    }
  };

  const isSubmitDisabled = !description.trim() || selectedServerIds.length === 0;

  return (
    <div className="mt-4 p-3 bg-zinc-800/70 rounded-lg border border-zinc-700">
      <h3 className="text-md font-semibold text-zinc-300 mb-2">Launch New Task</h3>
      <form onSubmit={handleSubmit}>
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

        {/* --- ADD THE NEW SERVER SELECTOR HERE --- */}
        <McpServerSelector
            allServers={mcpServers}
            selectedServerIds={selectedServerIds}
            onSelectionChange={setSelectedServerIds}
        />
        {/* --- END SERVER SELECTOR --- */}

      </form>
    </div>
  );
};