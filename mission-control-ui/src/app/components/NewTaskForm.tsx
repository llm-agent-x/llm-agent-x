// app/components/NewTaskForm.tsx
"use client";

import { useState } from 'react';
import { Send } from 'lucide-react';
import { addTask } from '@/lib/api';

export const NewTaskForm = () => {
  const [description, setDescription] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (description.trim()) {
      await addTask(description);
      setDescription('');
    }
  };

  return (
    <div className="mt-4 p-3 bg-zinc-800/70 rounded-lg border border-zinc-700">
      <h3 className="text-md font-semibold text-zinc-300 mb-2">Launch New Task</h3>
      <form onSubmit={handleSubmit} className="flex gap-2">
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
          disabled={!description.trim()}
          className="p-2 bg-blue-600 hover:bg-blue-700 rounded-md text-white disabled:bg-zinc-600 disabled:cursor-not-allowed transition-colors shrink-0"
          aria-label="Submit new task"
        >
          <Send size={18} />
        </button>
      </form>
    </div>
  );
};