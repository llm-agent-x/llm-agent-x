// app/components/CommandPalette.tsx

import { useState } from 'react';
import { sendDirective } from '@/lib/api';
import { Play, Pause, X, Send, Edit } from 'lucide-react';

export const CommandPalette = ({ taskId, taskStatus }: { taskId: string; taskStatus: string; }) => {
  const [redirectInput, setRedirectInput] = useState('');
  const [overrideInput, setOverrideInput] = useState('');

  const handleCommand = async (command: string, payload?: string) => {
    await sendDirective(taskId, command, payload);
    setRedirectInput('');
    setOverrideInput('');
  };

  return (
    <div className="bg-zinc-900/50 border border-zinc-700 p-4 rounded-lg mt-4">
      <h3 className="text-md font-semibold text-zinc-300 mb-4">Operator Directives</h3>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-2 mb-4">
        {taskStatus === 'paused_by_human' ? (
          <button onClick={() => handleCommand('RESUME')} className="flex items-center justify-center gap-2 p-2 bg-green-600 hover:bg-green-700 rounded-md text-sm transition-colors">
            <Play size={16} /> Resume
          </button>
        ) : (
          <button onClick={() => handleCommand('PAUSE')} className="flex items-center justify-center gap-2 p-2 bg-yellow-600 hover:bg-yellow-700 rounded-md text-sm transition-colors disabled:bg-zinc-600" disabled={['complete', 'failed'].includes(taskStatus)}>
            <Pause size={16} /> Pause
          </button>
        )}
        <button onClick={() => handleCommand('TERMINATE', 'Operator intervention')} className="flex items-center justify-center gap-2 p-2 bg-red-600 hover:bg-red-700 rounded-md text-sm transition-colors disabled:bg-zinc-600" disabled={['complete', 'failed'].includes(taskStatus)}>
          <X size={16} /> Terminate
        </button>
      </div>

      <div className="space-y-2 mb-4">
        <label className="text-sm font-medium text-zinc-400">Redirect (Corrective Guidance)</label>
        <div className="flex gap-2">
            <input type="text" value={redirectInput} onChange={(e) => setRedirectInput(e.target.value)} placeholder="e.g., Focus only on financial figures." className="w-full bg-zinc-800 border border-zinc-600 rounded-md p-2 text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none" />
            <button onClick={() => handleCommand('REDIRECT', redirectInput)} disabled={!redirectInput} className="p-2 bg-blue-600 hover:bg-blue-700 rounded-md disabled:bg-zinc-600 shrink-0">
                <Send size={16}/>
            </button>
        </div>
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium text-zinc-400">Manual Override (Set Result)</label>
         <div className="flex gap-2">
            <textarea value={overrideInput} onChange={(e) => setOverrideInput(e.target.value)} placeholder="Manually enter the final result for this task..." className="w-full bg-zinc-800 border border-zinc-600 rounded-md p-2 text-sm h-24 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none" />
             <button onClick={() => handleCommand('MANUAL_OVERRIDE', overrideInput)} disabled={!overrideInput} className="p-2 bg-purple-600 hover:bg-purple-700 rounded-md disabled:bg-zinc-600 self-start shrink-0">
                <Edit size={16}/>
            </button>
        </div>
      </div>
    </div>
  );
};