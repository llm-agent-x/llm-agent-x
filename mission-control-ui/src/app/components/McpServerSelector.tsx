// src/app/components/McpServerSelector.tsx
"use client";

import { useState } from 'react';
import { McpServer } from './McpServerManager';
import { ChevronDown } from 'lucide-react';

interface McpServerSelectorProps {
  allServers: McpServer[];
  selectedServerIds: string[];
  onSelectionChange: (newSelectedIds: string[]) => void;
}

export const McpServerSelector = ({ allServers, selectedServerIds, onSelectionChange }: McpServerSelectorProps) => {
  const [isOpen, setIsOpen] = useState(false); // Controls the collapsible panel

  const handleCheckboxChange = (serverId: string) => {
    const newSelection = selectedServerIds.includes(serverId)
      ? selectedServerIds.filter(id => id !== serverId)
      : [...selectedServerIds, serverId];
    onSelectionChange(newSelection);
  };

  // Select/Deselect all servers
  const handleSelectAll = () => {
    if (selectedServerIds.length === allServers.length) {
      onSelectionChange([]); // Deselect all
    } else {
      onSelectionChange(allServers.map(s => s.id)); // Select all
    }
  };

  return (
    <div className="mb-2">
      {/* --- TRIGGER BUTTON --- */}
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex justify-between items-center bg-zinc-900 border border-zinc-700 rounded-md p-2 text-sm text-zinc-300 hover:bg-zinc-800 transition-colors"
        aria-expanded={isOpen}
      >
        <span>
          Targeting {selectedServerIds.length} of {allServers.length} Servers
        </span>
        <ChevronDown size={18} className={`transition-transform duration-300 ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {/* --- COLLAPSIBLE CONTENT --- */}
      <div
        className={`grid transition-all duration-300 ease-in-out overflow-hidden ${
          isOpen ? 'grid-rows-[1fr] opacity-100' : 'grid-rows-[0fr] opacity-0'
        }`}
      >
        <div className="overflow-hidden"> {/* This inner div is required for the grid transition to work smoothly */}
            <div className="mt-2 p-2 border border-zinc-700 rounded-md bg-zinc-900/50">
              <div className="max-h-40 overflow-y-auto pr-1">
                {allServers.map(server => (
                  <label
                    key={server.id}
                    className="flex items-center gap-3 p-2 text-sm rounded-md hover:bg-zinc-800 cursor-pointer"
                  >
                    <input
                      type="checkbox"
                      checked={selectedServerIds.includes(server.id)}
                      onChange={() => handleCheckboxChange(server.id)}
                      className="h-4 w-4 rounded bg-zinc-700 border-zinc-600 text-blue-500 focus:ring-blue-600 cursor-pointer"
                    />
                    <div className="flex flex-col">
                      <span className="text-zinc-200">{server.name}</span>
                      <span className="text-zinc-500 text-xs font-mono">{server.address}</span>
                    </div>
                  </label>
                ))}
                {allServers.length === 0 && <div className="p-2 text-sm text-zinc-500">No servers configured.</div>}
              </div>
              {allServers.length > 0 && (
                 <div className="pt-2 mt-2 border-t border-zinc-700">
                    <button type="button" onClick={handleSelectAll} className="w-full text-center text-xs text-zinc-400 hover:text-blue-400">
                        {selectedServerIds.length === allServers.length ? 'Deselect All' : 'Select All'}
                    </button>
                 </div>
              )}
            </div>
        </div>
      </div>
    </div>
  );
};