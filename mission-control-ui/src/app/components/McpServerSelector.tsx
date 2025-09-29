// src/app/components/McpServerSelector.tsx
"use client";

import { useState, useRef, useEffect } from 'react';
import { McpServer } from './McpServerManager'; // Re-use the interface
import { ChevronDown, Check } from 'lucide-react';

interface McpServerSelectorProps {
  allServers: McpServer[];
  selectedServerIds: string[];
  onSelectionChange: (newSelectedIds: string[]) => void;
}

export const McpServerSelector = ({ allServers, selectedServerIds, onSelectionChange }: McpServerSelectorProps) => {
  const [isOpen, setIsOpen] = useState(false);
  const wrapperRef = useRef<HTMLDivElement>(null);

  // Close dropdown on outside click
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (wrapperRef.current && !wrapperRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleCheckboxChange = (serverId: string) => {
    const newSelection = selectedServerIds.includes(serverId)
      ? selectedServerIds.filter(id => id !== serverId)
      : [...selectedServerIds, serverId];
    onSelectionChange(newSelection);
  };

  return (
    <div className="relative" ref={wrapperRef}>
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="w-full mt-2 flex justify-between items-center bg-zinc-900 border border-zinc-700 rounded-md p-2 text-sm text-zinc-300 hover:bg-zinc-800 transition-colors"
      >
        <span>
          Selected {selectedServerIds.length} of {allServers.length} Servers
        </span>
        <ChevronDown size={18} className={`transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {isOpen && (
        <div className="absolute z-10 w-full mt-1 bg-zinc-900 border border-zinc-700 rounded-md shadow-lg max-h-48 overflow-y-auto">
          {allServers.map(server => (
            <label
              key={server.id}
              className="flex items-center gap-3 p-2 text-sm hover:bg-zinc-800 cursor-pointer"
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
      )}
    </div>
  );
};