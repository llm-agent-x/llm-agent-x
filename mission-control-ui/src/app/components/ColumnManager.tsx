// mission-control-ui/src/app/components/ColumnManager.tsx
"use client";

import { useState } from "react";
import { SlidersHorizontal, Pin, PinOff } from "lucide-react";

interface ColumnManagerProps {
  allTaskIds: string[];
  visibleTaskIds: Set<string> | null;
  onToggleVisibility: (id: string) => void;
  pinnedTaskIds: Set<string>;
  onTogglePin: (id: string) => void;
}

export const ColumnManager = ({
  allTaskIds,
  visibleTaskIds,
  onToggleVisibility,
  pinnedTaskIds,
  onTogglePin,
}: ColumnManagerProps) => {
  const [isOpen, setIsOpen] = useState(false);

  const isVisible = (id: string) =>
    visibleTaskIds === null || visibleTaskIds.has(id);

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 p-2 rounded-md bg-zinc-700 text-zinc-200 hover:bg-zinc-600"
        title="Manage Columns"
      >
        <SlidersHorizontal size={16} />
      </button>

      {isOpen && (
        <>
          <div
            className="fixed inset-0 z-10"
            onClick={() => setIsOpen(false)}
          />
          <div className="absolute right-0 top-full mt-2 w-72 bg-zinc-800 border border-zinc-700 rounded-lg shadow-xl z-20 p-2">
            <h4 className="text-sm font-semibold text-zinc-300 px-2 pb-2 border-b border-zinc-700">
              Manage Columns
            </h4>
            <div className="max-h-80 overflow-y-auto mt-2 space-y-1 pr-1">
              {allTaskIds.map((id) => (
                <div
                  key={id}
                  className="flex items-center justify-between p-2 rounded-md hover:bg-zinc-700"
                >
                  <label className="flex items-center gap-2 text-sm text-zinc-200 cursor-pointer flex-grow truncate">
                    <input
                      type="checkbox"
                      checked={isVisible(id)}
                      onChange={() => onToggleVisibility(id)}
                      className="h-4 w-4 rounded bg-zinc-700 border-zinc-600 text-blue-500 focus:ring-blue-600 cursor-pointer"
                    />
                    <span className="font-mono truncate">{id}</span>
                  </label>
                  <button
                    onClick={() => onTogglePin(id)}
                    className="p-1 text-zinc-400 hover:text-white rounded-md"
                    title={pinnedTaskIds.has(id) ? "Unpin" : "Pin"}
                  >
                    {pinnedTaskIds.has(id) ? (
                      <Pin size={16} className="text-blue-400" />
                    ) : (
                      <PinOff size={16} />
                    )}
                  </button>
                </div>
              ))}
              {allTaskIds.length === 0 && (
                <p className="text-xs text-zinc-500 text-center p-4">
                  No tasks in log.
                </p>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
};