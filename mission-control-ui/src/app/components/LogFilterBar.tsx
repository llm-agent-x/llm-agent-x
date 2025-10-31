// mission-control-ui/src/app/components/LogFilterBar.tsx
"use client";

import { Search, X } from "lucide-react";

interface LogFilterBarProps {
  filterText: string;
  onFilterChange: (text: string) => void;
}

export const LogFilterBar = ({
  filterText,
  onFilterChange,
}: LogFilterBarProps) => {
  return (
    <div className="relative w-full">
      <Search
        className="absolute left-3 top-1/2 -translate-y-1/2 text-zinc-500"
        size={18}
      />
      <input
        type="text"
        placeholder="Filter by Task ID or #tag..."
        value={filterText}
        onChange={(e) => onFilterChange(e.target.value)}
        className="w-full bg-zinc-900 border border-zinc-700 rounded-md py-2 pl-10 pr-8 text-sm placeholder:text-zinc-500 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
      />
      {filterText && (
        <button
          onClick={() => onFilterChange("")}
          className="absolute right-2 top-1/2 -translate-y-1/2 p-1 text-zinc-500 hover:text-zinc-200"
          title="Clear filter"
        >
          <X size={16} />
        </button>
      )}
    </div>
  );
};