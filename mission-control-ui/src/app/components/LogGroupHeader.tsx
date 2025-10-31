// mission-control-ui/src/app/components/LogGroupHeader.tsx
"use client";

import { ChevronDown, ChevronRight } from "lucide-react";

interface LogGroupHeaderProps {
  parentId: string;
  isCollapsed: boolean;
  onToggle: () => void;
  onHover: (id: string | null) => void;
}

export const LogGroupHeader = ({
  parentId,
  isCollapsed,
  onToggle,
  onHover,
}: LogGroupHeaderProps) => {
  return (
    <div
      className="flex items-center gap-2 p-2 rounded-md bg-zinc-700/50 border border-zinc-600 cursor-pointer hover:bg-zinc-700 w-full"
      onClick={onToggle}
      onMouseEnter={() => onHover(parentId)}
      onMouseLeave={() => onHover(null)}
    >
      <div className="p-0.5 rounded-sm bg-zinc-800/50">
        {isCollapsed ? <ChevronRight size={16} /> : <ChevronDown size={16} />}
      </div>
      <div className="flex-grow min-w-0">
        <h4 className="text-xs text-zinc-400 font-semibold uppercase">Group</h4>
        <p className="font-mono text-sm text-zinc-200 truncate">{parentId}</p>
      </div>
    </div>
  );
};