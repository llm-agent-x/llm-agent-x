// mission-control-ui/src/app/components/DAGView.tsx
"use client";

import React, { useEffect } from "react";
import ReactFlow, {
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  MarkerType,
  Node,
  Edge,
  Handle,     // Import Handle
  Position,   // Import Position
} from "reactflow";
import "reactflow/dist/style.css";

// ============================================================================
// ** 1. DEFINE ALL COMPONENTS AND CONSTANTS AT THE TOP LEVEL (OUTSIDE) **
// ============================================================================

// --- StatusBadge Component ---
export const StatusBadge = ({ status }: { status: string }) => {
  const statusStyles: { [key: string]: string } = {
    running: "bg-blue-500/20 text-blue-300 border-blue-400/30",
    complete: "bg-green-500/20 text-green-300 border-green-400/30",
    failed: "bg-red-500/20 text-red-300 border-red-400/30",
    pending: "bg-yellow-500/20 text-yellow-300 border-yellow-400/30",
  };
  const defaultStyle = "bg-zinc-500/20 text-zinc-300 border-zinc-400/30";
  const style = statusStyles[status] || defaultStyle;
  return (
    <span
      className={`px-2 py-0.5 text-xs font-medium rounded-full border ${style}`}
    >
      {status}
    </span>
  );
};


// --- CustomTaskNode Component (with Handles) ---
interface CustomTaskNodeData {
  id: string;
  desc: string;
  status: string;
  human_directive: any;
  current_question: any;
}

const CustomTaskNode = ({ data, selected }: { data: CustomTaskNodeData, selected: boolean }) => {
  return (
    <div
      className={`px-4 py-2 shadow-md rounded-lg border-2 ${
        selected ? "border-blue-500" : "border-zinc-700"
      } bg-zinc-800 w-[250px]`} // Set a fixed width
    >
      {/* Target handle on top (edges can connect TO this) */}
      <Handle type="target" position={Position.Top} className="!bg-zinc-500" />

      <div className="flex justify-between items-center mb-1">
        <div className="text-sm font-mono text-zinc-400 truncate max-w-[120px]">
          {data.id}
        </div>
        <StatusBadge status={data.status} />
      </div>
      <div className="text-zinc-100 text-sm whitespace-pre-wrap font-semibold">
        {data.desc}
      </div>
      {data.human_directive && (
        <div className="mt-2 text-xs text-blue-300 bg-blue-900/20 p-1 rounded-md">
          Directive: {data.human_directive.slice(0, 50)}...
        </div>
      )}
      {data.current_question && (
        <div className="mt-2 text-xs text-yellow-300 bg-yellow-900/20 p-1 rounded-md">
          Question: {data.current_question.question.slice(0, 50)}...
        </div>
      )}

      {/* Source handle on bottom (edges can connect FROM this) */}
      <Handle type="source" position={Position.Bottom} className="!bg-zinc-500" />
    </div>
  );
};

// --- Stable, top-level constant for nodeTypes ---
const nodeTypes = {
  customTaskNode: CustomTaskNode,
};

// ============================================================================
// ** 2. THE MAIN DAGVIEW COMPONENT IS NOW CLEAN AND RELIABLE **
// ============================================================================

interface DAGViewProps {
  tasks: any[];
  // `selectedTaskId` and `onSelectTask` are no longer needed for the graph itself,
  // but you can keep them if other parts of your UI depend on them.
  // We'll remove them here for simplicity as the graph is self-contained.
}

export const DAGView: React.FC<DAGViewProps> = ({ tasks }) => {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  const proOptions = { hideAttribution: true };

  useEffect(() => {
    if (!tasks) return;

    // We'll create a simple grid layout manually. No more dagre.
    const newNodes: Node<CustomTaskNodeData>[] = tasks.map((task, index) => ({
      id: task.id,
      position: { x: (index % 4) * 300, y: Math.floor(index / 4) * 200 }, // Simple grid
      type: "customTaskNode",
      data: {
        id: task.id,
        desc: task.desc,
        status: task.status,
        human_directive: task.human_directive,
        current_question: task.current_question,
      },
    }));

    const newEdges: Edge[] = [];
    tasks.forEach((task) => {
      // Create an edge only if the parent exists in the provided tasks.
      if (task.parent && tasks.some(t => t.id === task.parent)) {
        newEdges.push({
          id: `e${task.parent}-${task.id}`,
          source: task.parent,
          target: task.id,
          type: 'default', // Using a default edge for simplicity
          markerEnd: { type: MarkerType.ArrowClosed, color: "#a3a3a3" },
          style: { stroke: "#a3a3a3", strokeWidth: 1.5 },
        });
      }
    });

    setNodes(newNodes);
    setEdges(newEdges);
  }, [tasks, setNodes, setEdges]); // Depend only on the stable `tasks` prop.

  return (
    <div className="flex-grow h-full bg-zinc-800/50 rounded-lg border border-zinc-700">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        fitView
        proOptions={proOptions}
        selectionOnDrag
        panOnDrag={[1, 2]}
      >
        <MiniMap nodeStrokeWidth={3} zoomable pannable />
        <Controls />
        <Background variant="dots" gap={12} size={1} />
        {/* We can remove the layout panel since the layout is now automatic */}
      </ReactFlow>
    </div>
  );
};