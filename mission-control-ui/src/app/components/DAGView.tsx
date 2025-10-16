// mission-control-ui/src/app/components/DAGView.tsx
"use client";

import React, { useEffect, useRef } from "react"; // Import useRef
import ReactFlow, {
    MiniMap,
    Controls,
    Background,
    useNodesState,
    useEdgesState,
    MarkerType,
    Node,
    Edge,
    Handle,
    Position,
    Panel, BackgroundVariant,
} from "reactflow";
import dagre from "dagre";
import "reactflow/dist/style.css";
import {CurrentQuestion, HumanDirective, Task} from "@/lib/types";

// ============================================================================
// ** 1. ALL COMPONENTS AND CONSTANTS REMAIN AT THE TOP LEVEL **
// ============================================================================

// --- (StatusBadge, CustomTaskNode, nodeTypes, getLayoutedElements are unchanged)
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



interface CustomTaskNodeData {
  id: string;
  desc: string;
  status: string;
  human_directive: HumanDirective | null;
  current_question: CurrentQuestion | null;
}
const CustomTaskNode = ({ data, selected }: { data: CustomTaskNodeData, selected: boolean }) => {
  return (
    <div
      className={`px-4 py-2 shadow-md rounded-lg border-2 ${
        selected ? "border-blue-500" : "border-zinc-700"
      } bg-zinc-800 w-[250px]`}
    >
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
      {data.human_directive?.instruction && (
        <div className="mt-2 text-xs text-blue-300 bg-blue-900/20 p-1 rounded-md">
          Directive: {data.human_directive.instruction.slice(0, 50)}...
        </div>
      )}
      {data.current_question?.question && (
        <div className="mt-2 text-xs text-yellow-300 bg-yellow-900/20 p-1 rounded-md">
          Question: {data.current_question.question.slice(0, 50)}...
        </div>
      )}
      <Handle type="source" position={Position.Bottom} className="!bg-zinc-500" />
    </div>
  );
};
const nodeTypes = { customTaskNode: CustomTaskNode };

const getLayoutedElements = (nodes: Node[], edges: Edge[], direction = 'TB') => {
  const dagreGraph = new dagre.graphlib.Graph();
  dagreGraph.setDefaultEdgeLabel(() => ({}));
  dagreGraph.setGraph({ rankdir: direction, nodesep: 50, ranksep: 100 });
  nodes.forEach((node) => {
    dagreGraph.setNode(node.id, { width: node.width, height: node.height });
  });
  edges.forEach((edge) => {
    dagreGraph.setEdge(edge.source, edge.target);
  });
  dagre.layout(dagreGraph);
  const layoutedNodes = nodes.map((node) => {
    const nodeWithPosition = dagreGraph.node(node.id);
    return {
      ...node,
      position: {
        x: nodeWithPosition.x - (node.width! / 2),
        y: nodeWithPosition.y - (node.height! / 2),
      },
    };
  });
  return { nodes: layoutedNodes, edges };
};

// ============================================================================
// ** 2. THE MAIN DAGVIEW COMPONENT WITH THE ROBUST `useRef` FIX **
// ============================================================================

interface DAGViewProps {
  tasks: Task[];
  selectedTaskId: string | null;
  onSelectTask: (id: string) => void;
}
export const DAGView: React.FC<DAGViewProps> = ({ tasks, selectedTaskId, onSelectTask }) => {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [layoutDirection, setLayoutDirection] = React.useState('TB');
  // **THE FIX**: A ref to track if the current `tasks` have been laid out.
  const layouted = useRef(false);

  // **PASS 1: Create initial nodes**
  useEffect(() => {
    if (!tasks) return;
    // When the tasks change, reset the layout flag.
    layouted.current = false;

    const initialNodes: Node<CustomTaskNodeData>[] = tasks.map((task) => ({
      id: task.id,
      selected: task.id === selectedTaskId,
      position: { x: 0, y: 0 },
      type: "customTaskNode",
      data: {
        id: task.id,
        desc: task.desc || '', // Provide default empty string if desc is undefined
        status: task.status || '', // Provide default empty string if status is undefined
        human_directive: task.human_directive ?? null, // Convert undefined to null
        current_question: task.current_question ?? null, // Convert undefined to null
      },
    }));

    // --- START OF LOGICAL FIX ---
    const initialEdges: Edge[] = [];
    const taskIds = new Set(tasks.map(t => t.id));

    tasks.forEach((task) => {
      // 1. Create edges for parent -> child relationships (structural hierarchy)
      if (task.parent && taskIds.has(task.parent)) {
        initialEdges.push({
          id: `e-parent-${task.parent}-${task.id}`,
          source: task.parent,
          target: task.id,
          type: 'default',
          markerEnd: { type: MarkerType.ArrowClosed, color: "#6b7280" }, // Gray for hierarchy
          style: { stroke: "#6b7280", strokeWidth: 1, strokeDasharray: '5 5' }, // Dashed for hierarchy
        });
      }

      // 2. Create edges for dependency -> task relationships (data flow)
      if (task.deps && Array.isArray(task.deps)) {
        task.deps.forEach((depId: string) => {
          if (taskIds.has(depId)) {
            initialEdges.push({
              id: `e-dep-${depId}-${task.id}`,
              source: depId,
              target: task.id,
              type: 'default',
              markerEnd: { type: MarkerType.ArrowClosed, color: "#a3a3a3" }, // Lighter for data flow
              style: { stroke: "#a3a3a3", strokeWidth: 2 }, // Solid, thicker for data flow
            });
          }
        });
      }
    });
    setNodes(initialNodes);
    setEdges(initialEdges);
  }, [tasks, setNodes, setEdges]);

  // **PASS 2: Calculate layout after nodes are measured**
  useEffect(() => {
    // GUARD CLAUSE: Only run layout if nodes exist, have been measured,
    // and have NOT been laid out yet for this set of tasks.
    if (nodes.length === 0 || !nodes[0].width || layouted.current) {
      return;
    }
    const { nodes: finalNodes, edges: finalEdges } = getLayoutedElements(
      nodes, edges, layoutDirection
    );
    setNodes(finalNodes);
    setEdges(finalEdges);
    // Mark this set as laid out to break the loop.
    layouted.current = true;
  }, [nodes, edges, layoutDirection, setNodes, setEdges]);

  return (
    <div className="flex-grow h-full bg-zinc-800/50 rounded-lg border border-zinc-700">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        fitView
        proOptions={{ hideAttribution: true }}
        selectionOnDrag
        panOnDrag={[1, 2]}
        onNodeClick={(_, node) => onSelectTask(node.id)}
      >
        <MiniMap nodeStrokeWidth={3} zoomable pannable />
        <Controls />
        <Background variant={BackgroundVariant.Dots} gap={12} size={1} />
        <Panel
          position="top-right"
          className="bg-zinc-900/70 p-2 rounded-md shadow-lg border border-zinc-700"
        >
          <h3 className="text-zinc-200 text-sm font-semibold mb-2">Layout</h3>
          <div className="flex gap-2">
            <button onClick={() => { layouted.current = false; setLayoutDirection('TB'); }}
              className={`px-3 py-1 text-xs rounded-md ${layoutDirection === 'TB' ? 'bg-blue-600' : 'bg-zinc-600 hover:bg-zinc-500'}`} >
              Top-Bottom
            </button>
            <button onClick={() => { layouted.current = false; setLayoutDirection('LR'); }}
              className={`px-3 py-1 text-xs rounded-md ${layoutDirection === 'LR' ? 'bg-blue-600' : 'bg-zinc-600 hover:bg-zinc-500'}`} >
              Left-Right
            </button>
          </div>
        </Panel>
      </ReactFlow>
    </div>
  );
};