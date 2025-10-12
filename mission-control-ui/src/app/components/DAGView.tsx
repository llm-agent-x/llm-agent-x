// mission-control-ui/src/app/components/DAGView.tsx
import React, { useCallback, useEffect, useState } from "react";
import ReactFlow, {
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  MarkerType,
  Node,
  Edge,
  Panel,
} from "reactflow";
import dagre from "dagre";
import "reactflow/dist/style.css";
import { StatusBadge } from "./StatusBadge";

interface CustomTaskNodeData {
  id: string;
  desc: string;
  status: string;
  selectedTaskId: string | null;
  human_directive: any; // Use a more specific type if known
  current_question: any; // Use a more specific type if known
  onSelectTask: (id: string) => void; // This is the function we want
}
// Custom Node Component to display task info
const CustomTaskNode = ({
  data,
  id,
  selected,
}: {
  data: CustomTaskNodeData;
  id: string;
  selected: boolean;
}) => {
  const isSelected = selected || id === data.selectedTaskId;
  return (
    <div
      className={`px-4 py-2 shadow-md rounded-lg border transition-all duration-200 cursor-pointer ${
        isSelected
          ? "border-blue-500 ring-2 ring-blue-500 bg-blue-900/40"
          : "border-zinc-700 bg-zinc-800/60 hover:bg-zinc-700/70"
      }`}
      onClick={() => data.onSelectTask(id)} // Corrected: Access data.onSelectTask
    >
      <div className="flex justify-between items-center mb-1">
        <div className="text-sm font-mono text-zinc-400 truncate max-w-[120px]">
          {data.id}
        </div>
        <StatusBadge status={data.status} />
      </div>
      <div className="text-zinc-100 text-sm whitespace-pre-wrap max-w-[200px] font-semibold">
        {data.desc}
      </div>
      {data.human_directive && (
        <div className="mt-2 text-xs text-blue-300 bg-blue-900/20 p-1 rounded-md animate-pulse">
          Directive: {data.human_directive.slice(0, 50)}...
        </div>
      )}
      {data.current_question && (
        <div className="mt-2 text-xs text-yellow-300 bg-yellow-900/20 p-1 rounded-md animate-pulse">
          Question: {data.current_question.question.slice(0, 50)}...
        </div>
      )}
    </div>
  );
};

const nodeTypes = {
  customTaskNode: CustomTaskNode,
};

// Dagre layout setup
const dagreGraph = new dagre.graphlib.Graph();
dagreGraph.setDefaultEdgeLabel(() => ({}));

const nodeWidth = 250;
const nodeHeight = 100;

const getLayoutedElements = (
  nodes: Node[],
  edges: Edge[],
  direction = "TB",
) => {
  dagreGraph.setGraph({ rankdir: direction, nodesep: 50, ranksep: 50 });

  nodes.forEach((node) => {
    dagreGraph.setNode(node.id, { width: nodeWidth, height: nodeHeight });
  });

  edges.forEach((edge) => {
    dagreGraph.setEdge(edge.source, edge.target);
  });

  dagre.layout(dagreGraph);

  const layoutedNodes = nodes.map((node) => {
    const nodeWithPosition = dagreGraph.node(node.id);
    // We need to pass a negative position to React Flow for proper rendering
    node.position = {
      x: nodeWithPosition.x - nodeWidth / 2,
      y: nodeWithPosition.y - nodeHeight / 2,
    };
    return node;
  });

  return { nodes: layoutedNodes, edges };
};

interface DAGViewProps {
  tasks: any[];
  selectedTaskId: string | null;
  onSelectTask: (id: string) => void;
}

export const DAGView: React.FC<DAGViewProps> = ({
  tasks,
  selectedTaskId,
  onSelectTask,
}) => {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [layoutDirection, setLayoutDirection] = useState<"TB" | "LR">("TB"); // Top-Bottom or Left-Right

  const proOptions = { hideAttribution: true, showConnections: true }; // Show connections
  useEffect(() => {
    const newNodes: Node[] = [];
    const newEdges: Edge[] = [];

    // Add debug logging to see task structure
    console.log(
      "Tasks for DAG:",
      tasks.map((t) => ({ id: t.id, deps: t.deps, children: t.children })),
    );

    tasks.forEach((task) => {
      newNodes.push({
        id: task.id,
        position: { x: 0, y: 0 }, // Position will be set by dagre
        data: {
          id: task.id,
          desc: task.desc,
          status: task.status,
          selectedTaskId: selectedTaskId,
          human_directive: task.human_directive,
          current_question: task.current_question,
          onSelectTask: onSelectTask,
        },
        type: "customTaskNode",
      });

      // Add dependencies as edges
      (task.deps || []).forEach((dep: any) => {
        // Debug log each dependency
        console.log(`Processing dep for task ${task.id}:`, dep, typeof dep);

        // Extract the task_id from the dependency object
        const depId = typeof dep === "string" ? dep : dep?.task_id || dep?.id;

        console.log(`Extracted depId: ${depId}`);

        // CRITICAL FIX: Check if depId is a valid string and not "undefined"
        if (
          depId &&
          typeof depId === "string" &&
          depId !== "undefined" &&
          tasks.some((t) => t.id === depId)
        ) {
          console.log(`Creating edge from ${depId} to ${task.id}`);
          newEdges.push({
            id: `e${depId}-${task.id}`,
            source: depId,
            target: task.id,
            type: "default",
            markerEnd: {
              type: MarkerType.ArrowClosed,
              color: "#a3a3a3", // zinc-400
            },
            style: { stroke: "#a3a3a3", strokeWidth: 1.5 },
          });
        } else {
          console.log(
            `Skipping edge: depId="${depId}", type=${typeof depId}, exists=${depId ? tasks.some((t) => t.id === depId) : false}`,
          );
        }
      });

      // Add parent-child relationships as edges
      (task.children || []).forEach((child: any) => {
        console.log(`Processing child for task ${task.id}:`, child);

        // Extract child ID - handle both string and object
        const childId =
          typeof child === "string" ? child : child?.task_id || child?.id;

        console.log(`Extracted childId: ${childId}`);

        // CRITICAL FIX: Same check for children
        if (
          childId &&
          typeof childId === "string" &&
          childId !== "undefined" &&
          tasks.some((t) => t.id === childId)
        ) {
          console.log(`Creating child edge from ${childId} to ${task.id}`);
          newEdges.push({
            id: `e${task.id}-${childId}-child`,
            source: childId,
            target: task.id,
            type: "default",
            markerEnd: {
              type: MarkerType.ArrowClosed,
              color: "#60a5fa", // blue-400
            },
            style: {
              stroke: "#60a5fa",
              strokeWidth: 1.5,
              strokeDasharray: "5,5",
            },
            data: { label: "Synthesizes from" },
          });
        } else {
          console.log(
            `Skipping child edge: childId="${childId}", type=${typeof childId}, exists=${childId ? tasks.some((t) => t.id === childId) : false}`,
          );
        }
      });
    });

    console.log("Created edges:", newEdges);

    const { nodes: layoutedNodes, edges: layoutedEdges } = getLayoutedElements(
      newNodes,
      newEdges,
      layoutDirection,
    );

    setNodes(layoutedNodes);
    setEdges(layoutedEdges);
  }, [
    tasks,
    selectedTaskId,
    layoutDirection,
    setNodes,
    setEdges,
    onSelectTask,
  ]);

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
        panOnDrag={[1, 2]} // Allow pan with left click
      >
        <MiniMap nodeStrokeWidth={3} zoomable pannable />
        <Controls />
        <Background variant="dots" gap={12} size={1} />
        <Panel
          position="top-right"
          className="bg-zinc-900/70 p-2 rounded-md shadow-lg border border-zinc-700"
        >
          <h3 className="text-zinc-200 text-sm font-semibold mb-2">Layout</h3>
          <div className="flex gap-2">
            <button
              onClick={() => setLayoutDirection("TB")}
              className={`px-3 py-1 text-xs rounded-md ${layoutDirection === "TB" ? "bg-blue-600" : "bg-zinc-600 hover:bg-zinc-500"}`}
            >
              Top-Bottom
            </button>
            <button
              onClick={() => setLayoutDirection("LR")}
              className={`px-3 py-1 text-xs rounded-md ${layoutDirection === "LR" ? "bg-blue-600" : "bg-zinc-600 hover:bg-zinc-500"}`}
            >
              Left-Right
            </button>
          </div>
        </Panel>
      </ReactFlow>
    </div>
  );
};
