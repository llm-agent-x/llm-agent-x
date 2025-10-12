// mission-control-ui/src/app/page.tsx
"use client";

// ... (all imports remain the same)
import { useState, useEffect } from "react";
import { io, Socket } from "socket.io-client";
import { TaskList } from "./components/TaskList";
import { TaskInspector } from "./components/TaskInspector";
import { NewTaskForm } from "./components/NewTaskForm";
import { DAGView } from "./components/DAGView";
import { McpServerManager, McpServer } from "./components/McpServerManager";

const API_BASE_URL = "http://localhost:8000";

const DEFAULT_SERVERS: McpServer[] = [
  {
    id: "1",
    name: "Test MCP Server 1",
    address: "http://localhost:8081/mcp",
    type: "sse",
  },
  {
    id: "2",
    name: "Test MCP Server 2",
    address: "http://localhost:8082/mcp",
    type: "streamable_http",
  },
];
const LOCAL_STORAGE_KEY = "mcpServers";

export default function MissionControl() {
  // ... (all state and useEffect hooks remain the same)
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
  const [tasks, setTasks] = useState<{ [key: string]: any }>({});
  const [isConnected, setIsConnected] = useState(false);
  const [mcpServers, setMcpServers] = useState<McpServer[]>([]);

  // On initial mount, load servers from localStorage
  useEffect(() => {
    try {
      const saved = window.localStorage.getItem(LOCAL_STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        setMcpServers(Array.isArray(parsed) ? parsed : DEFAULT_SERVERS);
      } else {
        setMcpServers(DEFAULT_SERVERS);
      }
    } catch (error) {
      console.error("Failed to parse MCP servers from localStorage:", error);
      setMcpServers(DEFAULT_SERVERS);
    }
  }, []); // Empty array ensures this runs only once on mount

  // When servers change, save them back to localStorage
  useEffect(() => {
    // We check if the initial load is done to avoid overwriting on first render
    if (mcpServers.length > 0) {
      try {
        window.localStorage.setItem(
          LOCAL_STORAGE_KEY,
          JSON.stringify(mcpServers),
        );
      } catch (error) {
        console.error("Failed to save MCP servers to localStorage:", error);
      }
    }
  }, [mcpServers]);
  // --- END MCP SERVER LOGIC ---

  useEffect(() => {
    const socket: Socket = io(API_BASE_URL);

    const onConnect = () => {
      console.log("Connected to gateway!");
      setIsConnected(true);
      fetch(`${API_BASE_URL}/api/tasks`)
        .then((res) => res.json())
        .then((data) => {
          console.log("Fetched initial state:", data);
          setTasks(data.tasks || {});
        })
        .catch((err) => console.error("Failed to fetch initial state:", err));
    };

    const onDisconnect = () => {
      console.log("Disconnected from gateway.");
      setIsConnected(false);
    };

    const onTaskUpdate = (data: { task: any }) => {
      if (data && data.task) {
        // console.log("Received task update:", data.task.id, data.task.status);
        setTasks((prevTasks) => ({
          ...prevTasks,
          [data.task.id]: data.task,
        }));
      }
    };

    socket.on("connect", onConnect);
    socket.on("disconnect", onDisconnect);
    socket.on("task_update", onTaskUpdate);
    return () => {
      socket.disconnect();
    };
  }, []);

  const taskList = Object.values(tasks).sort((a, b) =>
    a.id.localeCompare(b.id),
  );
  const selectedTask = tasks[selectedTaskId!] || null;

  useEffect(() => {
    if (selectedTaskId && !tasks[selectedTaskId]) {
      setSelectedTaskId(null);
    }
    if (!selectedTaskId && taskList.length > 0) {
      setSelectedTaskId(taskList[0].id);
    }
  }, [tasks, selectedTaskId, taskList]);

  // Now you can use `mcpServers` anywhere in this component or pass it to children
  useEffect(() => {
    console.log("Current MCP Servers:", mcpServers);
  }, [mcpServers]);

  return (
    <main className="bg-zinc-900 text-white min-h-screen p-4 md:p-6 lg:p-8 max-h-screen">
      <header className="mb-6 flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-zinc-100">
            Collaborative Intelligence Swarm
          </h1>
          <p className="text-zinc-400">Operator: Strategic Cortex</p>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-sm text-zinc-400">
            <span>Gateway Status</span>
            <div
              className={`w-3 h-3 rounded-full transition-colors ${isConnected ? "bg-green-500 animate-pulse" : "bg-red-500"}`}
              title={isConnected ? "Connected" : "Disconnected"}
            ></div>
          </div>
          {/* Pass state and setters down as props */}
          <McpServerManager
            servers={mcpServers}
            setServers={setMcpServers}
            defaultServers={DEFAULT_SERVERS}
          />
        </div>
      </header>
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_2fr_1fr] gap-6 h-[calc(100vh-120px)]">
        <div className="flex flex-col h-[calc(100vh-120px)]">
          <div className="flex-grow overflow-y-auto pr-2 bg-zinc-800/50 p-3 rounded-lg border border-zinc-700">
            <TaskList
              tasks={taskList}
              selectedTaskId={selectedTaskId}
              onSelectTask={setSelectedTaskId}
            />
          </div>
          <div className="flex-shrink-0">
            {/* --- PASS MCP SERVERS TO THE FORM --- */}
            <NewTaskForm mcpServers={mcpServers} />
          </div>
        </div>
        <div className="lg:col-span-1 h-full">
          <DAGView
            tasks={taskList}
            selectedTaskId={selectedTaskId}
            onSelectTask={setSelectedTaskId}
          />
        </div>
        <div className="lg:col-span-1 h-full">
          <TaskInspector task={selectedTask} />
        </div>
      </div>
    </main>
  );
}
