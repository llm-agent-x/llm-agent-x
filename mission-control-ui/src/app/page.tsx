// mission-control-ui/src/app/page.tsx
"use client";

import { useState, useEffect, useMemo, useCallback, useRef } from "react";
import { io, Socket } from "socket.io-client";
import { Download, Upload } from "lucide-react"; // Import new icons
import { TaskList } from "./components/TaskList";
import { TaskInspector } from "./components/TaskInspector";
import { NewTaskForm } from "./components/NewTaskForm";
import { DAGView } from "./components/DAGView";
import { McpServerManager, McpServer } from "./components/McpServerManager";
import { DocumentManager } from "./components/DocumentManager";
import {Task} from "@/lib/types";

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
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
  const [tasks, setTasks] = useState<{ [key: string]: Task }>({});
  const [isConnected, setIsConnected] = useState(false);
  const [mcpServers, setMcpServers] = useState<McpServer[]>([]);

  const fileInputRef = useRef<HTMLInputElement>(null); // Ref for the hidden file input

  // --- Handlers for state management ---

  const handleDownloadState = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/state/download`);
      if (!response.ok) {
        throw new Error(`Failed to download state: ${response.statusText}`);
      }

      // Extract filename from Content-Disposition header
      const disposition = response.headers.get("content-disposition");
      let filename = "graph_state.json"; // Default filename
      if (disposition && disposition.indexOf("attachment") !== -1) {
        const filenameRegex = /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/;
        const matches = filenameRegex.exec(disposition);
        if (matches != null && matches[1]) {
          filename = matches[1].replace(/['"]/g, "");
        }
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Error downloading state:", error);
      alert(`Error: ${(error as Error).message}`);
    }
  };

  const handleUploadClick = () => {
    // Trigger the hidden file input
    fileInputRef.current?.click();
  };

  const handleFileChange = async (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_BASE_URL}/api/state/upload`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          `Failed to upload state: ${errorData.detail || response.statusText}`,
        );
      }

      alert(
        "State file uploaded successfully. The agent will now reset to the new state.",
      );
      // The UI will automatically update via WebSocket broadcasts from the agent.
    } catch (error) {
      console.error("Error uploading state:", error);
      alert(`Error: ${(error as Error).message}`);
    } finally {
      // Reset the file input so the same file can be uploaded again
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  // --- (No changes to the useEffect hooks for localStorage, servers, or sockets) ---
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
  }, []);

  useEffect(() => {
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
    const onTaskUpdate = (data: { task: Task }) => {
      if (data && data.task) {
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

  const taskList = useMemo(() => {
    return Object.values(tasks).sort((a, b) => a.id.localeCompare(b.id));
  }, [tasks]);

  const selectedTask = tasks[selectedTaskId!] || null;

  const handleSelectTask = useCallback((id: string) => {
    setSelectedTaskId(id);
  }, []);

  useEffect(() => {
    if (selectedTaskId && !tasks[selectedTaskId]) {
      setSelectedTaskId(null);
    }
    if (!selectedTaskId && taskList.length > 0) {
      setSelectedTaskId(taskList[0].id);
    }
  }, [tasks, selectedTaskId, taskList]);

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
              className={`w-3 h-3 rounded-full transition-colors ${
                isConnected ? "bg-green-500 animate-pulse" : "bg-red-500"
              }`}
              title={isConnected ? "Connected" : "Disconnected"}
            ></div>
          </div>
          <DocumentManager />
          <McpServerManager
            servers={mcpServers}
            setServers={setMcpServers}
            defaultServers={DEFAULT_SERVERS}
          />

          {/* --- NEW BUTTONS --- */}
          <button
            onClick={handleDownloadState}
            className="inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 h-10 w-10 p-0 border border-zinc-700 bg-zinc-900 hover:bg-zinc-800 hover:text-zinc-100"
            title="Download Graph State"
          >
            <Download className="h-5 w-5" />
          </button>

          <button
            onClick={handleUploadClick}
            className="inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 h-10 w-10 p-0 border border-zinc-700 bg-zinc-900 hover:bg-zinc-800 hover:text-zinc-100"
            title="Upload Graph State"
          >
            <Upload className="h-5 w-5" />
          </button>
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept=".json"
            style={{ display: "none" }}
          />
        </div>
      </header>
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_2fr_1fr] gap-6 h-[calc(100vh-120px)]">
        <div className="flex flex-col h-[calc(100vh-120px)]">
          <div className="flex-grow overflow-y-auto pr-2 bg-zinc-800/50 p-3 rounded-lg border border-zinc-700">
            <TaskList
              tasks={taskList}
              selectedTaskId={selectedTaskId}
              onSelectTask={handleSelectTask}
            />
          </div>
          <div className="flex-shrink-0">
            <NewTaskForm mcpServers={mcpServers} />
          </div>
        </div>
        <div className="lg:col-span-1 h-full">
          <DAGView
            tasks={taskList}
            selectedTaskId={selectedTaskId}
            onSelectTask={handleSelectTask}
          />
        </div>
        <div className="lg:col-span-1 h-full">
          <TaskInspector task={selectedTask} />
        </div>
      </div>
    </main>
  );
}