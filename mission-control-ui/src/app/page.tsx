// mission-control-ui/src/app/page.tsx
"use client";

import { useState, useEffect } from 'react';
import { io, Socket } from "socket.io-client";
import { TaskList } from './components/TaskList';
import { TaskInspector } from './components/TaskInspector';
import { NewTaskForm } from './components/NewTaskForm';
import { DAGView } from './components/DAGView';
import { McpServerManager } from './components/McpServerManager'; // Import the new component

const API_BASE_URL = "http://localhost:8000";

export default function MissionControl() {
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
  const [tasks, setTasks] = useState<{ [key: string]: any }>({});
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    const socket: Socket = io(API_BASE_URL);

    const onConnect = () => {
        console.log("Connected to gateway!");
        setIsConnected(true);
        fetch(`${API_BASE_URL}/api/tasks`)
            .then(res => res.json())
            .then(data => {
                console.log("Fetched initial state:", data);
                setTasks(data.tasks || {});
            }).catch(err => console.error("Failed to fetch initial state:", err));
    };

    const onDisconnect = () => {
        console.log("Disconnected from gateway.");
        setIsConnected(false);
    };

    const onTaskUpdate = (data: { task: any }) => {
        if (data && data.task) {
            // console.log("Received task update:", data.task.id, data.task.status);
            setTasks(prevTasks => ({
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

  const taskList = Object.values(tasks).sort((a,b) => a.id.localeCompare(b.id));
  const selectedTask = tasks[selectedTaskId!] || null;

  useEffect(() => {
      // If the selected task is removed or no longer exists, deselect it
      if (selectedTaskId && !tasks[selectedTaskId]) {
          setSelectedTaskId(null);
      }
      // If there's no selected task but tasks exist, select the first one
      if (!selectedTaskId && taskList.length > 0) {
        setSelectedTaskId(taskList[0].id);
      }
  }, [tasks, selectedTaskId, taskList]);

  return (
    <main className="bg-zinc-900 text-white min-h-screen p-4 md:p-6 lg:p-8 max-h-screen">
      <header className="mb-6 flex justify-between items-center">
        <div>
            <h1 className="text-3xl font-bold text-zinc-100">Collaborative Intelligence Swarm</h1>
            <p className="text-zinc-400">Operator: Strategic Cortex</p>
        </div>
        <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-sm text-zinc-400">
                <span>Gateway Status</span>
                <div className={`w-3 h-3 rounded-full transition-colors ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} title={isConnected ? 'Connected' : 'Disconnected'}></div>
            </div>
            {/* Use the new component here */}
            <McpServerManager />
        </div>
      </header>
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_2fr_1fr] gap-6 h-[calc(100vh-120px)]">
        {/* Left Column: Task List & New Task Form */}
        <div className="flex flex-col h-[calc(100vh-120px)]">
          <div className="flex-grow overflow-y-auto pr-2 bg-zinc-800/50 p-3 rounded-lg border border-zinc-700">
            <TaskList
              tasks={taskList}
              selectedTaskId={selectedTaskId}
              onSelectTask={setSelectedTaskId}
            />
          </div>
          <div className="flex-shrink-0">
            <NewTaskForm />
          </div>
        </div>

        {/* Middle Column: DAG Visualization */}
        <div className="lg:col-span-1 h-full">
            <DAGView
                tasks={taskList}
                selectedTaskId={selectedTaskId}
                onSelectTask={setSelectedTaskId}
            />
        </div>

        {/* Right Column: Task Inspector */}
        <div className="lg:col-span-1 h-full">
          <TaskInspector task={selectedTask} />
        </div>
      </div>
    </main>
  );
}