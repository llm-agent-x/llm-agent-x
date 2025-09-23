// lib/api.ts

const API_BASE_URL = "http://localhost:8000";

/**
 * Submits a new root task to the agent swarm.
 */
export async function addTask(desc: string) {
  if (!desc.trim()) {
    console.error("Task description cannot be empty.");
    return;
  }
  try {
    const response = await fetch(`${API_BASE_URL}/api/tasks`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ desc }),
    });
    if (!response.ok) {
      throw new Error(`Failed to add task: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error("Error adding task:", error);
    alert(`Error: ${error.message}`);
  }
}

/**
 * Sends a specific directive (PAUSE, REDIRECT, etc.) to a running task.
 */
export async function sendDirective(taskId: string, command: string, payload?: string) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/tasks/${taskId}/directive`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ command: command.toUpperCase(), payload }),
    });
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Failed to send directive: ${response.status} ${errorText}`);
    }
    return await response.json();
  } catch (error) {
    console.error("Error sending directive:", error);
    alert(`Error: ${error.message}`);
  }
}