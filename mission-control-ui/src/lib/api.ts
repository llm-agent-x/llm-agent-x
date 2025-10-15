// lib/api.ts

// Import the McpServer type for type safety and clarity.
// Adjust the path if your component is located elsewhere.
import type { McpServer } from "@/app/components/McpServerManager";

const API_BASE_URL = "http://localhost:8000";

/**
 * Submits a new root task to the agent swarm.
 * @param desc - The task description.
 * @param mcp_servers - An array of selected MCP server objects to be used for the task.
 */
export async function addTask(desc: string, mcp_servers: McpServer[]) {
  // <-- UPDATED SIGNATURE
  if (!desc.trim()) {
    console.error("Task description cannot be empty.");
    return;
  }
  try {
    const response = await fetch(`${API_BASE_URL}/api/tasks`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      // The body now includes the description and the array of server objects.
      body: JSON.stringify({ desc, mcp_servers }), // <-- UPDATED BODY
    });
    if (!response.ok) {
      throw new Error(`Failed to add task: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error("Error adding task:", error);
    // Avoid using alert() in modern applications if possible, but keeping it as it was in your original file.
    alert(`Error: ${(error as Error).message}`);
  }
}

export interface Document {
  id: string;
  name: string;
  content: string;
}

export async function fetchDocuments() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/documents`);
    if (!response.ok) {
      throw new Error(`Failed to fetch documents: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error("Error fetching documents:", error);
    alert(`Error: ${(error as Error).message}`);
  }
}

export async function addDocument(document: Document) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/documents`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(document),
    });
    if (!response.ok) {
      throw new Error(`Failed to add document: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error("Error adding document:", error);
    alert(`Error: ${(error as Error).message}`);
  }
}

export async function updateDocument(document: Document) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/documents/${document.id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(document),
    });
    if (!response.ok) {
      throw new Error(`Failed to update document: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error("Error updating document:", error);
    alert(`Error: ${(error as Error).message}`);
  }
}

export async function deleteDocument(documentId: string) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/documents/${documentId}`, {
      method: "DELETE",
    });
    if (!response.ok) {
      throw new Error(`Failed to delete document: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error("Error deleting document:", error);
    alert(`Error: ${(error as Error).message}`);
  }
}

/**
 * Sends a specific directive (PAUSE, REDIRECT, etc.) to a running task.
 */
export async function sendDirective(
  taskId: string,
  command: string,
  payload?: string,
) {
  try {
    const response = await fetch(
      `${API_BASE_URL}/api/tasks/${taskId}/directive`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ command: command.toUpperCase(), payload }),
      },
    );
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(
        `Failed to send directive: ${response.status} ${errorText}`,
      );
    }
    return await response.json();
  } catch (error) {
    console.error("Error sending directive:", error);
    alert(`Error: ${(error as Error).message}`);
  }
}
