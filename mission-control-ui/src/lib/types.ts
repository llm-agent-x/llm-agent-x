interface UserQuestion {
  priority: number;
  question: string;
  details: Record<string, unknown>;
}

export interface HumanDirective {
  instruction?: string;
  // Add other properties of human_directive here if they exist
  [key: string]: unknown;
}

export interface CurrentQuestion {
  question?: string;
  // Add other properties of current_question here if they exist
  priority?: number;
  details: Record<string, unknown>;
  [key: string]: unknown;
}

interface DocumentState {
  content: string;
  version: number;
  updated_at: string; // ISO date string
  content_hash: string;
}

export interface ExecutionLogEntry {
  type: 'thought' | 'tool_call' | 'tool_result' | 'final_answer' | 'error';
  content?: string;
  tool_name?: string;
  args?: Record<string, unknown>;
  result?: unknown; // More type-safe than any, but still flexible
}

export interface Task {
  id: string;
  desc: string;
  status: string;
  deps: string[];
  result: string | null;
  human_directive: HumanDirective | null;
  current_question: CurrentQuestion | null;
  user_response: string | null;
  task_type?: "task" | "document"; // The `?` makes it optional for regular tasks
  document_state?: DocumentState | null; // Optional and can be null
  parent?: string;

  execution_log: ExecutionLogEntry[];
}