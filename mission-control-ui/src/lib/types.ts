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
}