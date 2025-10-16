// mission-control-ui/src/app/components/McpServerManager.tsx
"use client";

import { useState, useRef, useEffect, ReactNode } from "react";
import { Server, PlusCircle, Trash2, X, RotateCcw } from "lucide-react";

export interface McpServer {
  id: string;
  name: string;
  address: string;
  type: "sse" | "streamable_http";
}

// --- REMOVED SELF-CONTAINED UI COMPONENTS ---
// The custom <Button> is gone. We use standard <button> with classes.
interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  className?: string;
}

const Input = ({
  className = "",
  ...props
}: InputProps) => (
  <input
    className={`flex h-10 w-full rounded-md border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 focus:ring-offset-zinc-900 ${className}`}
    {...props}
  />
);
const Label = ({
  children,
  ...props
}: {
  children: ReactNode;
} & React.LabelHTMLAttributes<HTMLLabelElement>) => (
  <label className="text-sm font-medium leading-none text-zinc-400" {...props}>
    {children}
  </label>
);

// --- NEW: Base classes for all buttons to keep styling consistent ---
const baseButtonClasses =
  "inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 disabled:opacity-50 disabled:pointer-events-none";

interface McpServerManagerProps {
  servers: McpServer[];
  setServers: (
    servers: McpServer[] | ((prev: McpServer[]) => McpServer[]),
  ) => void;
  defaultServers: McpServer[];
}

export function McpServerManager({
  servers,
  setServers,
  defaultServers,
}: McpServerManagerProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [isVisible, setIsVisible] = useState(false);
  const [isConfirmingReset, setIsConfirmingReset] = useState(false);
  const [newServerName, setNewServerName] = useState("");
  const [newServerAddress, setNewServerAddress] = useState("");
  const [newServerType, setNewServerType] = useState<"sse" | "streamable_http">(
    "sse",
  );
  const drawerRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const TRANSITION_DURATION = 300;

  const openDrawer = () => {
    setIsOpen(true);
    setTimeout(() => {
      setIsVisible(true);
    }, 10);
  };
  const closeDrawer = () => {
    setIsVisible(false);
    setTimeout(() => {
      setIsOpen(false);
      setIsConfirmingReset(false);
    }, TRANSITION_DURATION);
  };
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        closeDrawer();
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, []);
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        drawerRef.current &&
        !drawerRef.current.contains(event.target as Node) &&
        triggerRef.current &&
        !triggerRef.current.contains(event.target as Node)
      ) {
        closeDrawer();
      }
    };
    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [isOpen]);

  const handleAddServer = () => {
    if (!newServerAddress.trim()) {
      return;
    }
    const newServer: McpServer = {
      id: Date.now().toString(),
      name: newServerName.trim(),
      address: newServerAddress.trim(),
      type: newServerType,
    };
    setServers((prev) => [...prev, newServer]);
    setNewServerAddress("");
    setNewServerName("");
  };
  const handleRemoveServer = (idToRemove: string) => {
    setServers((prev) => prev.filter((server) => server.id !== idToRemove));
  };
  const handleConfirmReset = () => {
    setServers(defaultServers);
    setIsConfirmingReset(false);
  };
  const handleCancelReset = () => {
    setIsConfirmingReset(false);
  };
  useEffect(() => {
    setIsConfirmingReset(false);
  }, [servers]);

  return (
    <>
      <button
        ref={triggerRef}
        onClick={openDrawer}
        className={`${baseButtonClasses} h-10 w-10 p-0 border border-zinc-700 bg-zinc-900 hover:bg-zinc-800 hover:text-zinc-100`}
        title="Manage MCP Servers"
      >
        <Server className="h-5 w-5" />
      </button>

      {isOpen && (
        <div className="fixed inset-0 z-50">
          <div
            onClick={closeDrawer}
            className={`fixed inset-0 bg-black/60 backdrop-blur-sm transition-opacity duration-${TRANSITION_DURATION} ease-in-out ${isVisible ? "opacity-100" : "opacity-0"}`}
          />
          <div
            ref={drawerRef}
            className={`fixed top-0 right-0 h-full w-full max-w-md bg-zinc-950 border-l border-zinc-800 text-zinc-100 flex flex-col shadow-2xl transition-transform duration-${TRANSITION_DURATION} ease-in-out ${isVisible ? "translate-x-0" : "translate-x-full"}`}
          >
            <div className="flex items-center justify-between p-6 border-b border-zinc-800">
              <div>
                <h2 className="text-xl font-semibold text-zinc-100">
                  Manage MCP Servers
                </h2>
                <p className="text-sm text-zinc-400">
                  Add or remove servers from the network.
                </p>
              </div>
              <button
                onClick={closeDrawer}
                className={`${baseButtonClasses} h-8 w-8 p-0 bg-transparent hover:bg-zinc-800 text-zinc-400 hover:text-zinc-100`}
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            <div className="flex-grow overflow-y-auto p-6">
              <div className="flex flex-col gap-3">
                {servers.map((server) => (
                  <div
                    key={server.id}
                    className="flex items-center justify-between p-3 bg-zinc-900/70 border border-zinc-800 rounded-md"
                  >
                    <span
                      className="font-mono text-sm text-zinc-300 truncate rounded-md"
                      style={{ maxWidth: "16ch" }}
                    >
                      {server.name}
                    </span>
                    <div className="flex flex-col">
                      <span className="font-mono text-sm text-zinc-300">
                        {server.address}
                      </span>
                      <span className="text-xs uppercase bg-zinc-700 text-zinc-300 px-2 py-0.5 rounded-full w-fit mt-1">
                        {server.type.replace("_", " ")}
                      </span>
                    </div>
                    <button
                      onClick={() => handleRemoveServer(server.id)}
                      className={`${baseButtonClasses} h-8 w-8 p-0 text-zinc-400 hover:bg-red-900/50 hover:text-red-400`}
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                ))}
                {servers.length === 0 && (
                  <p className="text-zinc-500 text-center py-8">
                    No MCP servers configured.
                  </p>
                )}
              </div>
            </div>
            <div className="flex-shrink-0 p-6 border-t border-zinc-800 bg-zinc-950">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold text-zinc-200">
                  Add New Server
                </h3>
                {!isConfirmingReset ? (
                  <button
                    onClick={() => setIsConfirmingReset(true)}
                    className={`${baseButtonClasses} px-3 py-1 text-xs border border-zinc-700 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100`}
                  >
                    <RotateCcw className="mr-2 h-3 w-3" />
                    Reset Defaults
                  </button>
                ) : (
                  <div className="flex items-center gap-2" role="alert">
                    <span className="text-sm text-zinc-300">Are you sure?</span>
                    <button
                      onClick={handleCancelReset}
                      className={`${baseButtonClasses} px-3 py-1 text-xs border border-zinc-700 bg-zinc-800 text-zinc-300 hover:bg-zinc-700 hover:text-zinc-100`}
                    >
                      Cancel
                    </button>
                    <button
                      onClick={handleConfirmReset}
                      className={`${baseButtonClasses} px-3 py-1 text-xs border border-red-800 bg-red-900/80 text-red-300 hover:bg-red-800/80 hover:border-red-700`}
                    >
                      Confirm
                    </button>
                  </div>
                )}
              </div>
              <div className="grid gap-4">
                <div className="grid w-full items-center gap-1.5">
                  <Label htmlFor="server-name">Server Name</Label>
                  <Input
                    type="text"
                    id="server-name"
                    placeholder="My MCP Server"
                    value={newServerName}
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) => setNewServerName(e.target.value)}
                  />
                </div>
                <div className="grid w-full items-center gap-1.5">
                  <Label htmlFor="server-address">Server Address</Label>
                  <Input
                    type="text"
                    id="server-address"
                    placeholder="http://localhost:8080/mcp"
                    value={newServerAddress}
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) => setNewServerAddress(e.target.value)}
                  />
                </div>
                <div className="grid w-full items-center gap-1.5">
                  <Label htmlFor="server-type">Server Type</Label>
                  <select
                    id="server-type"
                    value={newServerType}
                    onChange={(e) =>
                      setNewServerType(
                        e.target.value as "sse" | "streamable_http",
                      )
                    }
                    className="h-10 w-full rounded-md border border-zinc-700 bg-zinc-800 px-3 text-sm text-zinc-100 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 focus:ring-offset-zinc-900"
                  >
                    <option value="streamable_http">Streamable HTTP</option>
                    <option value="sse">SSE (Server-Sent Events)</option>
                  </select>
                </div>
              </div>
              <button
                onClick={handleAddServer}
                className={`${baseButtonClasses} w-full mt-4 h-10 bg-indigo-600 hover:bg-indigo-700 text-white`}
              >
                <PlusCircle className="mr-2 h-4 w-4" /> Add Server
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
