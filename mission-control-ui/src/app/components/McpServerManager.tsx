// mission-control-ui/src/app/components/McpServerManager.tsx
"use client";

import { useState, useRef, useEffect, ReactNode } from 'react';
import { Server, PlusCircle, Trash2, X } from 'lucide-react';

// --- TYPE DEFINITION ---
interface McpServer {
  id: string;
  address: string;
  type: 'sse' | 'streamable_http';
}

// --- SELF-CONTAINED UI COMPONENTS (No shadcn/ui needed) ---

const Button = ({ children, onClick, className = '', ...props }: { children: ReactNode; onClick?: () => void; className?: string; [key: string]: any; }) => (
  <button
    onClick={onClick}
    className={`inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none ring-offset-background ${className}`}
    {...props}
  >
    {children}
  </button>
);

const Input = ({ className = '', ...props }: { className?: string; [key: string]: any; }) => (
    <input
        className={`flex h-10 w-full rounded-md border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 focus:ring-offset-zinc-900 ${className}`}
        {...props}
    />
);

const Label = ({ children, ...props }: { children: ReactNode; [key: string]: any; }) => (
    <label className="text-sm font-medium leading-none text-zinc-400" {...props}>
        {children}
    </label>
);


// --- MCP SERVER MANAGER COMPONENT ---

export function McpServerManager() {
  const [isOpen, setIsOpen] = useState(false);
  const [mcpServers, setMcpServers] = useState<McpServer[]>([
    { id: '1', address: 'http://localhost:8081/mcp', type: 'sse' },
    { id: '2', address: 'http://localhost:8082/mcp', type: 'streamable_http' },
  ]);
  const [newServerAddress, setNewServerAddress] = useState('');
  const [newServerType, setNewServerType] = useState<'sse' | 'streamable_http'>('sse');

  const drawerRef = useRef<HTMLDivElement>(null);

  // Close drawer on escape key
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setIsOpen(false);
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Close drawer on outside click
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
        if (drawerRef.current && !drawerRef.current.contains(event.target as Node)) {
            setIsOpen(false);
        }
    };
    if (isOpen) {
        document.addEventListener('mousedown', handleClickOutside);
    }
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen]);


  const handleAddServer = () => {
    if (!newServerAddress.trim()) {
      alert("Server address cannot be empty.");
      return;
    }
    const newServer: McpServer = {
      id: Date.now().toString(),
      address: newServerAddress.trim(),
      type: newServerType,
    };
    setMcpServers(prev => [...prev, newServer]);
    setNewServerAddress(''); // Reset form
  };

  const handleRemoveServer = (idToRemove: string) => {
    setMcpServers(prev => prev.filter(server => server.id !== idToRemove));
  };

  return (
    <>
      {/* TRIGGER BUTTON */}
      <Button
        onClick={() => setIsOpen(true)}
        className="h-10 w-10 p-0 border border-zinc-700 bg-zinc-900 hover:bg-zinc-800 hover:text-zinc-100"
        title="Manage MCP Servers"
      >
        <Server className="h-5 w-5" />
        <span className="sr-only">Manage MCP Servers</span>
      </Button>

      {/* DRAWER / SIDE PANEL */}
      {isOpen && (
        <div className="fixed inset-0 z-50">
          {/* OVERLAY */}
          <div
            onClick={() => setIsOpen(false)}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm"
            aria-hidden="true"
          />

          {/* CONTENT */}
          <div
            ref={drawerRef}
            className="fixed top-0 right-0 h-full w-full max-w-md bg-zinc-950 border-l border-zinc-800 text-zinc-100 flex flex-col shadow-2xl"
          >
            {/* HEADER */}
            <div className="flex items-center justify-between p-6 border-b border-zinc-800">
                <div>
                    <h2 className="text-xl font-semibold text-zinc-100">Manage MCP Servers</h2>
                    <p className="text-sm text-zinc-400">Add or remove servers from the network.</p>
                </div>
                <Button onClick={() => setIsOpen(false)} className="h-8 w-8 p-0 bg-transparent hover:bg-zinc-800 text-zinc-400 hover:text-zinc-100">
                    <X className="h-5 w-5" />
                </Button>
            </div>

            {/* SERVER LIST */}
            <div className="flex-grow overflow-y-auto p-6">
              <div className="flex flex-col gap-3">
                {mcpServers.length > 0 ? mcpServers.map(server => (
                  <div key={server.id} className="flex items-center justify-between p-3 bg-zinc-900/70 border border-zinc-800 rounded-md">
                    <div className="flex flex-col">
                      <span className="font-mono text-sm text-zinc-300">{server.address}</span>
                      <span className="text-xs uppercase bg-zinc-700 text-zinc-300 px-2 py-0.5 rounded-full w-fit mt-1">
                        {server.type.replace('_', ' ')}
                      </span>
                    </div>
                    <Button onClick={() => handleRemoveServer(server.id)} className="h-8 w-8 p-0 text-zinc-400 hover:bg-red-900/50 hover:text-red-400">
                       <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                )) : (
                    <p className="text-zinc-500 text-center py-8">No MCP servers configured.</p>
                )}
              </div>
            </div>

            {/* ADD SERVER FORM */}
            <div className="flex-shrink-0 p-6 border-t border-zinc-800 bg-zinc-950">
                <h3 className="text-lg font-semibold text-zinc-200 mb-4">Add New Server</h3>
                <div className="grid gap-4">
                    <div className="grid w-full items-center gap-1.5">
                        <Label htmlFor="server-address">Server Address</Label>
                        <Input
                            type="text"
                            id="server-address"
                            placeholder="http://localhost:8080/mcp"
                            value={newServerAddress}
                            onChange={(e) => setNewServerAddress(e.target.value)}
                        />
                    </div>
                    <div className="grid w-full items-center gap-1.5">
                        <Label htmlFor="server-type">Server Type</Label>
                        <select
                            id="server-type"
                            value={newServerType}
                            onChange={(e) => setNewServerType(e.target.value as 'sse' | 'streamable_http')}
                            className="h-10 w-full rounded-md border border-zinc-700 bg-zinc-800 px-3 text-sm text-zinc-100 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 focus:ring-offset-zinc-900"
                        >
                            <option value="sse">SSE (Server-Sent Events)</option>
                            <option value="streamable_http">Streamable HTTP</option>
                        </select>
                    </div>
                </div>
                <Button onClick={handleAddServer} className="w-full mt-4 h-10 bg-indigo-600 hover:bg-indigo-700 text-white">
                    <PlusCircle className="mr-2 h-4 w-4" /> Add Server
                </Button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}