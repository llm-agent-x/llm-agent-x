// mission-control-ui/src/app/components/McpServerManager.tsx
"use client";

import { useState, useRef, useEffect, ReactNode } from 'react';
// Import the new RotateCcw icon for the reset button
import { Server, PlusCircle, Trash2, X, RotateCcw } from 'lucide-react';

// --- TYPE DEFINITION ---
interface McpServer {
  id: string;
  address: string;
  type: 'sse' | 'streamable_http';
}

// --- 1. DEFINE DEFAULT STATE ---
const DEFAULT_SERVERS: McpServer[] = [
  { id: '1', address: 'http://localhost:8081/mcp', type: 'sse' },
  { id: '2', address: 'http://localhost:8082/mcp', type: 'streamable_http' },
];

const LOCAL_STORAGE_KEY = 'mcpServers';

// --- SELF-CONTAINED UI COMPONENTS (No changes here) ---
const Button = ({ children, onClick, className = '', ...props }: { children: ReactNode; onClick?: (e: React.MouseEvent<HTMLButtonElement>) => void; className?: string; [key: string]: any; }) => (
  <button
    onClick={onClick}
    className={`inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none ring-offset-background ${className}`}
    {...props}
  >
    {children}
  </button>
);
// ... (Input and Label components remain the same)
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
  const [isVisible, setIsVisible] = useState(false);

  // --- 2. LOAD FROM LOCALSTORAGE ON INIT ---
  // Use a function in useState to lazily initialize the state from localStorage
  const [mcpServers, setMcpServers] = useState<McpServer[]>(() => {
    try {
      const saved = window.localStorage.getItem(LOCAL_STORAGE_KEY);
      if (saved) {
        // Ensure the parsed data is an array before returning
        const parsed = JSON.parse(saved);
        return Array.isArray(parsed) ? parsed : DEFAULT_SERVERS;
      }
    } catch (error) {
      console.error("Failed to parse MCP servers from localStorage:", error);
    }
    return DEFAULT_SERVERS;
  });

  // --- 3. SAVE TO LOCALSTORAGE ON CHANGE ---
  // This effect runs whenever the mcpServers state changes
  useEffect(() => {
    try {
      window.localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(mcpServers));
    } catch (error) {
        console.error("Failed to save MCP servers to localStorage:", error);
    }
  }, [mcpServers]);


  const [newServerAddress, setNewServerAddress] = useState('');
  const [newServerType, setNewServerType] = useState<'sse' | 'streamable_http'>('sse');
  const drawerRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const TRANSITION_DURATION = 300;

  // ... (Animation and event listener logic remains the same)
  const openDrawer = () => { setIsOpen(true); setTimeout(() => { setIsVisible(true); }, 10); };
  const closeDrawer = () => { setIsVisible(false); setTimeout(() => { setIsOpen(false); }, TRANSITION_DURATION); };
  useEffect(() => { const handleKeyDown = (event: KeyboardEvent) => { if (event.key === 'Escape') { closeDrawer(); } }; document.addEventListener('keydown', handleKeyDown); return () => document.removeEventListener('keydown', handleKeyDown); }, []);
  useEffect(() => { const handleClickOutside = (event: MouseEvent) => { if (drawerRef.current && !drawerRef.current.contains(event.target as Node) && triggerRef.current && !triggerRef.current.contains(event.target as Node)) { closeDrawer(); } }; if (isOpen) { document.addEventListener('mousedown', handleClickOutside); } return () => document.removeEventListener('mousedown', handleClickOutside); }, [isOpen]);


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
    setNewServerAddress('');
  };

  const handleRemoveServer = (idToRemove: string) => {
    setMcpServers(prev => prev.filter(server => server.id !== idToRemove));
  };

  // --- 4. ADD A RESET HANDLER ---
  const handleResetServers = () => {
      if(window.confirm("Are you sure you want to reset the server list to the default configuration?")) {
          setMcpServers(DEFAULT_SERVERS);
      }
  };


  return (
    <>
      <Button
        ref={triggerRef}
        onClick={openDrawer}
        className="h-10 w-10 p-0 border border-zinc-700 bg-zinc-900 hover:bg-zinc-800 hover:text-zinc-100"
        title="Manage MCP Servers"
      >
        <Server className="h-5 w-5" />
      </Button>
      {isOpen && (
        <div className="fixed inset-0 z-50">
          <div onClick={closeDrawer} className={`fixed inset-0 bg-black/60 backdrop-blur-sm transition-opacity duration-${TRANSITION_DURATION} ease-in-out ${isVisible ? 'opacity-100' : 'opacity-0'}`} />
          <div ref={drawerRef} className={`fixed top-0 right-0 h-full w-full max-w-md bg-zinc-950 border-l border-zinc-800 text-zinc-100 flex flex-col shadow-2xl transition-transform duration-${TRANSITION_DURATION} ease-in-out ${isVisible ? 'translate-x-0' : 'translate-x-full'}`}>
            <div className="flex items-center justify-between p-6 border-b border-zinc-800">
                <div>
                    <h2 className="text-xl font-semibold text-zinc-100">Manage MCP Servers</h2>
                    <p className="text-sm text-zinc-400">Add or remove servers from the network.</p>
                </div>
                <Button onClick={closeDrawer} className="h-8 w-8 p-0 bg-transparent hover:bg-zinc-800 text-zinc-400 hover:text-zinc-100"> <X className="h-5 w-5" /> </Button>
            </div>
            <div className="flex-grow overflow-y-auto p-6">
                {/* ... (server list mapping code remains the same) */}
                <div className="flex flex-col gap-3">
                    {mcpServers.map(server => (
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
                    ))}
                    {mcpServers.length === 0 && (
                        <p className="text-zinc-500 text-center py-8">No MCP servers configured.</p>
                    )}
                </div>
            </div>
            <div className="flex-shrink-0 p-6 border-t border-zinc-800 bg-zinc-950">
                <div className="flex justify-between items-center mb-4">
                    <h3 className="text-lg font-semibold text-zinc-200">Add New Server</h3>
                    {/* --- 4. ADD RESET BUTTON TO UI --- */}
                    <Button onClick={handleResetServers} className="px-3 py-1 text-xs border border-zinc-700 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100">
                        <RotateCcw className="mr-2 h-3 w-3" />
                        Reset Defaults
                    </Button>
                </div>
                <div className="grid gap-4">
                    <div className="grid w-full items-center gap-1.5">
                        <Label htmlFor="server-address">Server Address</Label>
                        <Input type="text" id="server-address" placeholder="http://localhost:8080/mcp" value={newServerAddress} onChange={(e) => setNewServerAddress(e.target.value)} />
                    </div>
                    <div className="grid w-full items-center gap-1.5">
                        <Label htmlFor="server-type">Server Type</Label>
                        <select id="server-type" value={newServerType} onChange={(e) => setNewServerType(e.target.value as 'sse' | 'streamable_http')} className="h-10 w-full rounded-md border border-zinc-700 bg-zinc-800 px-3 text-sm text-zinc-100 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 focus:ring-offset-zinc-900">
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