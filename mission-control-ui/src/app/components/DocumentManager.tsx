// mission-control-ui/src/app/components/DocumentManager.tsx
"use client";

import { useState, useRef, useEffect, ReactNode } from "react";
import {
  FileText,
  PlusCircle,
  Trash2,
  X,
  Save,
  ArrowLeft,
  Loader2,
  AlertTriangle,
  Pencil,
} from "lucide-react";
import {
  Document,
  fetchDocuments,
  addDocument,
  updateDocument,
  deleteDocument,
} from "@/lib/api"; // Adjust this path to your api.ts file

// --- UI Components ---
const Input = ({
  className = "",
  ...props
}: {
  className?: string;
  [key: string]: any;
}) => (
  <input
    className={`flex h-10 w-full rounded-md border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 focus:ring-offset-zinc-900 ${className}`}
    {...props}
  />
);

const Textarea = ({
  className = "",
  ...props
}: {
  className?: string;
  [key: string]: any;
}) => (
  <textarea
    className={`flex min-h-[250px] w-full rounded-md border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 focus:ring-offset-zinc-900 ${className}`}
    {...props}
  />
);

const Label = ({
  children,
  ...props
}: {
  children: ReactNode;
  [key: string]: any;
}) => (
  <label className="text-sm font-medium leading-none text-zinc-400" {...props}>
    {children}
  </label>
);

const baseButtonClasses =
  "inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 disabled:opacity-50 disabled:pointer-events-none";

// --- Main Component ---
export function DocumentManager() {
  const [isOpen, setIsOpen] = useState(false);
  const [isVisible, setIsVisible] = useState(false);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [view, setView] = useState<"list" | "editor">("list");
  const [currentDocument, setCurrentDocument] = useState<Document | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const drawerRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const TRANSITION_DURATION = 300;

  const loadDocuments = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const docs = await fetchDocuments();
      setDocuments(docs || []);
    } catch (err) {
      setError("Failed to load documents.");
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const openDrawer = () => {
    setIsOpen(true);
    setTimeout(() => setIsVisible(true), 10);
  };

  const closeDrawer = () => {
    setIsVisible(false);
    setTimeout(() => {
      setIsOpen(false);
      setView("list"); // Reset view on close
      setCurrentDocument(null);
    }, TRANSITION_DURATION);
  };

  // Effect to load documents when drawer opens
  useEffect(() => {
    if (isOpen && view === "list") {
      loadDocuments();
    }
  }, [isOpen, view]);

  // Effects for keyboard shortcuts and clicking outside
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") closeDrawer();
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
    if (isOpen) document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [isOpen]);

  const handleNewDocument = () => {
    setCurrentDocument({ id: "", name: "", content: "" });
    setView("editor");
  };

  const handleEditDocument = (doc: Document) => {
    setCurrentDocument(doc);
    setView("editor");
  };

  const handleDeleteDocument = async (docId: string) => {
    if (window.confirm("Are you sure you want to delete this document?")) {
      await deleteDocument(docId);
      loadDocuments(); // Refresh list
    }
  };

  const handleSaveDocument = async () => {
    if (!currentDocument || !currentDocument.name.trim()) {
      alert("Document name cannot be empty.");
      return;
    }
    setIsLoading(true);
    try {
      if (currentDocument.id) {
        await updateDocument(currentDocument);
      } else {
        await addDocument({ ...currentDocument, id: Date.now().toString() });
      }
      setView("list");
    } catch (err) {
      setError("Failed to save document.");
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const renderListView = () => (
    <>
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold text-zinc-200">
          All Documents
        </h3>
        <button
          onClick={handleNewDocument}
          className={`${baseButtonClasses} px-3 py-1 text-sm bg-indigo-600 hover:bg-indigo-700 text-white`}
        >
          <PlusCircle className="mr-2 h-4 w-4" /> New Document
        </button>
      </div>
      {isLoading ? (
        <div className="flex justify-center items-center h-40">
          <Loader2 className="h-8 w-8 animate-spin text-zinc-500" />
        </div>
      ) : error ? (
        <div className="text-center py-8 text-red-400">
          <AlertTriangle className="mx-auto h-8 w-8 mb-2" /> {error}
        </div>
      ) : documents.length > 0 ? (
        <div className="flex flex-col gap-3">
          {documents.map((doc) => (
            <div
              key={doc.id}
              className="flex items-center justify-between p-3 bg-zinc-900/70 border border-zinc-800 rounded-md"
            >
              <span className="font-mono text-sm text-zinc-300 truncate">
                {doc.name}
              </span>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => handleEditDocument(doc)}
                  className={`${baseButtonClasses} h-8 w-8 p-0 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100`}
                  title="Edit"
                >
                  <Pencil className="h-4 w-4" />
                </button>
                <button
                  onClick={() => handleDeleteDocument(doc.id)}
                  className={`${baseButtonClasses} h-8 w-8 p-0 text-zinc-400 hover:bg-red-900/50 hover:text-red-400`}
                  title="Delete"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-zinc-500 text-center py-8">No documents found.</p>
      )}
    </>
  );

  const renderEditorView = () =>
    currentDocument && (
      <div className="flex flex-col h-full">
        <div className="flex-shrink-0 grid gap-4">
          <div className="grid w-full items-center gap-1.5">
            <Label htmlFor="doc-name">Document Name</Label>
            <Input
              id="doc-name"
              placeholder="My Document"
              value={currentDocument.name}
              onChange={(e) =>
                setCurrentDocument({ ...currentDocument, name: e.target.value })
              }
            />
          </div>
          <div className="grid w-full items-center gap-1.5">
            <Label htmlFor="doc-content">Content</Label>
            <Textarea
              id="doc-content"
              placeholder="Type your content here..."
              value={currentDocument.content}
              onChange={(e) =>
                setCurrentDocument({
                  ...currentDocument,
                  content: e.target.value,
                })
              }
            />
          </div>
        </div>
        <div className="flex-grow"></div>
        <div className="flex-shrink-0 mt-4">
          <button
            onClick={handleSaveDocument}
            disabled={isLoading}
            className={`${baseButtonClasses} w-full h-10 bg-indigo-600 hover:bg-indigo-700 text-white`}
          >
            {isLoading ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <Save className="mr-2 h-4 w-4" />
            )}
            Save Document
          </button>
        </div>
      </div>
    );

  return (
    <>
      <button
        ref={triggerRef}
        onClick={openDrawer}
        className={`${baseButtonClasses} h-10 w-10 p-0 border border-zinc-700 bg-zinc-900 hover:bg-zinc-800 hover:text-zinc-100`}
        title="Manage Documents"
      >
        <FileText className="h-5 w-5" />
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
                {view === "list" ? (
                  <h2 className="text-xl font-semibold text-zinc-100">
                    Manage Documents
                  </h2>
                ) : (
                  <button
                    onClick={() => setView("list")}
                    className={`${baseButtonClasses} text-xl font-semibold text-zinc-100 hover:text-indigo-400`}
                  >
                    <ArrowLeft className="mr-3 h-5 w-5" />
                    {currentDocument?.id ? "Edit Document" : "New Document"}
                  </button>
                )}
              </div>
              <button
                onClick={closeDrawer}
                className={`${baseButtonClasses} h-8 w-8 p-0 bg-transparent hover:bg-zinc-800 text-zinc-400 hover:text-zinc-100`}
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            <div className="flex-grow overflow-y-auto p-6">
              {view === "list" ? renderListView() : renderEditorView()}
            </div>
          </div>
        </div>
      )}
    </>
  );
}