// mission-control-ui/src/app/layout.tsx
"use client";

import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { ReactFlowProvider } from "@reactflow/core"; // Import ReactFlow

const inter = Inter({ subsets: ["latin"] });

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${inter.className} bg-zinc-900 text-zinc-50 vsc-initialized`}
      >
        <ReactFlowProvider>
          {" "}
          {/* Wrap children with ReactFlowProvider */}
          {children}
        </ReactFlowProvider>
      </body>
    </html>
  );
}
