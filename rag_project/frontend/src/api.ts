import type { AskRequest, AskResponse, DocumentListResponse } from "./types";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(message: string, public status?: number) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: { Accept: "application/json", ...init.headers },
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    throw new ApiError(payload?.detail || `请求失败（${response.status}）`, response.status);
  }
  return payload as T;
}

export const api = {
  health: (signal?: AbortSignal) =>
    request<{ status: string; project: string; pipeline_loaded: boolean }>("/health", { signal }),
  ask: (body: AskRequest, signal?: AbortSignal) =>
    request<AskResponse>("/ask", {
      method: "POST",
      signal,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  documents: () => request<DocumentListResponse>("/documents"),
  uploadDocument: (file: File) => {
    const body = new FormData();
    body.append("file", file);
    return request<{ message: string; document_id: string }>("/documents/upload", {
      method: "POST",
      body,
    });
  },
  reparseDocument: (documentId: string) =>
    request<{ message: string }>(`/documents/${encodeURIComponent(documentId)}/reparse`, { method: "POST" }),
  deleteDocument: (documentId: string) =>
    request<{ message: string }>(`/documents/${encodeURIComponent(documentId)}`, { method: "DELETE" }),
};
