export interface SourceItem {
  rank: number;
  title: string;
  chunk_id: string;
  score: number;
}

export interface AskRequest {
  query: string;
  top_k: number;
  min_similarity: number;
  max_new_tokens: number;
}

export interface AskResponse {
  query: string;
  answer: string;
  passed: boolean;
  max_score: number;
  sources: SourceItem[];
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  status?: "loading" | "done" | "error" | "refused";
  result?: AskResponse;
  feedback?: "up" | "down" | null;
}

export interface KnowledgeDocument {
  id: string;
  name: string;
  type: "PDF" | "TXT" | "MD";
  size_bytes: number;
  updated_at: string;
  status: "ready" | "processing" | "failed";
  chunks: number;
  error?: string | null;
}

export interface DocumentListResponse {
  documents: KnowledgeDocument[];
  total: number;
  ready: number;
  chunks: number;
}
