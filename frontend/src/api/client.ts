const API_BASE = "/api/v1";

export interface HealthResponse {
  status: string;
  service?: string;
  version?: string;
}

export interface TokenizeRequest {
  ohlcv: number[][][];
}

export interface TokenizeResponse {
  token_ids: number[][];
  shape: number[];
}

export interface DetokenizeRequest {
  token_ids: number[][];
}

export interface DetokenizeResponse {
  ohlcv: number[][][];
  shape: number[];
}

export interface ApiError {
  detail: string;
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
      ...options,
    });

    if (!response.ok) {
      const error: ApiError = await response.json().catch(() => ({
        detail: `HTTP ${response.status}`,
      }));
      throw new Error(error.detail);
    }

    return response.json();
  }

  async getHealth(): Promise<HealthResponse> {
    return this.request<HealthResponse>("/health");
  }

  async tokenize(request: TokenizeRequest): Promise<TokenizeResponse> {
    return this.request<TokenizeResponse>("/tokenizer/tokenize", {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  async detokenize(request: DetokenizeRequest): Promise<DetokenizeResponse> {
    return this.request<DetokenizeResponse>("/tokenizer/detokenize", {
      method: "POST",
      body: JSON.stringify(request),
    });
  }
}

export const apiClient = new ApiClient();