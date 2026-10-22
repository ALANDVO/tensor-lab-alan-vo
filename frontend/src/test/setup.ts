import { vi } from "vitest";
import "@testing-library/jest-dom";

vi.mock("@/api/client", () => ({
  apiClient: {
    get: vi.fn(),
    post: vi.fn(),
  },
}));