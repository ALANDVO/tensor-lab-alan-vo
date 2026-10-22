import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";
import { apiClient } from "./api/client";

vi.mock("./api/client", () => ({
  apiClient: {
    getHealth: vi.fn(),
  },
}));

describe("App", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders title and description", () => {
    render(<App />);
    expect(screen.getByText("Kronos Tensor Lab")).toBeInTheDocument();
    expect(screen.getByText("A Foundation Model for the Language of Financial Markets")).toBeInTheDocument();
  });

  it("shows loading state initially", () => {
    render(<App />);
    expect(screen.getByText("Checking backend connection...")).toBeInTheDocument();
  });

  it("displays healthy status when API returns ok", async () => {
    vi.mocked(apiClient.getHealth).mockResolvedValue({
      status: "healthy",
      service: "kronos-tensor-lab",
      version: "1.0.0",
    });

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText("Status:")).toBeInTheDocument();
      expect(screen.getByText("healthy")).toBeInTheDocument();
      expect(screen.getByText("Service: kronos-tensor-lab")).toBeInTheDocument();
      expect(screen.getByText("Version: 1.0.0")).toBeInTheDocument();
    });
  });

  it("displays error when API fails", async () => {
    vi.mocked(apiClient.getHealth).mockRejectedValue(new Error("Connection refused"));

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText("Error: Connection refused")).toBeInTheDocument();
    });
  });

  it("refresh button triggers new health check", async () => {
    vi.mocked(apiClient.getHealth)
      .mockResolvedValueOnce({ status: "healthy", service: "kronos-tensor-lab", version: "1.0.0" })
      .mockResolvedValueOnce({ status: "healthy", service: "kronos-tensor-lab", version: "1.0.0" });

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText("healthy")).toBeInTheDocument();
    });

    const button = screen.getByRole("button", { name: /refresh/i });
    await userEvent.click(button);

    await waitFor(() => {
      expect(apiClient.getHealth).toHaveBeenCalledTimes(2);
    });
  });
});