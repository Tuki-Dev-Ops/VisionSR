"use client";

import { useQuery, type UseQueryResult } from "@tanstack/react-query";
import { getHealth, getModels } from "@/lib/api";
import type { HealthResponse, ModelInfo } from "@/lib/types";

/** Polls /health so the UI can tell the user the backend is down, not "loading". */
export function useHealth(): UseQueryResult<HealthResponse, Error> {
  return useQuery({
    queryKey: ["health"],
    queryFn: ({ signal }) => getHealth(signal),
    refetchInterval: 15_000,
    refetchOnWindowFocus: true,
    retry: 1,
    staleTime: 5_000,
  });
}

export function useModels(): UseQueryResult<ModelInfo[], Error> {
  return useQuery({
    queryKey: ["models"],
    queryFn: async ({ signal }) => (await getModels(signal)).models,
    staleTime: 60_000,
    retry: 1,
  });
}
