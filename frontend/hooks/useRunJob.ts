"use client";

import { useCallback } from "react";
import { useMutation } from "@tanstack/react-query";
import { createJob, deleteJob } from "@/lib/api";
import { errorMessage } from "@/lib/utils";
import { useAppStore } from "@/store/appStore";
import type { JobCreatedResponse } from "@/lib/types";

interface UseRunJob {
  run: () => void;
  cancel: () => void;
  submitting: boolean;
}

export function useRunJob(): UseRunJob {
  const setError = useAppStore((s) => s.setError);
  const startJob = useAppStore((s) => s.startJob);
  const clearJob = useAppStore((s) => s.clearJob);

  const mutation = useMutation<JobCreatedResponse, Error, void>({
    mutationFn: async () => {
      const { source, settings } = useAppStore.getState();
      if (!source) throw new Error("Load an image first.");
      return createJob({ file: source.file, ...settings });
    },
    onSuccess: (data) => startJob(data.job_id),
    onError: (err) => setError(errorMessage(err)),
  });

  const { mutate } = mutation;

  const run = useCallback(() => {
    const { source, job } = useAppStore.getState();
    if (!source) {
      setError("Load an image first.");
      return;
    }
    // Best-effort cleanup of the previous job's server-side artifacts.
    if (job) void deleteJob(job.id).catch(() => undefined);
    setError(null);
    mutate();
  }, [mutate, setError]);

  const cancel = useCallback(() => {
    const { job } = useAppStore.getState();
    if (!job) return;
    void deleteJob(job.id).catch(() => undefined);
    clearJob();
  }, [clearJob]);

  return { run, cancel, submitting: mutation.isPending };
}
