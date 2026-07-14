"use client";

import { useEffect } from "react";
import { getJob, getJobResult, subscribeToJobEvents } from "@/lib/api";
import { errorMessage } from "@/lib/utils";
import { useAppStore } from "@/store/appStore";

const POLL_INTERVAL_MS = 500;
/** If SSE connects but sends nothing for this long, assume it's dead (proxy
 *  buffering, stalled worker) and switch to polling rather than spin forever. */
const SSE_SILENCE_TIMEOUT_MS = 10_000;
/** Consecutive poll failures tolerated before we declare the job lost. */
const MAX_POLL_FAILURES = 6;

/**
 * Drives live job progress for the currently active job.
 *
 * Primary transport is the SSE stream; it falls back to polling
 * GET /api/v1/jobs/{id} every 500ms if the stream errors, never opens, or goes
 * silent. Terminal status always triggers a full GET so we pick up `result`,
 * followed by a fetch of the result bytes into an object URL.
 */
export function useJobProgress(): void {
  const jobId = useAppStore((s) => s.job?.id ?? null);
  const applyJobEvent = useAppStore((s) => s.applyJobEvent);
  const applyJob = useAppStore((s) => s.applyJob);
  const setTransport = useAppStore((s) => s.setTransport);
  const setResultUrl = useAppStore((s) => s.setResultUrl);
  const failJob = useAppStore((s) => s.failJob);

  useEffect(() => {
    if (!jobId) return;

    const controller = new AbortController();
    let disposed = false;
    let finalizing = false;
    let polling = false;
    let failures = 0;
    let unsubscribe: (() => void) | null = null;
    let pollTimer: ReturnType<typeof setInterval> | null = null;
    let watchdog: ReturnType<typeof setTimeout> | null = null;

    const clearWatchdog = (): void => {
      if (watchdog !== null) {
        clearTimeout(watchdog);
        watchdog = null;
      }
    };

    const stopPolling = (): void => {
      if (pollTimer !== null) {
        clearInterval(pollTimer);
        pollTimer = null;
      }
    };

    const closeStream = (): void => {
      unsubscribe?.();
      unsubscribe = null;
    };

    const isTerminal = (status: string): boolean =>
      status === "done" || status === "failed";

    /** Terminal status reached: pull the authoritative job, then the bytes. */
    const finalize = async (): Promise<void> => {
      if (disposed || finalizing) return;
      finalizing = true;
      clearWatchdog();
      closeStream();
      stopPolling();

      try {
        const job = await getJob(jobId, controller.signal);
        if (disposed) return;
        applyJob(job);

        if (job.status === "done") {
          const blob = await getJobResult(jobId, controller.signal);
          if (disposed) return;
          setResultUrl(URL.createObjectURL(blob));
        } else if (job.status === "failed") {
          failJob(job.error ?? "The job failed without an error message.");
        } else {
          // Race: the stream said terminal but the job isn't. Resume polling.
          finalizing = false;
          polling = false;
          startPolling();
        }
      } catch (err) {
        if (disposed) return;
        failJob(errorMessage(err));
      }
    };

    const tick = async (): Promise<void> => {
      if (disposed || finalizing) return;
      try {
        const job = await getJob(jobId, controller.signal);
        if (disposed) return;
        failures = 0;
        applyJob(job);
        if (isTerminal(job.status)) void finalize();
      } catch (err) {
        if (disposed) return;
        failures += 1;
        if (failures >= MAX_POLL_FAILURES) {
          stopPolling();
          failJob(
            `Lost contact with the backend while the job was running. ${errorMessage(err)}`,
          );
        }
      }
    };

    function startPolling(): void {
      if (disposed || polling || finalizing) return;
      polling = true;
      failures = 0;
      clearWatchdog();
      closeStream();
      setTransport("polling");
      void tick();
      pollTimer = setInterval(() => void tick(), POLL_INTERVAL_MS);
    }

    const armWatchdog = (): void => {
      clearWatchdog();
      if (polling) return;
      watchdog = setTimeout(startPolling, SSE_SILENCE_TIMEOUT_MS);
    };

    setTransport("sse");
    unsubscribe = subscribeToJobEvents(jobId, {
      onEvent: (event) => {
        if (disposed || polling) return;
        armWatchdog();
        applyJobEvent(event);
        if (isTerminal(event.status)) void finalize();
      },
      onError: () => {
        if (disposed || finalizing) return;
        startPolling();
      },
    });
    armWatchdog();

    return () => {
      disposed = true;
      clearWatchdog();
      stopPolling();
      closeStream();
      controller.abort();
    };
  }, [jobId, applyJobEvent, applyJob, setTransport, setResultUrl, failJob]);
}
