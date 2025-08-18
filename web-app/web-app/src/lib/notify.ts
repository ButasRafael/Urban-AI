// src/lib/notify.ts
import { toast } from 'sonner';

export const notify = {
  success: (title: string, desc?: string) =>
    toast.success(title, { description: desc }),
  error: (title: string, desc?: string) =>
    toast.error(title, { description: desc }),
  info: (title: string, desc?: string) =>
    toast(title, { description: desc }),
  promise: <T>(
    p: Promise<T>,
    msgs: { loading?: string; success?: string | ((v: T) => string); error?: string | ((e: unknown) => string) }
  ) =>
    toast.promise(p, {
      loading: msgs.loading ?? 'Working…',
      success: (v) =>
        typeof msgs.success === 'function' ? msgs.success(v) : msgs.success ?? 'Done',
      error: (e) =>
        typeof msgs.error === 'function' ? msgs.error(e) : msgs.error ?? 'Something went wrong',
    }),
};
