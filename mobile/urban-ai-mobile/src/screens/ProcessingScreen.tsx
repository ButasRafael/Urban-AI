// src/screens/ProcessingScreen.tsx
import React, { useEffect, useRef, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text } from 'react-native';
import { StackNavigationProp } from '@react-navigation/stack';
import { RouteProp } from '@react-navigation/native';
import { LinearGradient } from 'expo-linear-gradient';

import { RootStackParamList } from '../navigation/types';
import { API_BASE } from '../config';
import client from '../api/client';
import { authService } from '../api/auth';

import { Box } from '../components/restylePrimitives';
import StyledButton from '../components/StyledButton';
import { useTheme } from '@shopify/restyle';
import { Theme } from '../theme';

type ProcessingScreenNavigationProp = StackNavigationProp<RootStackParamList, 'Processing'>;
type ProcessingScreenRouteProp = RouteProp<RootStackParamList, 'Processing'>;

interface Props {
  navigation: ProcessingScreenNavigationProp;
  route: ProcessingScreenRouteProp;
}

type TaskStatus = {
  task_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress?: number;
  error?: string;
};

type MediaResult = {
  media_id: number;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  media_type?: 'image' | 'video';
  annotated_image_url?: string;
  annotated_video_url?: string;
  thumbnail_url?: string;
  created_at?: string;
  address?: string;
  latitude?: number;
  longitude?: number;
  frames?: any[];
  detections?: any[];
  summary_description?: string;
  summary_solution?: string;
  error_message?: string;
};

const normalize = (s?: string): TaskStatus['status'] => {
  switch ((s || '').toUpperCase()) {
    case 'PENDING': return 'pending';
    case 'STARTED':
    case 'RETRY':
    case 'PROCESSING': return 'processing';
    case 'SUCCESS':
    case 'COMPLETED': return 'completed';
    case 'FAILURE':
    case 'FAILED': return 'failed';
    default: return 'processing';
  }
};

function clamp01(x: number) {
  if (Number.isNaN(x)) return 0;
  return Math.max(0, Math.min(1, x));
}

function useAnimatedProgress(value?: number, isDone?: boolean) {
  const [anim, setAnim] = useState(0);
  const raf = useRef<number | null>(null);
  const lastTs = useRef<number>(0);

  useEffect(() => {
    let target = typeof value === 'number' ? value / 100 : undefined;

    // cap at 0.95 until done for snappier finish
    if (target !== undefined && !isDone) target = Math.min(target, 0.95);
    if (isDone) target = 1;

    // unknown → gentle pulse between 0.08..0.18
    if (target === undefined && !isDone) {
      let dir = 1;
      const step = (ts: number) => {
        const dt = Math.min(32, ts - lastTs.current);
        lastTs.current = ts;
        setAnim(prev => {
          let next = prev + (dir * dt) / 3000; // slow pulse
          if (next > 0.18) { next = 0.18; dir = -1; }
          if (next < 0.08) { next = 0.08; dir = 1; }
          return next;
        });
        raf.current = requestAnimationFrame(step);
      };
      raf.current = requestAnimationFrame(step);
      return () => { if (raf.current) cancelAnimationFrame(raf.current); };
    }

    // known progress → ease towards target
    let current = anim;
    const ease = (t: number) => (1 - Math.pow(1 - t, 3));
    const run = (ts: number) => {
      if (lastTs.current === 0) lastTs.current = ts;
      const dt = Math.min(48, ts - lastTs.current);
      lastTs.current = ts;

      const t = clamp01(dt / 250); // ease window
      current = current + (ease(t) * (clamp01(target ?? 0) - current));
      setAnim(current);

      const done = Math.abs(current - clamp01(target ?? 0)) < 0.002;
      if (!done) raf.current = requestAnimationFrame(run);
    };
    raf.current = requestAnimationFrame(run);
    return () => { if (raf.current) cancelAnimationFrame(raf.current); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value, isDone]);

  return Math.round(anim * 100);
}

const StageChip = ({ label, active, done }: { label: string; active?: boolean; done?: boolean }) => (
  <Box
    px="s"
    py="xs"
    borderRadius="m"
    style={[
      styles.chip,
      {
        borderColor: done ? 'transparent' : 'rgba(255,255,255,0.25)',
        backgroundColor: done ? 'rgba(34,197,94,0.25)' : active ? 'rgba(255,255,255,0.12)' : 'rgba(255,255,255,0.06)',
      },
    ]}
  >
    <Text style={{ color: '#fff', fontWeight: '600', fontSize: 12 }}>{label}</Text>
  </Box>
);

const ProcessingScreen: React.FC<Props> = ({ navigation, route }) => {
  const { taskId, mediaId } = route.params;
  const theme = useTheme<Theme>();
  const hasTaskId = !!taskId;

  const [status, setStatus] = useState<TaskStatus>({ task_id: taskId || '', status: 'pending' });
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const watchdogRef = useRef<NodeJS.Timeout | null>(null);

  const animatedPct = useAnimatedProgress(status.progress, status.status === 'completed');

  useEffect(() => {
    let mounted = true;

    const armWatchdog = () => {
      if (watchdogRef.current) clearTimeout(watchdogRef.current);
      watchdogRef.current = setTimeout(() => {
        if (mounted && !pollIntervalRef.current) startPolling();
      }, 10000);
    };
    const disarmWatchdog = () => {
      if (watchdogRef.current) {
        clearTimeout(watchdogRef.current);
        watchdogRef.current = null;
      }
    };

    const connectWebSocket = async () => {
      // Skip WebSocket if no taskId available
      if (!hasTaskId) {
        startPollingByMedia();
        return;
      }

      try {
        const token = await authService.getToken();
        if (!token || !mounted) return;

        const wsUrl = `${API_BASE.replace(/^http/i, 'ws')}/ws/inference?token=${encodeURIComponent(token)}`;
        const ws = new WebSocket(wsUrl);
        wsRef.current = ws;

        ws.onopen = () => {
          ws.send(JSON.stringify({ type: 'subscribe', task_id: taskId }));
          armWatchdog();
        };

        ws.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data);
            if (message.type === 'task_update' && message.task_id === taskId) {
              const u = message.update || message;
              const s = normalize(u.status);
              const next: TaskStatus = {
                task_id: taskId,
                status: s,
                progress: typeof u.progress === 'number' ? u.progress : undefined,
                error: u.error,
              };

              if (mounted) {
                setStatus(next);
                disarmWatchdog();
                armWatchdog();

                if (s === 'completed') {
                  setTimeout(fetchResultAndNavigate, 250);
                } else if (s === 'failed') {
                  setError(next.error || 'Procesarea a eșuat.');
                }
              }
            }
          } catch {
            // ignore parse errors
          }
        };

        ws.onerror = () => startPolling();
        ws.onclose = () => {
          wsRef.current = null;
          disarmWatchdog();
          if (mounted && status.status !== 'completed' && status.status !== 'failed') startPolling();
        };
      } catch {
        startPolling();
      }
    };

    const startPollingByMedia = () => {
      // Fallback: Poll by media ID when taskId is not available
      if (pollIntervalRef.current) return;
      const poll = async () => {
        try {
          const { data } = await client.get<MediaResult>(`/infer/result/${mediaId}`);
          if (!mounted) return;

          // Map media processing status to task status
          let taskStatus: TaskStatus['status'] = 'pending';
          if (data.status === 'completed') taskStatus = 'completed';
          else if (data.status === 'failed') taskStatus = 'failed';
          else if (data.status === 'processing') taskStatus = 'processing';

          setStatus({
            task_id: '',
            status: taskStatus,
            error: data.error_message
          });

          if (taskStatus === 'completed') {
            stopPolling();
            setTimeout(fetchResultAndNavigate, 150);
          } else if (taskStatus === 'failed') {
            stopPolling();
            setError(data.error_message || 'Procesarea a eșuat.');
          }
        } catch {
          if (mounted) setError('Nu s-a putut obține statusul sarcinii.');
        }
      };
      poll();
      pollIntervalRef.current = setInterval(poll, 2000);
    };

    const startPolling = () => {
      if (!hasTaskId) {
        startPollingByMedia();
        return;
      }

      if (pollIntervalRef.current) return;
      const poll = async () => {
        try {
          const { data } = await client.get<{ task_id: string; status: string; progress?: number; error?: string }>(
            `/infer/status/${taskId}`
          );
          if (!mounted) return;
          const s = normalize(data.status);
          setStatus({ task_id: taskId, status: s, progress: data.progress, error: data.error });

          if (s === 'completed') {
            stopPolling();
            setTimeout(fetchResultAndNavigate, 150);
          } else if (s === 'failed') {
            stopPolling();
            setError(data.error || 'Procesarea a eșuat.');
          }
        } catch {
          if (mounted) setError('Nu s-a putut obține statusul sarcinii.');
        }
      };
      poll();
      pollIntervalRef.current = setInterval(poll, 2000);
    };

    const stopPolling = () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
    };

    const fetchResultAndNavigate = async () => {
      try {
        const { data } = await client.get<MediaResult>(`/infer/result/${mediaId}`);
        if (!mounted) return;

        const mediaForDetail: RootStackParamList['Detail']['media'] = {
          media_id: data.media_id,
          media_type: data.media_type ?? (data.annotated_video_url ? 'video' : 'image'),
          annotated_image_url: data.annotated_image_url,
          annotated_video_url: data.annotated_video_url,
          created_at: data.created_at ?? new Date().toISOString(),
          address: data.address || '',
          latitude: data.latitude,
          longitude: data.longitude,
          predicted_classes: [],
          descriptions: [],
          summary_description: data.summary_description,
          summary_solution: data.summary_solution,
        };

        navigation.replace('Detail', { media: mediaForDetail, showInfo: true });
      } catch {
        if (mounted) setError('Nu s-a putut încărca rezultatul procesării.');
      }
    };

    // Start appropriate monitoring method based on taskId availability
    if (hasTaskId) {
      connectWebSocket();
    } else {
      startPollingByMedia();
    }

    return () => {
      mounted = false;
      if (wsRef.current) { wsRef.current.close(); wsRef.current = null; }
      if (watchdogRef.current) { clearTimeout(watchdogRef.current); watchdogRef.current = null; }
      if (pollIntervalRef.current) { clearInterval(pollIntervalRef.current); pollIntervalRef.current = null; }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [taskId, mediaId, navigation]);

  const goBackground = () => navigation.navigate('Gallery');
  const tryAgain = () => navigation.goBack();

  const stage = status.status;
  const pct = typeof status.progress === 'number' ? status.progress : undefined;
  const shownPct = stage === 'completed' ? 100 : animatedPct;

  const subtitle =
    stage === 'pending'
      ? 'Pregătim fișierul tău...'
      : stage === 'processing'
      ? (typeof pct === 'number' ? `Se procesează... ${shownPct}%` : 'Se procesează conținutul tău...')
      : stage === 'completed'
      ? 'Gata! Se încarcă rezultatul...'
      : 'A apărut o eroare la procesare.';

  return (
    <Box flex={1} backgroundColor="background">
      {/* Header with gradient */}
      <Box height={160} overflow="hidden" borderBottomLeftRadius="l" borderBottomRightRadius="l">
        <LinearGradient
          colors={[theme.colors.primary500, theme.colors.primary300]}
          start={{ x: 0, y: 0 }} end={{ x: 1, y: 1 }}
          style={StyleSheet.absoluteFill}
        />
        <Box flex={1} px="l" py="l" justifyContent="flex-end">
          <Text style={{ color: '#fff', fontSize: 22, fontWeight: '700' }}>
            Procesăm încărcarea ta
          </Text>
          <Text style={{ color: '#ffffffcc', marginTop: 6 }}>
            Rămâi pe acest ecran sau continuă în fundal.
          </Text>
        </Box>
      </Box>

      {/* Card */}
      <Box mx="l" p="l" borderRadius="l" backgroundColor="card" style={[styles.cardShadow, { marginTop: -theme.spacing.l }]}>
        {/* Stages */}
        <Box flexDirection="row" justifyContent="space-between" mb="m">
          <StageChip label="Coada" active={stage === 'pending'} done={stage !== 'pending'} />
          <StageChip label="Analiză" active={stage === 'processing'} done={stage === 'completed'} />
          <StageChip label="Finalizare" active={stage === 'completed'} done={stage === 'completed'} />
        </Box>

        {/* Status */}
        <Box alignItems="center" mb="m">
          <ActivityIndicator size="large" color={theme.colors.primary500} />
          <Text style={{ marginTop: 10, fontSize: 16, fontWeight: '600', color: theme.colors.text }}>
            {subtitle}
          </Text>
          {stage === 'processing' && (
            <Text style={{ marginTop: 6, fontSize: 13, color: theme.colors.muted, textAlign: 'center' }}>
              Analiza poate dura câteva momente în funcție de dimensiunea fișierului.
            </Text>
          )}
        </Box>

        {/* Progress bar */}
        <Box mt="s" mb="m">
          <Box height={10} borderRadius="m" backgroundColor="surface0" overflow="hidden">
            <Box
              height={10}
              width={`${shownPct}%`}
              backgroundColor="primary500"
              borderRadius="m"
            />
          </Box>
          <Text style={{ marginTop: 6, fontSize: 12, color: theme.colors.muted, textAlign: 'right' }}>
            {typeof pct === 'number' ? `${shownPct}%` : 'Estimăm progresul...'}
          </Text>
        </Box>

        {/* Actions */}
        {!error ? (
          <Box flexDirection="row" columnGap="s">
            <StyledButton title="Continuă în fundal" variant="tonal" onPress={goBackground} flex={1} />
          </Box>
        ) : (
          <Box>
            <Text style={{ color: theme.colors.error, marginBottom: 12, textAlign: 'center' }}>
              {error}
            </Text>
            <Box flexDirection="row" columnGap="s">
              <StyledButton title="Încearcă din nou" onPress={tryAgain} flex={1} />
              <StyledButton title="Galerie" variant="ghost" onPress={goBackground} flex={1} />
            </Box>
          </Box>
        )}
      </Box>
    </Box>
  );
};

const styles = StyleSheet.create({
  chip: {
    borderWidth: 1,
  },
  cardShadow: {
    shadowColor: '#000',
    shadowOpacity: 0.06,
    shadowRadius: 12,
    shadowOffset: { width: 0, height: 6 },
    elevation: 2,
  },
});

export default ProcessingScreen;
